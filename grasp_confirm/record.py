"""Run one recording session: cues, beeps, continuous EMG capture.

    # dry run -- no hardware, fake data, proves the whole pipeline works
    python record.py --subject subj01 --session sess01 --recorder synthetic

    # real session
    python record.py --subject subj01 --session sess01 \
        --recorder delsys --recorder-arg sdk-path=/path/to/Example-Applications/Python

What it writes (see README for the full schema):

    data/<subject>/<session>/
        session.json   metadata, channel names, sample rate, electrode notes
        schedule.csv   the planned trial order (generated here if absent)
        cues.csv       every cue event with its ACTUAL perf_counter timestamp
        raw.npz        the continuous recording -- the source of truth
        report.csv     per-trial self-report flags

Design note: the continuous stream is saved whole and segmented offline by
dataset.py using cues.csv. Nothing is cut at record time. If you later decide
the hold window should start at 3.8 s instead of 3.6 s you change timeline.py
and re-segment; you never have to re-record.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grasp_confirm import timeline  # noqa: E402
from grasp_confirm.recorders import make_recorder  # noqa: E402

HERE = Path(__file__).resolve().parent

EFFORT_GAIN = {"none": 0.0, "light": 0.5, "hard": 2.0, "normal": 1.0, "firm": 1.5}

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
RED, GREEN, YELLOW, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"


# ---------------------------------------------------------------- audio


def _write_tone(path: Path, freq: float, seconds: float = 0.12, rate: int = 44100) -> None:
    n = int(rate * seconds)
    fade = int(rate * 0.008)
    frames = bytearray()
    for i in range(n):
        env = min(1.0, i / fade, (n - i) / fade)
        value = int(0.6 * env * 32767 * math.sin(2 * math.pi * freq * i / rate))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


class Beeper:
    """Pre-rendered cue tones played through the OS audio player.

    Playback latency (spawning afplay/aplay) is tens of milliseconds and is NOT
    compensated. That is tolerable here only because every label window starts
    at least 300 ms after the beep that precedes it. Do not shrink those gaps
    without switching to a pre-opened audio stream.
    """

    PLAYERS = ("afplay", "aplay", "paplay")

    def __init__(self, directory: Path, enabled: bool = True):
        self.enabled = enabled
        self.player = next(
            (p for p in self.PLAYERS if shutil.which(p)), None
        ) if enabled else None
        self.paths: dict[str, Path] = {}
        if not self.enabled:
            return
        directory.mkdir(parents=True, exist_ok=True)
        for name, freq in timeline.BEEP_TONES.items():
            path = directory / f"beep_{name}.wav"
            if not path.exists():
                _write_tone(path, freq)
            self.paths[name] = path
        if self.player is None:
            print(f"{YELLOW}no audio player found; falling back to terminal bell{RESET}")

    def warm_up(self) -> None:
        """Spawn the player once so the first real beep is not the slow one."""
        if self.player and self.paths:
            self.play(next(iter(self.paths)))
            time.sleep(0.3)

    def play(self, name: str) -> None:
        if not self.enabled:
            return
        path = self.paths.get(name)
        if self.player and path:
            subprocess.Popen(
                [self.player, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()


# ---------------------------------------------------------------- session


class SessionRecorder:
    def __init__(self, recorder, beeper: Beeper, session_dir: Path):
        self.recorder = recorder
        self.beeper = beeper
        self.session_dir = session_dir
        self._t: list[np.ndarray] = []
        self._x: list[np.ndarray] = []
        self.cues: list[dict] = []
        self.marks: list[dict] = []

    # -- streaming --------------------------------------------------
    def pump(self) -> None:
        t, x = self.recorder.poll()
        if t.size:
            self._t.append(np.asarray(t, dtype=np.float64))
            self._x.append(np.asarray(x, dtype=np.float32))

    def wait_until(self, t_abs: float) -> None:
        while True:
            self.pump()
            remaining = t_abs - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(0.004, remaining))

    def collect_for(self, seconds: float, label: str) -> tuple[float, float]:
        t0 = time.perf_counter()
        self.wait_until(t0 + seconds)
        t1 = time.perf_counter()
        self.marks.append({"label": label, "t_start": t0, "t_end": t1})
        return t0, t1

    def log_cue(self, trial: int, event: str, planned: float, actual: float, t0: float) -> None:
        self.cues.append(
            {
                "trial": trial,
                "event": event,
                "t_perf": actual,
                "t_rel": actual - t0,
                "t_planned_rel": planned,
                "jitter_ms": (actual - t0 - planned) * 1000.0,
            }
        )

    # -- persistence ------------------------------------------------
    def concatenated(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._t:
            return np.zeros(0), np.zeros((0, len(self.recorder.channel_names)), np.float32)
        return np.concatenate(self._t), np.concatenate(self._x, axis=0)

    def save(self) -> None:
        t, x = self.concatenated()
        np.savez_compressed(self.session_dir / "raw.npz", t=t, x=x)
        with (self.session_dir / "cues.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["trial", "event", "t_perf", "t_rel", "t_planned_rel", "jitter_ms"],
            )
            writer.writeheader()
            writer.writerows(self.cues)
        with (self.session_dir / "marks.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["label", "t_start", "t_end"])
            writer.writeheader()
            writer.writerows(self.marks)


# ---------------------------------------------------------------- display


def banner(text: str, colour: str = CYAN, sub: str = "") -> None:
    print("\033[2J\033[H", end="")
    line = "=" * 64
    print(f"{colour}{line}")
    print(f"{BOLD}{text.center(64)}{RESET}")
    print(f"{colour}{line}{RESET}")
    if sub:
        print(f"\n  {sub}\n")


def prompt(message: str) -> str:
    try:
        return input(message)
    except EOFError:
        return ""


# ---------------------------------------------------------------- stages


def preflight(session: SessionRecorder, seconds: float = 5.0) -> dict:
    banner("PREFLIGHT", CYAN, "Relax the arm. Checking all channels for 5 s.")
    before = len(session._x)
    session.collect_for(seconds, "preflight")
    chunks = session._x[before:]
    names = session.recorder.channel_names
    if not chunks:
        raise SystemExit("no samples received during preflight -- check the device")
    x = np.concatenate(chunks, axis=0)
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=0))

    print(f"  {len(x)} samples, {len(names)} channels\n")
    problems = []
    for name, value in zip(names, rms):
        flag = ""
        if value < 1e-6:
            flag, problems = f"{RED}FLAT -- check the connection{RESET}", problems + [name]
        elif value > 0.5:
            flag, problems = f"{RED}VERY HIGH -- saturated or noisy{RESET}", problems + [name]
        print(f"    {name:10s} rest RMS {value:.6g}   {flag}")
    if problems:
        print(f"\n{RED}Problem channels: {', '.join(problems)}{RESET}")
    print(
        "\n  Expect a small non-zero rest RMS on every channel, of similar order.\n"
        "  Now ask the subject to wiggle their fingers and re-run if any channel is silent."
    )
    if prompt("\n  Enter to continue, 'q' to abort: ").strip().lower() == "q":
        raise SystemExit("aborted at preflight")
    return {"rest_rms": rms.tolist(), "problem_channels": problems}


def baseline_and_mvc(session: SessionRecorder, baseline_s: float, mvc_s: float, mvc_reps: int) -> None:
    banner("BASELINE", CYAN, f"Hand open and completely relaxed on the pad for {baseline_s:.0f} s.")
    prompt("  Enter when the subject is settled: ")
    session.beeper.play("reach")
    session.collect_for(baseline_s, "baseline")

    for rep in range(1, mvc_reps + 1):
        banner(f"MVC {rep}/{mvc_reps}", YELLOW, f"MAXIMUM grip on the beep, hold {mvc_s:.0f} s.")
        prompt("  Enter when ready: ")
        session.beeper.play("close")
        session.collect_for(mvc_s, f"mvc_{rep}")
        session.beeper.play("open")
        if rep < mvc_reps:
            banner("REST", DIM, "20 s rest before the next MVC.")
            session.collect_for(20.0, f"mvc_rest_{rep}")


def run_trial(session: SessionRecorder, trial: dict, objects: dict) -> None:
    """One trial: position the open hand, act on GO, relax on STOP."""
    spec = objects["objects"][trial["object_id"]]
    display = spec.get("display", trial["object_id"]).upper()
    if trial["family"] == "NOTHING":
        display = "NOTHING"
    effort = EFFORT_GAIN.get(trial["effort"], 1.0)
    trial_class = timeline.CLASS_OF_FAMILY[trial["family"]]

    events = (
        (timeline.CUE_S, "cue"),
        (timeline.GO_S, "go"),
        (timeline.STOP_S, "stop"),
    )

    t0 = time.perf_counter()
    for planned, event in events:
        session.wait_until(t0 + planned)
        now = time.perf_counter()
        session.log_cue(int(trial["trial"]), event, planned, now, t0)

        if event == "cue":
            colour = {"GRASP": GREEN, "AIR": YELLOW, "NOTHING": DIM}[trial["family"]]
            banner(
                f"[{trial['trial']}] {display}",
                colour,
                f"{trial['instruction']}\n  {DIM}condition {trial['condition']}{RESET}",
            )
        else:
            session.beeper.play(event)
            print(f"    {event.upper()}")

        if hasattr(session.recorder, "set_state"):
            if event == "go":
                session.recorder.set_state(trial_class, effort)
            elif event == "stop":
                session.recorder.set_state("REST", effort)

    session.wait_until(t0 + timeline.TRIAL_DURATION_S)


# ---------------------------------------------------------------- main


def parse_recorder_args(pairs: list[str]) -> dict:
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--recorder-arg expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        key = key.replace("-", "_")
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--objects", type=Path, default=HERE / "objects.json")
    parser.add_argument("--recorder", default="synthetic")
    parser.add_argument("--recorder-arg", action="append", default=[])
    parser.add_argument("--placement-notes", default="", help="electrode placement, free text")
    parser.add_argument("--baseline-s", type=float, default=10.0)
    parser.add_argument("--mvc-s", type=float, default=5.0)
    parser.add_argument("--mvc-reps", type=int, default=2)
    parser.add_argument("--break-every", type=int, default=20)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-self-report", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N trials")
    args = parser.parse_args()

    session_dir = args.out_root / args.subject / args.session
    session_dir.mkdir(parents=True, exist_ok=True)

    schedule_path = session_dir / "schedule.csv"
    if not schedule_path.exists():
        raise SystemExit(
            f"no schedule at {schedule_path}\n"
            f"generate it first:\n"
            f"  python schedule.py --subject {args.subject} --session {args.session} "
            f"--out-root {args.out_root}"
        )
    with schedule_path.open() as handle:
        trials = list(csv.DictReader(handle))
    if args.limit:
        trials = trials[: args.limit]

    with args.objects.open() as handle:
        objects = json.load(handle)
    shutil.copy(args.objects, session_dir / "objects.json")

    recorder = make_recorder(args.recorder, **parse_recorder_args(args.recorder_arg))
    beeper = Beeper(session_dir / "_beeps", enabled=not args.no_audio)
    session = SessionRecorder(recorder, beeper, session_dir)

    reports: list[dict] = []
    started = time.time()

    with recorder:
        beeper.warm_up()
        preflight_info = preflight(session)
        baseline_and_mvc(session, args.baseline_s, args.mvc_s, args.mvc_reps)

        banner("READY", GREEN, f"{len(trials)} trials. Enter to start.")
        prompt("  ")

        try:
            for index, trial in enumerate(trials, start=1):
                run_trial(session, trial, objects)

                flag = ""
                if not args.no_self_report:
                    flag = prompt(
                        f"  {DIM}Enter = went as instructed, or type a note "
                        f"(e.g. 'slipped'):{RESET} "
                    ).strip()
                reports.append({"trial": trial["trial"], "flag": flag})

                if args.break_every and index % args.break_every == 0 and index < len(trials):
                    banner(
                        f"BREAK  ({index}/{len(trials)} done)",
                        CYAN,
                        "Rest the arm. Do NOT move the electrodes. Enter to resume.",
                    )
                    session.save()
                    prompt("  ")
        except KeyboardInterrupt:
            print(f"\n{YELLOW}interrupted -- saving what has been recorded{RESET}")

    session.save()
    with (session_dir / "report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial", "flag"])
        writer.writeheader()
        writer.writerows(reports)

    t, x = session.concatenated()
    jitters = [abs(c["jitter_ms"]) for c in session.cues]
    metadata = {
        "subject": args.subject,
        "session": args.session,
        "started_unix": started,
        "finished_unix": time.time(),
        "recorder": args.recorder,
        "synthetic": bool(getattr(recorder, "synthetic", False)),
        "clock_note": getattr(recorder, "clock_note", ""),
        "sample_rate_hz": float(recorder.sample_rate),
        "channel_names": list(recorder.channel_names),
        "placement_notes": args.placement_notes,
        "n_trials_planned": len(trials),
        "n_trials_completed": len(reports),
        "baseline_s": args.baseline_s,
        "mvc_s": args.mvc_s,
        "mvc_reps": args.mvc_reps,
        "preflight": preflight_info,
        "n_samples": int(x.shape[0]),
        "duration_s": float(t[-1] - t[0]) if t.size else 0.0,
        "cue_jitter_ms_max": max(jitters) if jitters else 0.0,
        "timeline": {
            "cue_s": timeline.CUE_S,
            "go_s": timeline.GO_S,
            "stop_s": timeline.STOP_S,
            "trial_duration_s": timeline.TRIAL_DURATION_S,
            "settle_s": timeline.SETTLE_S,
            "hold_s": timeline.HOLD_S,
            "onset_k": timeline.ONSET_K,
        },
    }
    with (session_dir / "session.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    banner("DONE", GREEN)
    print(f"  {len(reports)}/{len(trials)} trials, {metadata['duration_s']:.0f} s recorded")
    print(f"  max cue jitter {metadata['cue_jitter_ms_max']:.1f} ms")
    print(f"  -> {session_dir}")
    if metadata["synthetic"]:
        print(f"\n{RED}  SYNTHETIC DATA -- not a result.{RESET}")
    print(f"\n  Next:  python qc.py --root {args.out_root}\n")


if __name__ == "__main__":
    main()
