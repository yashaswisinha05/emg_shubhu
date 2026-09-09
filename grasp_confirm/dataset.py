"""Segment recorded sessions into labelled windows, anchored on EMG onset.

The continuous ``raw.npz`` plus ``cues.csv`` are the source of truth; this
module cuts them using the policy in timeline.py. Re-run it whenever that
policy changes -- it never modifies the recordings.

Because trials are "act immediately on GO", reaction time varies from trial to
trial and a fixed offset would sometimes land on the closing transient and
sometimes on the plateau. So the window is anchored to the detected EMG onset
instead, with the threshold derived from that session's own rest baseline --
which makes it rescale automatically across sessions and electrode
re-applications.

One caveat if you ever report onset latency as a result: the envelope is built
with a zero-phase filter, which smears energy backwards as well as forwards, so
detected onsets sit roughly 30-60 ms EARLIER than the true burst. That bias is
harmless for windowing (SETTLE_S absorbs it) but it is not a physiological
anticipation, and it would be wrong to present it as one.

    from grasp_confirm.dataset import load_dataset
    windows = load_dataset("data")

    python -m grasp_confirm.dataset --root data
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import timeline
from .features import detect_onset, envelope

GRASP_TYPE_CODES = {"none": 0, "power": 1, "pinch": 2}

#: Fixed normalisation for the object property vector -- not fitted on the
#: training set, so an unseen object lands in the same space at test time.
MASS_SCALE_G = 1000.0
APERTURE_SCALE_MM = 100.0


@dataclass
class Window:
    """One labelled segment of EMG."""

    subject: str
    session: str
    trial: int
    condition: str
    family: str
    object_id: str
    effort: str
    label: str
    t0: float
    t1: float
    fs: float
    channel_names: list[str]
    x: np.ndarray  # (n_samples, n_channels)
    properties: np.ndarray
    onset_latency_s: float | None  # from GO beep to detected onset
    coverage: float
    synthetic: bool
    flag: str = ""

    @property
    def y(self) -> int:
        return timeline.LABEL_TO_INDEX[self.label]


@dataclass
class SegmentationReport:
    """Why trials were dropped. Read this -- it is a data-quality signal."""

    kept: int = 0
    no_onset: list[int] = field(default_factory=list)
    moved_during_nothing: list[int] = field(default_factory=list)
    hold_too_short: list[int] = field(default_factory=list)
    low_coverage: list[int] = field(default_factory=list)

    def merge(self, other: "SegmentationReport") -> None:
        self.kept += other.kept
        self.no_onset += other.no_onset
        self.moved_during_nothing += other.moved_during_nothing
        self.hold_too_short += other.hold_too_short
        self.low_coverage += other.low_coverage

    def __str__(self) -> str:
        lines = [f"kept {self.kept} trials"]
        for name, items, note in (
            ("no onset detected", self.no_onset, "subject did not act, or threshold too high"),
            ("moved during NOTHING", self.moved_during_nothing, "subject moved -- correctly dropped"),
            ("hold too short", self.hold_too_short, "onset too late for a full window"),
            ("low sample coverage", self.low_coverage, "dropped samples in the stream"),
        ):
            if items:
                lines.append(f"  dropped {len(items):3d}  {name:22s} ({note})")
        return "\n".join(lines)


def property_vector(spec: dict) -> np.ndarray:
    """Anchor-modality embedding: [mass, aperture, compliance, friction, one-hot grasp]."""
    grasp = str(spec.get("grasp_type", "none")).lower()
    one_hot = np.zeros(3)
    one_hot[GRASP_TYPE_CODES.get(grasp, 0)] = 1.0
    return np.concatenate(
        [
            [
                float(spec.get("mass_g", 0.0)) / MASS_SCALE_G,
                float(spec.get("aperture_mm", 0.0)) / APERTURE_SCALE_MM,
                float(spec.get("compliance", 0.0)),
                float(spec.get("friction", 0.0)),
            ],
            one_hot,
        ]
    )


def _read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _slice(t_raw: np.ndarray, x_raw: np.ndarray, t0: float, t1: float) -> np.ndarray:
    lo = np.searchsorted(t_raw, t0, side="left")
    hi = np.searchsorted(t_raw, t1, side="right")
    return x_raw[lo:hi]


def baseline_threshold(
    t_raw: np.ndarray, x_raw: np.ndarray, fs: float, marks: list[dict]
) -> tuple[float, float]:
    """Onset threshold from the session's own rest recording.

    Returns ``(threshold, baseline_mean)``. Falls back to a percentile of the
    whole session if no baseline mark exists, which is worse but not fatal.
    """
    rest = [m for m in marks if m["label"] == "baseline"]
    if rest:
        segment = _slice(t_raw, x_raw, float(rest[0]["t_start"]), float(rest[0]["t_end"]))
    else:
        segment = x_raw
    env, _ = envelope(segment, fs)
    trace = env.mean(axis=1)
    if not rest:
        # No baseline: assume the quietest 25% of the session is rest.
        trace = trace[trace <= np.percentile(trace, 25)]
    mean, std = float(trace.mean()), float(trace.std())
    return mean + timeline.ONSET_K * std, mean


def load_session(
    session_dir: Path, min_coverage: float = 0.9, harvest_rest_tail: bool = True
) -> tuple[list[Window], SegmentationReport]:
    """Cut one session directory into labelled windows."""
    session_dir = Path(session_dir)
    with (session_dir / "session.json").open() as handle:
        meta = json.load(handle)
    with (session_dir / "objects.json").open() as handle:
        objects = json.load(handle)["objects"]

    raw = np.load(session_dir / "raw.npz")
    t_raw, x_raw = raw["t"], raw["x"]
    fs = float(meta["sample_rate_hz"])

    marks_path = session_dir / "marks.csv"
    marks = _read_csv(marks_path) if marks_path.exists() else []
    threshold, _ = baseline_threshold(t_raw, x_raw, fs, marks)

    schedule = {int(r["trial"]): r for r in _read_csv(session_dir / "schedule.csv")}
    cues: dict[int, dict[str, float]] = {}
    for row in _read_csv(session_dir / "cues.csv"):
        cues.setdefault(int(row["trial"]), {})[row["event"]] = float(row["t_perf"])

    flags: dict[int, str] = {}
    report_path = session_dir / "report.csv"
    if report_path.exists():
        flags = {int(r["trial"]): r.get("flag", "") for r in _read_csv(report_path)}

    windows: list[Window] = []
    report = SegmentationReport()

    for trial_id, events in sorted(cues.items()):
        row = schedule.get(trial_id)
        if row is None or "go" not in events or "stop" not in events:
            continue
        go, stop = events["go"], events["stop"]
        spec = objects[row["object_id"]]
        props = property_vector(spec)
        is_nothing = row["family"] == "NOTHING"

        search = _slice(t_raw, x_raw, go - timeline.ONSET_SEARCH_PAD_S, stop)
        if search.shape[0] < 2:
            report.low_coverage.append(trial_id)
            continue
        env, fs_env = envelope(search, fs)
        onset_idx = detect_onset(env, fs_env, threshold, timeline.ONSET_PERSISTENCE_S)

        if is_nothing:
            if onset_idx is not None:
                report.moved_during_nothing.append(trial_id)
                continue
            w0, w1, latency = go + timeline.EDGE_S, stop - timeline.EDGE_S, None
        else:
            if onset_idx is None:
                report.no_onset.append(trial_id)
                continue
            onset = go - timeline.ONSET_SEARCH_PAD_S + onset_idx / fs_env
            latency = onset - go
            w0 = onset + timeline.SETTLE_S
            w1 = min(w0 + timeline.HOLD_S, stop)
            if w1 - w0 < timeline.MIN_HOLD_S:
                report.hold_too_short.append(trial_id)
                continue

        segment = _slice(t_raw, x_raw, w0, w1)
        coverage = segment.shape[0] / max(1.0, (w1 - w0) * fs)
        if coverage < min_coverage:
            report.low_coverage.append(trial_id)
            continue

        report.kept += 1
        windows.append(
            Window(
                subject=meta["subject"],
                session=meta["session"],
                trial=trial_id,
                condition=row["condition"],
                family=row["family"],
                object_id=row["object_id"],
                effort=row["effort"],
                label=row["label"],
                t0=w0,
                t1=w1,
                fs=fs,
                channel_names=list(meta["channel_names"]),
                x=np.asarray(segment, dtype=np.float32),
                properties=props,
                onset_latency_s=latency,
                coverage=float(coverage),
                synthetic=bool(meta.get("synthetic", False)),
                flag=flags.get(trial_id, ""),
            )
        )

        if harvest_rest_tail and not is_nothing:
            r0 = stop + timeline.EDGE_S
            r1 = r0 + timeline.REST_TAIL_S
            tail = _slice(t_raw, x_raw, r0, r1)
            if tail.shape[0] / max(1.0, (r1 - r0) * fs) >= min_coverage:
                windows.append(
                    Window(
                        subject=meta["subject"],
                        session=meta["session"],
                        trial=trial_id,
                        condition=row["condition"],
                        family="REST_TAIL",
                        object_id=row["object_id"],
                        effort=row["effort"],
                        label="REST",
                        t0=r0,
                        t1=r1,
                        fs=fs,
                        channel_names=list(meta["channel_names"]),
                        x=np.asarray(tail, dtype=np.float32),
                        properties=props,
                        onset_latency_s=None,
                        coverage=1.0,
                        synthetic=bool(meta.get("synthetic", False)),
                        flag=flags.get(trial_id, ""),
                    )
                )

    return windows, report


def load_dataset(
    root: Path | str, min_coverage: float = 0.9, harvest_rest_tail: bool = True
) -> list[Window]:
    """Load every session under ``root`` laid out as ``<subject>/<session>/``."""
    windows: list[Window] = []
    for session_json in sorted(Path(root).glob("*/*/session.json")):
        session_windows, _ = load_session(
            session_json.parent, min_coverage, harvest_rest_tail
        )
        windows.extend(session_windows)
    return windows


def summarise(root: Path) -> str:
    windows: list[Window] = []
    report = SegmentationReport()
    for session_json in sorted(Path(root).glob("*/*/session.json")):
        session_windows, session_report = load_session(session_json.parent)
        windows.extend(session_windows)
        report.merge(session_report)

    if not windows:
        return f"no windows found under {root}\n{report}"

    lines = [f"{len(windows)} windows", "", str(report), ""]
    if any(w.synthetic for w in windows):
        lines.append("*** contains SYNTHETIC data -- not a result ***\n")

    for title, counter in (
        ("by label", Counter(w.label for w in windows)),
        ("by condition", Counter(w.condition for w in windows if w.family != "REST_TAIL")),
        ("by object (AIR/GRASP only)", Counter(w.object_id for w in windows if w.label != "REST")),
    ):
        lines.append(f"{title}:")
        lines += [f"  {k:16s} {v:5d}" for k, v in sorted(counter.items())]
        lines.append("")

    lines.append("by subject / session:")
    for (s, ss), v in sorted(Counter((w.subject, w.session) for w in windows).items()):
        lines.append(f"  {s}/{ss:12s} {v:5d}")

    latencies = [w.onset_latency_s for w in windows if w.onset_latency_s is not None]
    if latencies:
        arr = np.array(latencies)
        lines.append(
            f"\nonset latency after GO: median {np.median(arr) * 1000:.0f} ms, "
            f"IQR {np.percentile(arr, 25) * 1000:.0f}-{np.percentile(arr, 75) * 1000:.0f} ms"
        )
        if np.median(arr) > 0.6:
            lines.append("  WARNING: slow reactions -- subjects may not be acting 'immediately'")

    flagged = [w for w in windows if w.flag]
    if flagged:
        lines.append(f"\n{len(flagged)} windows carry a self-report flag")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment and summarise recordings.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(summarise(args.root))


if __name__ == "__main__":
    main()
