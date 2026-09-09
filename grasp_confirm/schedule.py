"""Generate a randomised trial schedule for one recording session.

    python schedule.py --subject subj01 --session sess01

Session composition (84 trials, ~9 min of cue time):

    AIR_LIGHT      12   "close and hold GENTLY, as if holding an egg"
    AIR_HARD       12   "SQUEEZE HARD, as if the bottle were slipping"
    GRASP          48   4 objects x 2 efforts (normal / firm) x 6 repeats
    NOTHING        12   hand stays open, do nothing

AIR_LIGHT and AIR_HARD bracket the GRASP amplitude distribution from below and
above. That bracketing is the only thing standing between this experiment and
an RMS threshold: if you drop either one, a classifier can separate AIR from
GRASP by loudness alone and the result means nothing. qc.py checks it.

Ordering is randomised with no more than 3 consecutive trials of the same
family, so fatigue and drift cannot correlate with class.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

GRASP_EFFORTS = ["normal", "firm"]
GRASP_REPEATS = 6

# (condition, family, object_id, effort, n_trials)
FIXED_CONDITIONS = [
    ("AIR_LIGHT", "AIR", "null", "light", 12),
    ("AIR_HARD", "AIR", "null", "hard", 12),
    ("NOTHING", "NOTHING", "null", "none", 12),
]

INSTRUCTIONS = {
    "AIR_LIGHT": "Hand open around the SAME SPOT. On the beep: close and hold GENTLY, like an egg.",
    "AIR_HARD": "Hand open around the SAME SPOT. On the beep: close and SQUEEZE HARD.",
    "NOTHING": "Hand open and relaxed. On the beep: DO NOTHING. Stay still.",
    "GRASP_normal": "Hand open AROUND the object, not touching. On the beep: grasp with your NORMAL grip.",
    "GRASP_firm": "Hand open AROUND the object, not touching. On the beep: grasp FIRMLY.",
}

FIELDNAMES = [
    "trial",
    "condition",
    "family",
    "object_id",
    "effort",
    "label",
    "instruction",
]


def load_objects(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def build_trials(objects: dict) -> list[dict]:
    grasp_objects = [name for name in objects["objects"] if name != "null"]
    if not grasp_objects:
        raise SystemExit("objects.json defines no graspable objects")

    trials: list[dict] = []

    for condition, family, object_id, effort, count in FIXED_CONDITIONS:
        for _ in range(count):
            trials.append(
                {
                    "condition": condition,
                    "family": family,
                    "object_id": object_id,
                    "effort": effort,
                    "label": "REST" if family == "NOTHING" else "AIR",
                    "instruction": INSTRUCTIONS[condition],
                }
            )

    for object_id in grasp_objects:
        for effort in GRASP_EFFORTS:
            for _ in range(GRASP_REPEATS):
                trials.append(
                    {
                        "condition": f"GRASP_{effort.upper()}",
                        "family": "GRASP",
                        "object_id": object_id,
                        "effort": effort,
                        "label": "GRASP",
                        "instruction": INSTRUCTIONS[f"GRASP_{effort}"],
                    }
                )

    return trials


def order_trials(
    trials: list[dict], rng: random.Random, max_run: int = 3, attempts: int = 8000
) -> list[dict]:
    """Shuffle under a 'no more than ``max_run`` consecutive same-family' rule."""
    for _ in range(attempts):
        candidate = trials[:]
        rng.shuffle(candidate)
        run = 1
        ok = True
        for prev, cur in zip(candidate, candidate[1:]):
            run = run + 1 if cur["family"] == prev["family"] else 1
            if run > max_run:
                ok = False
                break
        if ok:
            return candidate
    raise SystemExit(
        f"could not order trials with max_run={max_run} in {attempts} attempts; "
        "relax --max-run"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--objects", type=Path, default=HERE / "objects.json")
    parser.add_argument("--seed", type=int, default=None, help="defaults to a hash of subject+session")
    parser.add_argument("--max-run", type=int, default=3)
    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = abs(hash(f"{args.subject}/{args.session}")) % (2**31)
    rng = random.Random(seed)

    objects = load_objects(args.objects)
    unmeasured = [n for n, spec in objects["objects"].items() if not spec.get("measured")]
    if unmeasured:
        print(
            "WARNING: these objects still hold PLACEHOLDER properties: "
            + ", ".join(sorted(unmeasured))
            + '\n         Measure mass/aperture and set "measured": true in objects.json.\n'
        )

    trials = order_trials(build_trials(objects), rng, max_run=args.max_run)

    out = args.out or args.out_root / args.subject / args.session / "schedule.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, trial in enumerate(trials, start=1):
            writer.writerow({"trial": index, **trial})

    counts: dict[str, int] = {}
    for trial in trials:
        counts[trial["condition"]] = counts.get(trial["condition"], 0) + 1

    print(f"wrote {len(trials)} trials -> {out}  (seed {seed})")
    for condition in sorted(counts):
        print(f"  {condition:16s} {counts[condition]:3d}")


if __name__ == "__main__":
    main()
