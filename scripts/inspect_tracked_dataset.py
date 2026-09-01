#!/usr/bin/env python3
"""Discover the real schema of a tracked EMG/IMU/Vive dataset tree.

Run this BEFORE validate_tracked_recording.py. That script checks a file
against an expected contract; this one works out what the contract actually
is, so the expectation is set from the data rather than guessed.

    python scripts/inspect_tracked_dataset.py "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive"

It walks the tree, groups sessions by configuration (a1, b2, mix7, ...),
reads one trial CSV per session, and classifies every column into timing /
EMG / IMU / tracker / unclassified. Nothing is silently dropped - anything
it cannot place is reported explicitly, because an unrecognised column is
usually either a naming mismatch or a channel nobody remembered was there.

The three questions it answers:

  1. What are the real column names, per configuration?
  2. Do the configurations agree with each other? A rig rebuilt between
     sessions, or a different electrode count for the 'mix' conditions,
     changes the loader's shape - and finding that out after training
     starts is expensive.
  3. What data.* config does this dataset need? It prints a ready YAML
     block, so the names come from the files rather than from memory.

Standalone: pandas and numpy only, no project imports, so it can be run
wherever the drive happens to be mounted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATTERN = re.compile(r"(?:^|[_-])(mix\d+|[ab]\d+)(?:_|$)", re.IGNORECASE)

TRACKER_FIELD_PATTERN = re.compile(
    r"_(pos_[xyz]_m|quat_[wxyz]|vel_[xyz]_mps|angvel_[xyz]_radps"
    r"|tracking_age_us|vive_timestamp_us|sync_error_ms)$",
    re.IGNORECASE,
)
IMU_PATTERN = re.compile(r"^(acc|gyro|mag)\b", re.IGNORECASE)


def configuration_of(path: Path) -> str:
    """Configuration label from any folder name on the path (a1, mix7, ...)."""
    for part in reversed(path.parts):
        match = CONFIG_PATTERN.search(part)
        if match:
            return match.group(1).lower()
    return "unknown"


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "timing": [], "emg": [], "imu": [], "tracker": [], "unclassified": []
    }
    for column in columns:
        lowered = column.lower()
        # Tracker first: a Vive angular-velocity column contains neither
        # "emg" nor an IMU prefix, but a poorly-named one could collide.
        if "vive" in lowered or TRACKER_FIELD_PATTERN.search(lowered):
            groups["tracker"].append(column)
        elif "emg" in lowered:
            groups["emg"].append(column)
        elif IMU_PATTERN.match(lowered):
            groups["imu"].append(column)
        elif lowered.startswith("time") or "timestamp" in lowered:
            groups["timing"].append(column)
        else:
            groups["unclassified"].append(column)
    return groups


def infer_sensor_names(emg_columns: list[str]) -> list[str]:
    """Trailing token of each EMG column, e.g. 'EMG RMS 1_S0' -> 'S0'."""
    names = []
    for column in emg_columns:
        token = column.rsplit("_", 1)[-1].strip()
        if token and token not in names:
            names.append(token)
    return names


def infer_tracker_prefixes(tracker_columns: list[str]) -> list[str]:
    """Distinct '{PREFIX}_{ID}' stems, so a second tracker shows up as its own."""
    prefixes = []
    for column in tracker_columns:
        match = TRACKER_FIELD_PATTERN.search(column)
        if not match:
            continue
        stem = column[: match.start()]
        if stem and stem not in prefixes:
            prefixes.append(stem)
    return prefixes


def read_sidecar(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001
        return None


def inspect_session(session: Path, sample_rows: int) -> dict | None:
    trials = sorted(session.glob("trial_*.csv"))
    if not trials:
        return None
    first = trials[0]
    try:
        header = list(pd.read_csv(first, nrows=0).columns)
    except Exception as error:  # noqa: BLE001
        return {"session": session, "trials": len(trials), "error": str(error)}

    groups = classify_columns(header)
    record: dict = {
        "session": session,
        "configuration": configuration_of(session),
        "trials": len(trials),
        "example": first,
        "columns": header,
        "groups": groups,
        "sensors": infer_sensor_names(groups["emg"]),
        "tracker_prefixes": infer_tracker_prefixes(groups["tracker"]),
        "sidecars": {
            name: read_sidecar(session / name)
            for name in ("sensor_placement.json", "session_summary.json")
        },
        "companions": {
            suffix: len(list(session.glob(f"trial_*{suffix}")))
            for suffix in (".npy", ".pkl")
        },
    }

    try:
        sample = pd.read_csv(first, nrows=sample_rows)
        clock = next(
            (c for c in ("time_perf_counter", "time_s") if c in sample.columns), None
        )
        if clock:
            values = pd.to_numeric(sample[clock], errors="coerce").to_numpy()
            values = values[np.isfinite(values)]
            intervals = np.diff(np.sort(values))
            intervals = intervals[intervals > 0]
            if len(intervals):
                record["sample_rate_hz"] = float(1.0 / np.median(intervals))
                record["sampled_rows"] = int(len(sample))
    except Exception:  # noqa: BLE001
        pass
    return record


def signature(record: dict) -> tuple:
    """What must match across sessions for one loader to read them all."""
    groups = record["groups"]
    return (
        tuple(sorted(groups["emg"])),
        tuple(sorted(groups["imu"])),
        tuple(sorted(groups["tracker"])),
        tuple(sorted(groups["timing"])),
    )


def summarise(label: str, columns: list[str], limit: int = 4) -> str:
    if not columns:
        return f"{label:14}: (none)"
    shown = ", ".join(columns[:limit])
    suffix = f", ... (+{len(columns) - limit} more)" if len(columns) > limit else ""
    return f"{label:14}: {len(columns):3d}  [{shown}{suffix}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="Dataset root containing the session folders")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2000,
        help="Rows read per example trial for the sample-rate estimate (default 2000).",
    )
    parser.add_argument(
        "--full-columns",
        action="store_true",
        help="Print every column name in full rather than a truncated preview.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"path does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    sessions = sorted({p.parent for p in root.rglob("trial_*.csv")})
    if not sessions:
        print(f"no trial_*.csv found anywhere under {root}", file=sys.stderr)
        sys.exit(2)

    records = []
    for session in sessions:
        record = inspect_session(session, args.sample_rows)
        if record:
            records.append(record)

    total_trials = sum(r.get("trials", 0) for r in records)
    by_configuration: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_configuration[record.get("configuration", "unknown")].append(record)

    print(f"root: {root}")
    print(
        f"{len(records)} session folder(s), {total_trials} trial CSV(s), "
        f"{len(by_configuration)} configuration group(s): "
        f"{', '.join(sorted(by_configuration))}"
    )
    print()

    for configuration in sorted(by_configuration):
        group = by_configuration[configuration]
        trials = sum(r.get("trials", 0) for r in group)
        print(f"=== configuration {configuration} : {len(group)} session(s), {trials} trial(s) ===")
        for record in group:
            if "error" in record:
                print(f"  {record['session'].name}: COULD NOT READ - {record['error']}")
                continue
            groups = record["groups"]
            rate = record.get("sample_rate_hz")
            rate_text = f"~{rate:.1f} Hz" if rate else "sample rate unknown"
            print(f"  {record['session'].name}")
            print(f"    {record['trials']} trials, {rate_text}, "
                  f"{record['companions']['.npy']} .npy, {record['companions']['.pkl']} .pkl")
            limit = 10_000 if args.full_columns else 4
            for key, label in (
                ("timing", "timing"), ("emg", "EMG"), ("imu", "IMU"),
                ("tracker", "tracker"), ("unclassified", "UNCLASSIFIED"),
            ):
                print(f"    {summarise(label, groups[key], limit)}")
            if record["sensors"]:
                print(f"    inferred sensors  : {record['sensors']}")
            if record["tracker_prefixes"]:
                print(f"    tracker prefixes  : {record['tracker_prefixes']}")
            placement = record["sidecars"].get("sensor_placement.json")
            if placement is not None:
                text = json.dumps(placement)
                print(f"    sensor_placement  : {text[:200]}{'...' if len(text) > 200 else ''}")
        print()

    # Consistency: can one loader configuration read the whole dataset?
    signatures: dict[tuple, list[str]] = defaultdict(list)
    for record in records:
        if "error" in record:
            continue
        signatures[signature(record)].append(
            f"{record.get('configuration', '?')}/{record['session'].name}"
        )

    print("=== schema consistency ===")
    if len(signatures) == 1:
        print("  all sessions share one column schema - a single loader config covers the dataset")
    else:
        print(f"  {len(signatures)} DIFFERENT schemas across sessions:")
        for index, (sig, members) in enumerate(signatures.items(), start=1):
            emg, imu, tracker, timing = sig
            print(f"    schema {index}: {len(emg)} EMG, {len(imu)} IMU, "
                  f"{len(tracker)} tracker, {len(timing)} timing  "
                  f"-> {len(members)} session(s)")
            print(f"      e.g. {members[0]}" + (f" (+{len(members)-1} more)" if len(members) > 1 else ""))
        print("  these need either separate configs or a reconciled export")

    unclassified = sorted({
        column
        for record in records
        if "error" not in record
        for column in record["groups"]["unclassified"]
    })
    if unclassified:
        print()
        print(f"=== {len(unclassified)} unclassified column(s) ===")
        print("  not recognised as timing/EMG/IMU/tracker - check whether these matter:")
        for column in unclassified[:40]:
            print(f"    {column}")
        if len(unclassified) > 40:
            print(f"    ... (+{len(unclassified) - 40} more)")

    # Config block, taken from the data rather than from memory. Built from
    # the schema the most sessions share, not simply the first one found -
    # with mixed schemas present, the first folder alphabetically can easily
    # be the minority case, and a config for the minority would look
    # perfectly plausible while failing on most of the dataset.
    reference = None
    if signatures:
        majority = max(signatures.values(), key=len)
        majority_names = set(majority)
        reference = next(
            (
                r for r in records
                if "error" not in r
                and f"{r.get('configuration', '?')}/{r['session'].name}" in majority_names
            ),
            None,
        )
    if reference is None:
        reference = next((r for r in records if "error" not in r), None)
    if reference:
        share = len(max(signatures.values(), key=len)) if signatures else 0
        print()
        print(
            f"=== suggested data config (schema shared by {share}/{len(records)} session(s), "
            f"from {reference.get('configuration', '?')}/{reference['session'].name}) ==="
        )
        if len(signatures) > 1:
            print("  WARNING: other sessions use a different schema - this config will not read them")
        sensors = reference["sensors"]
        prefixes = reference["tracker_prefixes"]
        print("data:")
        if sensors:
            print(f"  sensors: {sensors}")
        if prefixes:
            stem = prefixes[0]
            if "_" in stem:
                prefix, tracker_id = stem.rsplit("_", 1)
                print(f'  tracker_column_prefix: "{prefix}"')
                print(f'  tracker_id: "{tracker_id}"')
            else:
                print(f'  tracker_column_prefix: "{stem}"')
            if len(prefixes) > 1:
                print(f"  # NOTE: {len(prefixes)} trackers present: {prefixes}")
        rate = reference.get("sample_rate_hz")
        if rate:
            print(f"  sample_rate_hz: {rate:.5f}")
        emg_example = reference["groups"]["emg"][0] if reference["groups"]["emg"] else None
        if emg_example and sensors:
            template = emg_example.replace(sensors[0], "{sensor}")
            if template != emg_example:
                print(f'  emg_column_template: "{template}"')
        for label, key in (("acc", "ACC"), ("gyro", "GYRO")):
            match = next(
                (c for c in reference["groups"]["imu"] if c.upper().startswith(key)), None
            )
            if match and sensors:
                template = match.replace(sensors[0], "{sensor}")
                for axis in ("X", "Y", "Z"):
                    if f" {axis}" in template or f"_{axis}" in template:
                        template = template.replace(f" {axis}", " {axis}").replace(
                            f"_{axis}", "_{axis}"
                        )
                        break
                print(f'  {label}_column_template: "{template}"')


if __name__ == "__main__":
    main()
