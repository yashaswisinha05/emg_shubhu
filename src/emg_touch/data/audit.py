from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import CONFIG_PATTERN, configuration_from_participant_id


def audit_dataset(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    summaries = sorted(root.rglob("session_summary.json"))
    report: dict[str, Any] = {
        "data_root": str(root),
        "summary_files": len(summaries),
        "configurations": defaultdict(lambda: {"sessions": 0, "trials": 0, "subjects": set()}),
        "missing_trial_files": [],
        "misplaced_sessions": [],
        "metadata_id_mismatches": [],
        "duplicate_trial_ids": [],
        "schema_counts": Counter(),
    }
    seen: set[str] = set()

    for summary_path in summaries:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        source_participant_id = str(summary["participant_id"])
        participant_dir_id = summary_path.parent.parent.name
        try:
            configuration_from_participant_id(participant_dir_id)
            participant_id = participant_dir_id
        except ValueError:
            participant_id = source_participant_id
        if participant_id != source_participant_id:
            report["metadata_id_mismatches"].append(
                {
                    "metadata_participant_id": source_participant_id,
                    "canonical_folder_id": participant_id,
                    "summary_path": str(summary_path),
                }
            )
        session_id = str(summary["session_id"])
        config = configuration_from_participant_id(participant_id)
        subject_match = CONFIG_PATTERN.search(participant_id)
        subject = participant_id[: subject_match.start(1)].rstrip("_-").lower() if subject_match else "unknown"
        config_report = report["configurations"][config]
        config_report["sessions"] += 1
        config_report["subjects"].add(subject)

        relative_parts = summary_path.relative_to(root).parts
        physical_config = relative_parts[0].lower() if relative_parts else ""
        if physical_config != config:
            report["misplaced_sessions"].append(
                {
                    "participant_id": participant_id,
                    "logical_configuration": config,
                    "physical_top_level": physical_config,
                    "summary_path": str(summary_path),
                }
            )

        for trial in summary.get("trial_results", []):
            number = int(trial["trial_number"])
            stem = f"trial_{number:03d}"
            trial_id = f"{participant_id}__{session_id}__{stem}"
            if trial_id in seen:
                report["duplicate_trial_ids"].append(trial_id)
            seen.add(trial_id)
            config_report["trials"] += 1
            missing = [
                suffix
                for suffix in (".csv", ".npy", ".pkl")
                if not (summary_path.parent / f"{stem}{suffix}").exists()
            ]
            if missing:
                report["missing_trial_files"].append(
                    {"trial_id": trial_id, "missing_suffixes": missing}
                )

        first_csv = next(summary_path.parent.glob("trial_*.csv"), None)
        if first_csv is not None:
            columns = pd.read_csv(first_csv, nrows=0).columns
            signal_columns = tuple(
                column for column in columns if column.startswith(("EMG", "ACC", "GYRO"))
            )
            report["schema_counts"][str(signal_columns)] += 1

    report["configurations"] = {
        key: {
            **value,
            "subjects": sorted(value["subjects"]),
        }
        for key, value in sorted(report["configurations"].items())
    }
    report["schema_counts"] = dict(report["schema_counts"])
    return report
