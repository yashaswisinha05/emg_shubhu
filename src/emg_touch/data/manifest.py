from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import configuration_from_participant_id, subject_from_participant_id


MANIFEST_COLUMNS = (
    "trial_id",
    "configuration",
    "subject",
    "participant_id",
    "session_id",
    "trial_number",
    "csv_path",
    "npy_path",
    "pkl_path",
    "cache_path",
    "total_samples",
    "reaction_time_s",
    "click_x_norm",
    "click_y_norm",
    "target_x_norm",
    "target_y_norm",
    "canvas_width_px",
    "canvas_height_px",
    "button_width_px",
    "button_height_px",
)

OPTIONAL_MANIFEST_COLUMNS = (
    "touch_time_s",
    "touch_alignment_source",
)


def touch_time_from_trial_pickle(path: str | Path) -> float | None:
    """Map the GUI click/save timestamp into the signal's relative time axis.

    Trial files do not store cue time directly in the signal array.  They do store
    the monotonic timestamp at which the click callback saved the trial, plus the
    raw signal timestamps and their saved relative coordinates.  Using the same
    origin reconstruction as cache preparation yields a touch-aligned signal time.
    """

    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None
    required = {"save_timestamp_perf", "timestamps", "relative_time_s"}
    if not required.issubset(payload):
        return None
    timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
    relative = np.asarray(payload["relative_time_s"], dtype=np.float64)
    if timestamps.ndim != 1 or relative.ndim != 1 or len(timestamps) != len(relative):
        return None
    finite = np.isfinite(timestamps) & np.isfinite(relative)
    if not finite.any() or not np.isfinite(float(payload["save_timestamp_perf"])):
        return None
    timestamps = timestamps[finite]
    relative = relative[finite]
    zero_index = int(np.argmin(np.abs(relative)))
    signal_origin_perf = timestamps[zero_index] - relative[zero_index]
    return float(payload["save_timestamp_perf"]) - float(signal_origin_perf)


def build_manifest_rows(
    data_root: str | Path,
    cache_dir: str | Path,
    subject_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    data_root = Path(data_root).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    aliases = {key.lower(): value.lower() for key, value in (subject_aliases or {}).items()}
    rows: list[dict[str, Any]] = []
    seen_trial_ids: set[str] = set()

    for summary_path in sorted(data_root.rglob("session_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        source_participant_id = str(summary["participant_id"])
        session_id = str(summary["session_id"])
        session_dir = summary_path.parent
        participant_dir_id = session_dir.parent.name
        # Folder names are canonical because a small number of summary files contain
        # typographical IDs (for example a missing "a" or an extra digit).
        try:
            configuration_from_participant_id(participant_dir_id)
            participant_id = participant_dir_id
        except ValueError:
            participant_id = source_participant_id
        configuration = configuration_from_participant_id(participant_id)
        subject = subject_from_participant_id(participant_id)
        subject = aliases.get(subject, subject)

        for result in summary.get("trial_results", []):
            trial_number = int(result["trial_number"])
            stem = f"trial_{trial_number:03d}"
            trial_id = f"{participant_id}__{session_id}__{stem}"
            if trial_id in seen_trial_ids:
                raise ValueError(f"Duplicate logical trial: {trial_id}")
            seen_trial_ids.add(trial_id)

            click = result.get("click_coordinates_norm", [None, None])
            target = result.get("target_center_norm", [None, None])
            canvas = result.get("canvas_dimensions_px", [None, None])
            csv_path = session_dir / f"{stem}.csv"
            npy_path = session_dir / f"{stem}.npy"
            pkl_path = session_dir / f"{stem}.pkl"
            cache_path = cache_dir / configuration / participant_id / session_id / f"{stem}.npz"
            touch_time_s = touch_time_from_trial_pickle(pkl_path)

            rows.append(
                {
                    "trial_id": trial_id,
                    "configuration": configuration,
                    "subject": subject,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "trial_number": trial_number,
                    "csv_path": str(csv_path),
                    "npy_path": str(npy_path),
                    "pkl_path": str(pkl_path),
                    "cache_path": str(cache_path),
                    "total_samples": result.get("total_samples"),
                    "reaction_time_s": result.get("reaction_time_s"),
                    "click_x_norm": click[0],
                    "click_y_norm": click[1],
                    "target_x_norm": target[0],
                    "target_y_norm": target[1],
                    "canvas_width_px": canvas[0],
                    "canvas_height_px": canvas[1],
                    "button_width_px": 80,
                    "button_height_px": 80,
                    "touch_time_s": touch_time_s,
                    "touch_alignment_source": (
                        "save_timestamp_perf" if touch_time_s is not None else None
                    ),
                }
            )
    return rows


def write_manifest(rows: Iterable[dict[str, Any]], output_path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=(*MANIFEST_COLUMNS, *OPTIONAL_MANIFEST_COLUMNS))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return frame
