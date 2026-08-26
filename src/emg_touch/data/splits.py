from __future__ import annotations

from typing import Iterable

import pandas as pd


def make_subject_split(
    manifest: pd.DataFrame,
    test_subject: str,
    val_subject: str,
    configuration: str | None = None,
) -> dict[str, list[str] | str | None]:
    frame = manifest
    if configuration:
        frame = frame.loc[frame["configuration"] == configuration]
    subjects = set(frame["subject"].astype(str))
    if test_subject not in subjects:
        raise ValueError(f"test subject {test_subject!r} not available; choices={sorted(subjects)}")
    if val_subject not in subjects:
        raise ValueError(f"validation subject {val_subject!r} not available; choices={sorted(subjects)}")
    if test_subject == val_subject:
        raise ValueError("test and validation subjects must differ")
    train = frame.loc[~frame["subject"].isin([test_subject, val_subject]), "trial_id"].tolist()
    val = frame.loc[frame["subject"] == val_subject, "trial_id"].tolist()
    test = frame.loc[frame["subject"] == test_subject, "trial_id"].tolist()
    if not train or not val or not test:
        raise ValueError("subject-safe split produced an empty partition")
    return {
        "configuration": configuration,
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train": train,
        "val": val,
        "test": test,
    }


def subset_from_trial_ids(frame: pd.DataFrame, trial_ids: Iterable[str]) -> pd.DataFrame:
    wanted = set(trial_ids)
    result = frame.loc[frame["trial_id"].isin(wanted)].copy()
    missing = wanted - set(result["trial_id"])
    if missing:
        raise ValueError(f"Split references {len(missing)} trials absent from manifest")
    return result.reset_index(drop=True)

