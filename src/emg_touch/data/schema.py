from __future__ import annotations

import re

SENSORS = ("S0", "S4", "S8", "S12")
AXES = ("X", "Y", "Z")

EMG_COLUMNS = tuple(f"EMG RMS 1_{sensor}" for sensor in SENSORS)
IMU_COLUMNS = tuple(
    column
    for sensor in SENSORS
    for column in (
        *(f"ACC {axis}_{sensor}" for axis in AXES),
        *(f"GYRO {axis}_{sensor}" for axis in AXES),
    )
)

CONFIG_PATTERN = re.compile(r"(?:^|[_-])(mix\d+|[ab]\d+)(?:_|$)", re.IGNORECASE)


def configuration_from_participant_id(participant_id: str) -> str:
    match = CONFIG_PATTERN.search(participant_id)
    if not match:
        raise ValueError(f"Cannot derive configuration from participant_id={participant_id!r}")
    return match.group(1).lower()


def subject_from_participant_id(participant_id: str) -> str:
    match = CONFIG_PATTERN.search(participant_id)
    if not match:
        raise ValueError(f"Cannot derive subject from participant_id={participant_id!r}")
    return participant_id[: match.start(1)].rstrip("_-").lower()


def natural_configuration_key(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"([a-z]+)(\d+)", value.lower())
    if not match:
        return value, -1
    return match.group(1), int(match.group(2))

