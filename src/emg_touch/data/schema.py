from __future__ import annotations

import re

# This rig's placement: S0=anterior deltoid, S4=lateral deltoid, S8=biceps,
# S12=triceps. A different rig will have a different count and different
# muscles, so anything reading these should go through sensor_names() /
# emg_columns() / imu_columns() with the data config, not the module
# constants - those remain as the default for every existing experiment and
# checkpoint in this repository.
SENSORS = ("S0", "S4", "S8", "S12")
AXES = ("X", "Y", "Z")


def sensor_names(data_config: dict | None = None) -> tuple[str, ...]:
    """Sensor labels for a dataset, defaulting to this rig's four."""
    if data_config:
        configured = data_config.get("sensors")
        if configured:
            return tuple(str(name) for name in configured)
    return SENSORS


def emg_columns(data_config: dict | None = None) -> tuple[str, ...]:
    template = (data_config or {}).get("emg_column_template", "EMG RMS 1_{sensor}")
    return tuple(template.format(sensor=s) for s in sensor_names(data_config))


def imu_columns(data_config: dict | None = None) -> tuple[str, ...]:
    accelerometer = (data_config or {}).get("acc_column_template", "ACC {axis}_{sensor}")
    gyroscope = (data_config or {}).get("gyro_column_template", "GYRO {axis}_{sensor}")
    return tuple(
        column
        for sensor in sensor_names(data_config)
        for column in (
            *(accelerometer.format(axis=a, sensor=sensor) for a in AXES),
            *(gyroscope.format(axis=a, sensor=sensor) for a in AXES),
        )
    )


EMG_COLUMNS = emg_columns()
IMU_COLUMNS = imu_columns()

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

