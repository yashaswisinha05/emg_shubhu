"""Dataset and preprocessing for the tracked EMG/IMU/Vive recordings.

Every decision here follows from something measured on the real data by the
diagnostic scripts rather than assumed, so the reasoning is recorded next to
the code that depends on it.

Decimation (data.decimation, default 10 -> ~126 Hz from ~1259 Hz).
    The Vive lighthouse updates at roughly 250 Hz but the export carries a
    distinct pose on every row at ~1259 Hz, so something upsamples onto the
    EMG clock. Whether that is linear interpolation, a spline, or IMU-fused
    pose prediction changes what the high-rate acceleration means, and the
    piecewise-linearity check can only rule out the linear case. Decimating
    to well below the true tracker rate makes the question moot: at ~126 Hz
    we sample the underlying trajectory instead of the interpolant, whatever
    produced it. Differentiating at the row rate would instead amplify
    whatever the upsampler invented.

EMG envelope (data.emg_envelope_ms, default 40 ms).
    The tracked export records raw bipolar EMG ("EMG 1_S0"), measured at 52%
    negative samples - not the RMS envelope ("EMG RMS 1_S0") every earlier
    result in this project used. Raw is strictly more information, but it is
    not comparable until rectified and enveloped. The window is causal, for
    the same reason as everything else here: the task is early prediction.

Clock repair.
    time_perf_counter is monotonic by construction yet 2.17% of rows arrive
    out of order, so rows are sorted by it before anything else. This fixes
    ordering but cannot fix a row that pairs EMG from one instant with a pose
    from another, which is a capture-loop question rather than a
    preprocessing one.

Sync filtering (data.tracker_max_sync_error_ms, default 20).
    sync_error_ms has a ~3 ms median with a tail reaching 89 ms - longer than
    the 40-80 ms electromechanical delay that makes EMG predictive at all.
    Measured tail: 0.70% of samples over 10 ms, 0.13% over 20 ms, 0.05% over
    40 ms. A 20 ms cut discards about one sample in 800.
    tracking_age_us is NOT used: it reads exactly zero in every trial sampled
    across all 15 sessions, so it is a stub rather than a clean tracker.

Movement onset.
    Trials open with a stationary pre-buffer - the export carries
    pre_buffer_samples, and the flat-curvature regions were confirmed to sit
    at 0.0000 m/s. A still hand has no destination-directed acceleration, so
    those samples contribute noise to the attractor readout rather than
    signal. Onset is found from the speed profile rather than trusting the
    declared pre-buffer, and reported so the two can be compared.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .schema import emg_columns, imu_columns
from .tracked_trajectory import (
    position_columns,
    quaternion_columns,
    sync_error_column,
    velocity_columns,
)

TASK_TARGET_COLUMNS = ("click_x_norm", "click_y_norm")


def causal_envelope(values: np.ndarray, window: int) -> np.ndarray:
    """Rectify, then trailing-mean. Never reads a future sample."""
    rectified = np.abs(values)
    if window <= 1:
        return rectified
    padded = np.concatenate(
        [np.repeat(rectified[:1], window - 1, axis=0), rectified], axis=0
    )
    cumulative = np.cumsum(padded, axis=0)
    cumulative = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=cumulative.dtype), cumulative], axis=0
    )
    return ((cumulative[window:] - cumulative[:-window]) / float(window)).astype(
        np.float32
    )


def movement_onset(speed: np.ndarray, fraction: float = 0.05) -> int:
    """First index where speed passes a fraction of the trial's peak.

    Taken from the speed profile rather than the declared pre_buffer_samples
    so the two can be compared - a declared value that disagrees with the
    motion is worth knowing about.
    """
    if not len(speed):
        return 0
    peak = float(np.nanmax(speed))
    if not np.isfinite(peak) or peak <= 0:
        return 0
    moving = np.flatnonzero(speed >= fraction * peak)
    return int(moving[0]) if len(moving) else 0


def preprocess_tracked_trial(
    csv_path: str | Path, data_config: dict[str, Any]
) -> dict[str, np.ndarray] | None:
    frame = pd.read_csv(csv_path)
    if "time_perf_counter" not in frame.columns:
        return None

    perf = pd.to_numeric(frame["time_perf_counter"], errors="coerce").to_numpy()
    keep = np.isfinite(perf)
    frame, perf = frame.loc[keep].reset_index(drop=True), perf[keep]
    if len(perf) < 64:
        return None
    # 2.17% of rows arrive out of order on a monotonic clock; sort first.
    order = np.argsort(perf, kind="stable")
    frame, perf = frame.iloc[order].reset_index(drop=True), perf[order]

    position_names = position_columns(data_config)
    velocity_names = velocity_columns(data_config)
    emg_names = emg_columns(data_config)
    imu_names = imu_columns(data_config)
    for required in (position_names, velocity_names, emg_names, imu_names):
        if not all(name in frame.columns for name in required):
            return None

    def block(names: tuple[str, ...]) -> np.ndarray:
        return (
            frame[list(names)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )

    position = block(position_names)
    velocity = block(velocity_names)
    emg_raw = block(emg_names)
    imu = block(imu_names)

    valid = np.isfinite(position).all(axis=1) & np.isfinite(velocity).all(axis=1)
    sync_name = sync_error_column(data_config)
    limit = data_config.get("tracker_max_sync_error_ms", 20.0)
    if sync_name in frame.columns and limit is not None:
        sync = pd.to_numeric(frame[sync_name], errors="coerce").to_numpy()
        valid = valid & np.isfinite(sync) & (np.abs(sync) <= float(limit))
    if valid.sum() < 64:
        return None

    window = max(
        1,
        int(round(float(data_config.get("emg_envelope_ms", 40.0))
                  * len(perf) / max(perf[-1] - perf[0], 1e-9) / 1000.0)),
    )
    emg = causal_envelope(np.nan_to_num(emg_raw), window)

    decimation = max(1, int(data_config.get("decimation", 10)))
    index = np.arange(0, len(perf), decimation)
    index = index[valid[index]]
    if len(index) < 16:
        # Falling back to every valid sample keeps a trial with clustered
        # dropouts rather than discarding it for landing badly on the grid.
        index = np.flatnonzero(valid)[::decimation]
        if len(index) < 16:
            return None

    speed = np.linalg.norm(velocity[index], axis=1)
    onset = movement_onset(speed)

    result = {
        "time_s": (perf[index] - perf[index[0]]).astype(np.float32),
        "emg": emg[index],
        "imu": np.nan_to_num(imu[index]),
        "position": position[index],
        "velocity": velocity[index],
        "onset": np.int64(onset),
        "sample_rate_hz": np.float32(
            len(index) / max(perf[index[-1]] - perf[index[0]], 1e-9)
        ),
    }
    quaternion_names = quaternion_columns(data_config)
    if all(name in frame.columns for name in quaternion_names):
        result["quaternion"] = block(quaternion_names)[index]
    if all(name in frame.columns for name in TASK_TARGET_COLUMNS):
        target = (
            frame[list(TASK_TARGET_COLUMNS)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        finite = target[np.isfinite(target).all(axis=1)]
        if len(finite):
            result["screen_target"] = finite[-1]
    return result


def session_emg_scale(
    trials: list[Path], data_config: dict[str, Any], sample: int = 12
) -> np.ndarray | None:
    """Robust per-channel EMG amplitude for one session.

    Raw sEMG amplitude is not comparable between sessions: electrode
    impedance, exact placement over the muscle belly, and skin preparation
    move it by orders of magnitude for identical effort. On a held-out-session
    split that is fatal - the encoder is shown one amplitude range in training
    and a different one at test, so whatever it learned about EMG cannot
    transfer, and the muscle channels act as noise that costs generalisation
    rather than adding to it.

    The original screen-coordinate pipeline in this repository normalised per
    session for exactly this reason (raw_emg_features takes a `reference`);
    the tracked pipeline was built without it, which is the most likely reason
    EMG measured as actively harmful here.

    The 95th percentile is used rather than the max, which is a single noise
    spike, or the mean, which is dominated by the stationary pre-buffer that
    occupies most of a trial. Normalising per session rather than per trial is
    deliberate: per-trial scaling would divide out how hard THIS reach was,
    which is the part of the signal actually worth having.
    """
    stride = max(1, len(trials) // sample)
    collected = []
    for path in trials[::stride][:sample]:
        data = preprocess_tracked_trial(path, data_config)
        if data is not None and "emg" in data:
            collected.append(data["emg"])
    if not collected:
        return None
    stacked = np.concatenate(collected, axis=0)
    scale = np.percentile(np.abs(stacked), 95, axis=0).astype(np.float32)
    # A dead or disconnected channel would otherwise divide by ~0 and turn
    # its noise floor into a full-scale signal.
    return np.maximum(scale, 1e-8)


def session_imu_statistics(
    trials: list[Path], data_config: dict[str, Any], sample: int = 12
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-session IMU centre and scale.

    The sweep found the IMU, not EMG, doing the damage: removing it improved
    the model by 0.46 cm with EMG present and 0.48 cm with EMG absent, against
    a seed spread of 0.09 - the same effect measured twice, while EMG's own
    cost (0.03 and 0.05 cm) sat inside the noise.

    The mechanism is the same one that motivated normalising EMG, and the IMU
    is arguably worse for it. These are limb-mounted sensors, so each channel
    carries a gravity component fixed by how the unit happened to sit on the
    arm that day. Re-applying the sensors rotates that offset, and a
    held-out-session split then presents the encoder with 24 channels whose
    baseline has moved for reasons that have nothing to do with the reach.
    Worse, the tracker already measures the same arm motion in a properly
    calibrated frame, so the IMU contributes little the model needs while
    supplying ample session-specific detail to overfit.

    Centring as well as scaling, unlike EMG: an EMG envelope is non-negative
    with a meaningful zero, while an accelerometer's zero is wherever gravity
    happens to project. Removing the per-session median removes that
    orientation offset; the 95th percentile of the absolute deviation then
    puts the remaining motion on a common scale.
    """
    stride = max(1, len(trials) // sample)
    collected = []
    for path in trials[::stride][:sample]:
        data = preprocess_tracked_trial(path, data_config)
        if data is not None and "imu" in data:
            collected.append(data["imu"])
    if not collected:
        return None
    stacked = np.concatenate(collected, axis=0)
    centre = np.median(stacked, axis=0).astype(np.float32)
    scale = np.percentile(np.abs(stacked - centre), 95, axis=0).astype(np.float32)
    return centre, np.maximum(scale, 1e-6)


class TrackedTrajectoryDataset(Dataset):
    """Preprocessed trials, cached to .npz so epochs do not re-parse CSVs."""

    def __init__(
        self,
        trials: list[Path],
        data_config: dict[str, Any],
        cache_dir: Path | None = None,
        session_index: dict[str, int] | None = None,
    ) -> None:
        self.trials = list(trials)
        # Session identity, for the adversarial participant-invariance term.
        # Built from the full set of sessions rather than this split's, so
        # train and validation agree on what index a session has.
        self.session_index = session_index or {}
        self.data_config = data_config
        # Per-session EMG scales, filled by build_tracked_loaders. Each
        # session is normalised by its own amplitude, which is what makes the
        # channels comparable across participants.
        self.emg_scales: dict[str, np.ndarray] = {}
        self.imu_statistics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Preprocessing options are part of the cache identity, so changing
        # the envelope or decimation cannot silently reuse stale arrays.
        signature = repr(
            sorted(
                (k, v)
                for k, v in data_config.items()
                if k
                in {
                    "decimation", "emg_envelope_ms", "tracker_max_sync_error_ms",
                    "sensors", "emg_column_template", "tracker_id",
                }
            )
        )
        self.signature = hashlib.sha1(signature.encode()).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.trials)

    def _cache_path(self, path: Path) -> Path | None:
        if not self.cache_dir:
            return None
        stem = hashlib.sha1(str(path).encode()).hexdigest()[:16]
        return self.cache_dir / f"{stem}-{self.signature}.npz"

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.trials[index]
        cache = self._cache_path(path)
        if cache and cache.exists():
            with np.load(cache) as handle:
                data = {k: handle[k] for k in handle.files}
        else:
            data = preprocess_tracked_trial(path, self.data_config)
            if data is None:
                data = {"unusable": np.array(True)}
            if cache:
                np.savez_compressed(cache, **data)
        if "unusable" in data:
            return {"unusable": True, "path": str(path)}
        result = {
            key: torch.from_numpy(np.asarray(value))
            for key, value in data.items()
            if key not in {"onset", "sample_rate_hz"}
        }
        result["onset"] = int(np.asarray(data["onset"]))
        scale = self.emg_scales.get(path.parent.name)
        if scale is not None and "emg" in result:
            result["emg"] = result["emg"] / torch.from_numpy(scale)
        statistics = self.imu_statistics.get(path.parent.name)
        if statistics is not None and "imu" in result:
            centre, spread = statistics
            result["imu"] = (result["imu"] - torch.from_numpy(centre)) / torch.from_numpy(spread)
        result["length"] = int(result["position"].shape[0])
        result["path"] = str(path)
        result["session"] = int(self.session_index.get(path.parent.name, 0))
        return result


def collate_tracked(batch: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [item for item in batch if not item.get("unusable")]
    if not usable:
        return None
    longest = max(item["length"] for item in usable)
    out: dict[str, Any] = {}
    for key in ("emg", "imu", "position", "velocity"):
        channels = usable[0][key].shape[1]
        padded = torch.zeros(len(usable), longest, channels, dtype=torch.float32)
        mask = torch.zeros(len(usable), longest, dtype=torch.bool)
        for row, item in enumerate(usable):
            length = item["length"]
            padded[row, :length] = item[key]
            mask[row, :length] = True
        out[key] = padded
        out[f"{key}_mask"] = mask
    out["lengths"] = torch.tensor([i["length"] for i in usable], dtype=torch.long)
    out["onset"] = torch.tensor([i["onset"] for i in usable], dtype=torch.long)
    if "screen_target" in usable[0]:
        out["screen_target"] = torch.stack([i["screen_target"] for i in usable])
    out["session"] = torch.tensor([i.get("session", 0) for i in usable], dtype=torch.long)
    out["paths"] = [i["path"] for i in usable]
    return out


def discover_trials(root: str | Path) -> dict[str, list[Path]]:
    """Session directory name -> its trial CSVs."""
    sessions: dict[str, list[Path]] = {}
    for path in sorted(Path(root).rglob("trial_*.csv")):
        sessions.setdefault(path.parent.name, []).append(path)
    return sessions


def split_sessions(
    sessions: dict[str, list[Path]], config: dict[str, Any]
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split by session by default.

    Holding out whole sessions asks whether the model transfers to an unseen
    participant and electrode application, which is the question that matters
    for a wearable. Splitting trials within a session shares the session's own
    electrode placement and skin condition across train and test, and reports
    a number that flatters accordingly - available as split_by: trial, but not
    the default.
    """
    names = sorted(sessions)
    generator = np.random.default_rng(int(config.get("seed", 42)))
    mode = str(config.get("data", {}).get("split_by", "session"))
    validation_fraction = float(config.get("data", {}).get("validation_fraction", 0.2))
    test_fraction = float(config.get("data", {}).get("test_fraction", 0.2))

    if mode == "session":
        shuffled = list(names)
        generator.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
        test_names = shuffled[:n_test]
        validation_names = shuffled[n_test : n_test + n_validation]
        train_names = shuffled[n_test + n_validation :]
        pick = lambda group: [p for n in group for p in sessions[n]]  # noqa: E731
        return pick(train_names), pick(validation_names), pick(test_names)

    everything = [p for n in names for p in sessions[n]]
    generator.shuffle(everything)
    n_test = int(len(everything) * test_fraction)
    n_validation = int(len(everything) * validation_fraction)
    return (
        everything[n_test + n_validation :],
        everything[n_test : n_test + n_validation],
        everything[:n_test],
    )


def _with_scales(dataset, scales, imu_statistics=None):
    dataset.emg_scales = scales
    dataset.imu_statistics = imu_statistics or {}
    return dataset


def build_tracked_loaders(
    config: dict[str, Any], root: str | Path, cache_dir: str | Path | None = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    sessions = discover_trials(root)
    if not sessions:
        raise ValueError(f"no trial_*.csv found under {root}")
    train, validation, test = split_sessions(sessions, config)
    session_index = {name: index for index, name in enumerate(sorted(sessions))}
    # Computed per session from that session's own trials, including the held
    # out ones. That is not leakage of the label - it is the per-session
    # calibration any real deployment performs when the electrodes go on, and
    # withholding it would measure a system nobody would actually build.
    scales: dict[str, np.ndarray] = {}
    if bool(config["data"].get("emg_session_normalise", True)):
        for name, trials in sessions.items():
            scale = session_emg_scale(trials, config["data"])
            if scale is not None:
                scales[name] = scale
    imu_statistics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if bool(config["data"].get("imu_session_normalise", True)):
        for name, trials in sessions.items():
            found = session_imu_statistics(trials, config["data"])
            if found is not None:
                imu_statistics[name] = found
    config.setdefault("virtual_leader", {})["session_count"] = len(session_index)
    data_config = config["data"]
    batch_size = int(config["training"].get("batch_size", 16))
    workers = int(config["training"].get("num_workers", 0))

    def loader(trials: list[Path], shuffle: bool) -> DataLoader:
        return DataLoader(
            _with_scales(
                TrackedTrajectoryDataset(trials, data_config, cache_dir, session_index),
                scales,
                imu_statistics,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=collate_tracked,
            drop_last=False,
        )

    return loader(train, True), loader(validation, False), loader(test, False)
