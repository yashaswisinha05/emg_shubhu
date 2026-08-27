from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import EMG_COLUMNS, IMU_COLUMNS


def causal_median_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Trailing-window median; never uses samples later than the current sample."""
    if kernel_size <= 1:
        return values.copy()
    if kernel_size % 2 == 0:
        raise ValueError("median kernel_size must be odd")
    padded = np.pad(values, ((kernel_size - 1, 0), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=kernel_size, axis=0
    )
    return np.median(windows, axis=-1).astype(values.dtype, copy=False)


def _extract_channels(frame: pd.DataFrame, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    masks = []
    for name in names:
        if name in frame:
            column = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
            valid = np.isfinite(column)
            column = np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            column = np.zeros(len(frame), dtype=np.float32)
            valid = np.zeros(len(frame), dtype=bool)
        arrays.append(column)
        masks.append(valid)
    return np.stack(arrays, axis=1), np.stack(masks, axis=1)


def csv_to_signal_arrays(csv_path: str | Path) -> dict[str, np.ndarray]:
    """Extract only approved signal columns; target/metadata columns never enter the cache."""
    frame = pd.read_csv(csv_path)
    required = {"time_perf_counter", "time_s"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing timing columns: {sorted(missing)}")

    perf = pd.to_numeric(frame["time_perf_counter"], errors="coerce").to_numpy(dtype=np.float64)
    relative = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(perf) & np.isfinite(relative)
    frame = frame.loc[finite].copy()
    perf = perf[finite]
    relative = relative[finite]
    order = np.argsort(perf, kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    perf = perf[order]
    relative = relative[order]

    # Keep the final occurrence of duplicate hardware timestamps.
    _, unique_reverse = np.unique(perf[::-1], return_index=True)
    keep = np.sort(len(perf) - 1 - unique_reverse)
    frame = frame.iloc[keep].reset_index(drop=True)
    perf = perf[keep]
    relative = relative[keep]

    # Reconstruct time from the monotonic clock while retaining the saved relative-time
    # origin. In these recordings that origin is the start of the pre-buffer, not the cue.
    zero_index = int(np.argmin(np.abs(relative)))
    cue_perf = perf[zero_index] - relative[zero_index]
    time_s = perf - cue_perf
    emg, emg_mask = _extract_channels(frame, EMG_COLUMNS)
    imu, imu_mask = _extract_channels(frame, IMU_COLUMNS)
    return {
        "time_s": time_s.astype(np.float64),
        "emg": emg,
        "emg_mask": emg_mask,
        "imu": imu,
        "imu_mask": imu_mask,
    }


def previous_sample_resample(
    time_s: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly causal zero-order-hold resampling."""
    indices = np.searchsorted(time_s, grid, side="right") - 1
    in_range = (indices >= 0) & (indices < len(time_s))
    safe_indices = np.clip(indices, 0, max(len(time_s) - 1, 0))
    sampled = values[safe_indices].copy()
    sampled_valid = valid[safe_indices].copy()
    sampled[~in_range] = 0.0
    sampled_valid[~in_range] = False
    return sampled, sampled_valid


@dataclass
class RobustScaler:
    emg_center: np.ndarray
    emg_scale: np.ndarray
    imu_center: np.ndarray
    imu_scale: np.ndarray
    emg_log1p: bool = True
    # Per-session EMG gain references, keyed by participant_id. A single pooled
    # scale cannot remove a per-session multiplicative gain, and measured EMG
    # amplitude varies 2.1x-7.3x between a1 sessions from electrode impedance,
    # skin preparation, adiposity and placement. Fitted on training trials only.
    session_keys: tuple[str, ...] = ()
    session_reference: np.ndarray | None = None
    session_fallback: np.ndarray | None = None
    emg_derived: bool = False
    emg_derivative: bool = False
    emg_derivative_window: int = 21
    emg_sample_rate_hz: float = 148.14814813792788

    @classmethod
    def load(cls, path: str | Path) -> "RobustScaler":
        data = np.load(path, allow_pickle=False)
        keys: tuple[str, ...] = ()
        reference = None
        fallback = None
        if "session_keys" in data:
            keys = tuple(str(key) for key in data["session_keys"])
            reference = data["session_reference"]
            fallback = data["session_fallback"]
        return cls(
            emg_center=data["emg_center"],
            emg_scale=data["emg_scale"],
            imu_center=data["imu_center"],
            imu_scale=data["imu_scale"],
            emg_log1p=bool(data["emg_log1p"].item()),
            session_keys=keys,
            session_reference=reference,
            session_fallback=fallback,
            emg_derived=bool(data["emg_derived"].item())
            if "emg_derived" in data
            else False,
            emg_derivative=bool(data["emg_derivative"].item())
            if "emg_derivative" in data
            else False,
            emg_derivative_window=int(data["emg_derivative_window"].item())
            if "emg_derivative_window" in data
            else 21,
            emg_sample_rate_hz=float(data["emg_sample_rate_hz"].item())
            if "emg_sample_rate_hz" in data
            else 148.14814813792788,
        )

    def emg_session_reference(self, participant_id: str | None) -> np.ndarray | None:
        """Gain reference for one session, or the cross-session median if the
        session was never seen in training (an unseen participant at test time).
        """
        if self.session_reference is None:
            return None
        if participant_id is not None:
            for index, key in enumerate(self.session_keys):
                if key == participant_id:
                    return self.session_reference[index]
        return self.session_fallback

    def transform_emg(
        self,
        values: np.ndarray,
        participant_id: str | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        # Imported lazily: grid_trajectory imports this module at load time.
        from .grid_trajectory import raw_emg_features

        if mask is None:
            mask = np.ones(values.shape, dtype=bool)
        features = raw_emg_features(
            values,
            mask,
            self.emg_session_reference(participant_id),
            self.emg_derived,
            self.emg_log1p,
            derivative=self.emg_derivative,
            sample_rate_hz=self.emg_sample_rate_hz,
            derivative_window=self.emg_derivative_window,
        )
        return (features - self.emg_center) / self.emg_scale

    def transform_imu(self, values: np.ndarray) -> np.ndarray:
        return (values - self.imu_center) / self.imu_scale


def robust_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    center = np.nan_to_num(center, nan=0.0)
    return center.astype(np.float32), scale.astype(np.float32)
