"""Signal processing shared by segmentation (dataset.py) and QC (qc.py).

Everything here works on plain arrays -- ``x`` is ``(n_samples, n_channels)``
-- so neither caller has to import the other.

The important function is :func:`structure_features`. Scaling every channel by
the same constant leaves its output unchanged, which is what lets you claim a
classifier built on it is not reading off contraction level. There is a test
for exactly that in ``test_features.py``; if you change this file, run it.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

#: Number of values returned by structure_features() for C channels:
#: C*(C+1)/2 covariance terms + C ripple terms. 14 for C=4.
EMG_BAND = (20.0, 450.0)
ENVELOPE_CUTOFF_HZ = 10.0
RIPPLE_BAND = (0.5, 5.0)


def _padlen(sos: np.ndarray, n: int) -> int:
    return int(min(3 * (2 * sos.shape[0] + 1), max(0, n - 1)))


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0, padlen=_padlen(sos, x.shape[0]))


def lowpass(x: np.ndarray, fs: float, cut: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    cut = min(cut, nyq * 0.95)
    sos = signal.butter(order, cut / nyq, btype="low", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0, padlen=_padlen(sos, x.shape[0]))


def preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    """Mean-remove and band-pass. Envelope-rate recordings pass through."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    x = x - x.mean(axis=0, keepdims=True)
    if fs >= 1000.0:
        x = bandpass(x, fs, *EMG_BAND)
    return x


def envelope(x: np.ndarray, fs: float, target_fs: float = 100.0) -> tuple[np.ndarray, float]:
    """Rectify, low-pass at 10 Hz, decimate to ~``target_fs``.

    The decimation is not cosmetic. A 0.5-5 Hz band-pass applied at 2 kHz sits
    at a normalised frequency of 5e-4, where a Butterworth design is poorly
    conditioned even in second-order-section form. Bringing the envelope to
    ~100 Hz first puts that band at a well-behaved 0.01-0.1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    smoothed = lowpass(np.abs(x), fs, ENVELOPE_CUTOFF_HZ)
    step = max(1, int(round(fs / target_fs)))
    return smoothed[::step], fs / step


def amplitude_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Per-channel log RMS. Deliberately the crudest thing that could work."""
    y = preprocess(x, fs)
    return np.log(np.sqrt((y**2).mean(axis=0)) + 1e-12)


def structure_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Features invariant to a global gain on the signal.

    Scaling every channel by one constant leaves all of these unchanged:

      * log-Euclidean embedding of the TRACE-NORMALISED channel covariance
        (C*(C+1)/2 values) -- the shape of co-activation, which changes under
        load because the object's reaction torque recruits the extensors
      * envelope ripple: std/mean of the 0.5-5 Hz component of each channel's
        envelope (C values) -- holding a real object is closed-loop grip-force
        control and fluctuates; squeezing air is open-loop and smooth
    """
    y = preprocess(x, fs)

    cov = np.atleast_2d(np.cov(y, rowvar=False))
    cov = cov / (np.trace(cov) + 1e-24)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    log_cov = (eigenvectors * np.log(np.clip(eigenvalues, 1e-12, None))) @ eigenvectors.T
    upper = log_cov[np.triu_indices(log_cov.shape[0])]

    env, fs_env = envelope(y, fs)
    ripple = bandpass(env, fs_env, *RIPPLE_BAND, order=2).std(axis=0) / (
        env.mean(axis=0) + 1e-12
    )
    return np.concatenate([upper, ripple])


def detect_onset(
    env: np.ndarray, fs_env: float, threshold: float, persistence_s: float
) -> int | None:
    """First index where the mean envelope stays above ``threshold``.

    Returns ``None`` when no sustained crossing exists -- which is the expected
    outcome for a NOTHING trial, and a dropped trial for any other condition.
    """
    trace = env.mean(axis=1) if env.ndim > 1 else env
    above = trace > threshold
    need = max(1, int(round(persistence_s * fs_env)))
    if above.size < need:
        return None
    # A run of `need` consecutive True values ending at i means the onset is
    # at i - need + 1; a cumulative-sum window finds the first such run.
    counts = np.convolve(above.astype(int), np.ones(need, dtype=int), mode="valid")
    hits = np.flatnonzero(counts == need)
    return int(hits[0]) if hits.size else None
