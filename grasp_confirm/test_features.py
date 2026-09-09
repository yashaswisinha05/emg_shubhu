"""Tests for the property the whole design rests on.

    python -m pytest grasp_confirm/test_features.py -q

If structure_features() stops being invariant to a global gain, every claim
about the model "not using amplitude" becomes false, silently. Run this after
touching features.py.
"""

from __future__ import annotations

import numpy as np

from .features import (
    amplitude_features,
    detect_onset,
    envelope,
    structure_features,
)

FS = 2000.0


def _signal(n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 4)) * np.array([1.0, 0.9, 0.5, 0.4])


def test_structure_features_are_gain_invariant():
    x = _signal()
    for gain in (0.13, 7.3, 250.0):
        assert np.allclose(structure_features(x, FS), structure_features(x * gain, FS), atol=1e-7)


def test_amplitude_features_track_gain():
    x = _signal()
    gain = 7.3
    delta = amplitude_features(x * gain, FS) - amplitude_features(x, FS)
    assert np.allclose(delta, np.log(gain), atol=1e-6)


def test_structure_features_respond_to_envelope_ripple():
    x = _signal()
    t = np.arange(x.shape[0]) / FS
    modulated = x * (1 + 0.35 * np.sin(2 * np.pi * 2.3 * t))[:, None]
    flat_ripple = structure_features(x, FS)[-4:].mean()
    mod_ripple = structure_features(modulated, FS)[-4:].mean()
    assert mod_ripple > flat_ripple * 1.3


def test_structure_features_respond_to_coactivation_change():
    """Changing the flexor:extensor ratio must move the covariance features."""
    x = _signal()
    y = x * np.array([1.0, 0.9, 1.5, 1.2])  # extensors up, as under load
    assert not np.allclose(structure_features(x, FS)[:10], structure_features(y, FS)[:10], atol=1e-3)


def test_features_are_finite():
    x = _signal()
    assert np.isfinite(structure_features(x, FS)).all()
    assert np.isfinite(amplitude_features(x, FS)).all()


def test_detect_onset_finds_a_burst():
    rng = np.random.default_rng(1)
    quiet = rng.standard_normal((2000, 4)) * 0.01
    loud = rng.standard_normal((2000, 4)) * 0.5
    env, fs_env = envelope(np.vstack([quiet, loud]), FS)
    baseline = env[: int(0.8 * fs_env)].mean()
    onset = detect_onset(env, fs_env, threshold=baseline * 5, persistence_s=0.1)
    assert onset is not None
    # The burst starts at 1.0 s; allow the 10 Hz envelope smoothing some lag.
    assert 0.9 < onset / fs_env < 1.2


def test_detect_onset_returns_none_when_quiet():
    rng = np.random.default_rng(2)
    env, fs_env = envelope(rng.standard_normal((4000, 4)) * 0.01, FS)
    assert detect_onset(env, fs_env, threshold=10.0, persistence_s=0.1) is None
