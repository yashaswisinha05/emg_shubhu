from __future__ import annotations

import numpy as np

from scripts.visualize_vive_trajectory_3d import prepare_vive_trajectory


def test_prepare_vive_trajectory_uses_onset_to_touch_and_relative_origin() -> None:
    position = np.array([
        [8.0, 2.0, -1.0],
        [8.0, 2.0, -1.0],
        [8.0, 2.0, -1.0],
        [8.03, 2.04, -1.0],
        [8.06, 2.08, -1.0],
    ])
    result = prepare_vive_trajectory(
        position, onset=2, sample_rate_hz=10.0, maximum_frames=20
    )
    np.testing.assert_allclose(result["points"][0], np.zeros(3))
    np.testing.assert_allclose(result["points"][-1], [0.06, 0.08, 0.0])
    np.testing.assert_allclose(result["path_length_m"], 0.1)
    np.testing.assert_allclose(result["displacement_m"], 0.1)
    np.testing.assert_allclose(result["duration_s"], 0.2)
    np.testing.assert_allclose(result["time_s"], [0.0, 0.1, 0.2])


def test_prepare_vive_trajectory_downsamples_but_preserves_endpoints() -> None:
    time = np.linspace(0.0, 1.0, 101)
    position = np.stack((time, time**2, np.zeros_like(time)), axis=-1)
    result = prepare_vive_trajectory(
        position,
        onset=10,
        sample_rate_hz=100.0,
        maximum_frames=12,
        relative_to_onset=False,
    )
    assert result["points"].shape == (12, 3)
    np.testing.assert_allclose(result["points"][0], position[10])
    np.testing.assert_allclose(result["points"][-1], position[-1])
    np.testing.assert_allclose(result["time_s"][[0, -1]], [0.0, 0.9])


def test_downsampling_does_not_change_full_resolution_path_length() -> None:
    theta = np.linspace(0.0, np.pi, 101)
    position = np.stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)), axis=-1)
    result = prepare_vive_trajectory(
        position, onset=0, sample_rate_hz=100.0, maximum_frames=2
    )
    assert result["points"].shape == (2, 3)
    np.testing.assert_allclose(result["displacement_m"], 2.0, atol=1e-7)
    np.testing.assert_allclose(result["path_length_m"], np.pi, rtol=2e-4)


def test_prepare_vive_trajectory_can_include_pre_onset() -> None:
    position = np.stack((np.arange(6.0), np.zeros(6), np.zeros(6)), axis=-1)
    result = prepare_vive_trajectory(
        position,
        onset=3,
        sample_rate_hz=2.0,
        maximum_frames=20,
        include_pre_onset=True,
    )
    assert len(result["points"]) == 6
    np.testing.assert_allclose(result["points"][3], np.zeros(3))
    np.testing.assert_allclose(result["points"][0], [-3.0, 0.0, 0.0])
    np.testing.assert_allclose(result["duration_s"], 2.5)
