import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.evaluate_residual_gru_roots import (
    classification_summary,
    discrete_frechet,
    downsample_path,
    normalized_trial_progress,
    position_rmse,
)


def test_multi_root_evaluator_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_residual_gru_roots.py", "--help"],
        cwd=root, capture_output=True, text=True)
    assert result.returncode == 0
    assert "multiple roots" in result.stdout


def test_classification_summary_reports_accuracy_and_handles_empty_bin():
    assert classification_summary([], []) is None
    assert classification_summary([0, 0, 1, 1], [0, 1, 1, 1]) == {
        "accuracy": .75,
        "frames": 4,
    }


def test_position_rmse_reports_vector_and_axis_errors_in_cm():
    actual = np.zeros((2, 3))
    predicted = np.asarray([[.03, .04, 0.], [.03, .04, 0.]])
    report = position_rmse(predicted, actual)
    assert report["rmse_3d_cm"] == 5.
    assert report["rmse_axis_cm"] == {"x": 3., "y": 4., "z": 0.}


def test_discrete_frechet_is_zero_for_identical_path_and_tracks_offset():
    path = np.asarray([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    assert discrete_frechet(path, path) == 0.
    assert discrete_frechet(path, path + [0., 2., 0.]) == 2.
    sampled = downsample_path(np.arange(30).reshape(10, 3), 4)
    assert len(sampled) == 4
    assert np.array_equal(sampled[[0, -1]], [[0, 1, 2], [27, 28, 29]])


def test_trial_progress_matches_training_batch_definition():
    assert np.allclose(normalized_trial_progress(5), [0., .25, .5, .75, 1.])
