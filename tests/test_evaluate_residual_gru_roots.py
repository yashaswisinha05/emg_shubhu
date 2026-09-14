import subprocess
import sys
from pathlib import Path

from scripts.evaluate_residual_gru_roots import classification_summary


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
