import subprocess
import sys
from pathlib import Path


def test_multi_root_evaluator_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_residual_gru_roots.py", "--help"],
        cwd=root, capture_output=True, text=True)
    assert result.returncode == 0
    assert "multiple roots" in result.stdout
