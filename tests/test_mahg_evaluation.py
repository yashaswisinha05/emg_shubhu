import subprocess
import sys
from pathlib import Path


def test_mahg_evaluator_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/test_mahg_emg.py", "--help"], cwd=root,
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "held-out subjects" in result.stdout
