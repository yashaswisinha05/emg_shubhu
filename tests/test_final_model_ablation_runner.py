import subprocess
import sys
from pathlib import Path


def test_final_model_ablation_runner_help_lists_core_variants():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_final_model_ablations.py", "--help"],
        cwd=root, capture_output=True, text=True)
    assert result.returncode == 0
    assert "no_state_conditioning" in result.stdout
    assert "no_state_loss" in result.stdout
    assert "no_current_position_loss" in result.stdout
    assert "no_pixel_loss" in result.stdout
    assert "no_future_imu_loss" in result.stdout
    assert "frozen_pretrained" in result.stdout
