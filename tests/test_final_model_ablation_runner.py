import subprocess
import sys
import importlib.util
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


def test_hold_baseline_is_a_lower_is_better_metric():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_final_model_ablations.py"
    spec = importlib.util.spec_from_file_location("final_ablations", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "future_200ms_hold_cm" in module.LOWER_IS_BETTER
