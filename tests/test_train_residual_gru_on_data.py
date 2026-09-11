import os
import subprocess
from pathlib import Path


def test_training_wrapper_requires_checkpoint_output_and_root():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "train_residual_gru_on_data.sh"
    result = subprocess.run([str(script)], cwd=root, text=True,
                            capture_output=True, env=os.environ.copy())
    assert result.returncode == 2
    assert "CLASSIFIER_CHECKPOINT OUTPUT_DIR DATA_ROOT" in result.stderr
