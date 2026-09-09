import json

from scripts.visualize_emg_grasp_franka import choose_trial


def test_choose_trial_uses_checkpoint_test_split(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    first = tmp_path / "trial_001.csv"
    second = tmp_path / "trial_002.csv"
    first.write_text("x\n1\n")
    second.write_text("x\n2\n")
    (run / "splits.json").write_text(json.dumps(
        {"train": [], "validation": [], "test": [str(first), str(second)]}))
    selected, source = choose_trial(checkpoint, None, None, 7)
    assert selected in {first, second}
    assert "held-out test" in source


def test_explicit_trial_takes_priority(tmp_path):
    path = tmp_path / "trial_009.csv"
    path.write_text("x\n1\n")
    selected, source = choose_trial(tmp_path / "missing.pt", path, None, 1)
    assert selected == path.resolve()
    assert "explicit" in source

