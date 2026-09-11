import numpy as np
import pandas as pd
import torch

from emg_touch.data.mahg_emg import make_windows, preprocess, subject_for
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from scripts.finetune_mahg_emg import (
    TransferPatch, initialize_from_checkpoint, source_spec,
)


def test_subject_grouping():
    from pathlib import Path
    assert subject_for(Path("EMGData1.csv")) == 1
    assert subject_for(Path("EMGData10.csv")) == 1
    assert subject_for(Path("EMGData11.csv")) == 2
    assert subject_for(Path("EMGData100.csv")) == 10


def test_preprocessing_and_windows_handle_bounded_missing_values(tmp_path):
    rate, count = 1259.0, 2600
    time = np.arange(count) / rate
    frame = pd.DataFrame({
        f"EMG_Channel_{channel}": .02 * np.sin(2 * np.pi * (40 + channel) * time)
        for channel in range(1, 5)
    })
    frame["State"] = np.where(np.arange(count) < count // 2, "GESTURE1", "GESTURE2")
    frame.loc[100:105, "EMG_Channel_1"] = np.nan
    path = tmp_path / "EMGData1.csv"
    frame.to_csv(path, index=False)
    features, flags, labels, valid = preprocess(path, raw_rate_hz=rate)
    windows, targets = make_windows(features, flags, labels, valid,
                                    window_steps=30, stride_steps=5)
    assert features.shape[1] == flags.shape[1] == 8
    assert windows.shape[1:] == (30, 16)
    assert set(targets) == {"GESTURE1", "GESTURE2"}
    assert np.isfinite(windows).all()


def test_neuromuscular_future_emg_encoder_transfer():
    args = {"modality": "emg+imu", "width": 16, "patch": 4, "stride": 2,
            "layers": 1, "heads": 4, "dropout": 0., "react_context": 10,
            "predict_click": True, "predict_final_pose": True, "future_steps": 2}
    source = NeuromuscularFutureGripperPoseModel(**args)
    checkpoint = {"format": "gripper_neuromuscular_future_v1",
                  "model_args": args, "state_dict": source.state_dict()}
    kind, model_args = source_spec(checkpoint)
    transfer = TransferPatch(**model_args, classes=5)
    initialize_from_checkpoint(transfer, checkpoint, kind)
    assert kind == "neuromuscular_patch"
    assert torch.equal(transfer.encoder.input[0].weight, source.emg.input[0].weight)
    assert transfer(torch.zeros(2, 30, 16)).shape == (2, 5)
