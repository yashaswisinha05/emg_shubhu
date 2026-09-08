import numpy as np
import torch

from emg_touch.data.reach_grasp import SENSORS, preprocess
from emg_touch.live_reach_grasp import (LiveReachGraspPredictor,
                                        LiveReachGraspPreprocessor)
from emg_touch.models.reach_grasp_orientation_hybrid import ReachGraspOrientationHybrid
from scripts.live_reach_grasp_orientation import channel_indices
from tests.test_reach_grasp import frame, settings
from tests.test_reach_grasp_orientation import add_orientation


def raw_row(data, index):
    emg = [data.loc[index, f"EMG 1_{sensor}"] for sensor in SENSORS]
    imu = [data.loc[index, f"{kind} {axis}_{sensor}"] for sensor in SENSORS
           for kind in ["ACC", "GYRO"] for axis in "XYZ"]
    return emg, imu


def test_online_preprocessing_matches_offline_complete_stream(tmp_path):
    data = add_orientation(frame(), 20)
    path = tmp_path / "trial.csv"
    data.to_csv(path, index=False)
    offline = preprocess(path, settings())
    online = LiveReachGraspPreprocessor(settings())
    for index, stamp in enumerate(data["time_perf_counter"]):
        emg, imu = raw_row(data, index)
        online.add_sample(stamp, emg, imu)
    time, emg, imu, ev, iv = online.arrays()
    np.testing.assert_allclose(time - time[0], offline["time"], atol=1e-8)
    np.testing.assert_allclose(emg, offline["emg"], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(imu, offline["imu"], atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(ev, offline["emg_valid"])
    np.testing.assert_array_equal(iv, offline["imu_valid"])


def test_live_predictor_outputs_every_head_without_vive(tmp_path):
    args = {"modality": "emg+imu", "width": 16, "patch": 8, "stride": 2,
            "layers": 1, "heads": 2, "dropout": .1}
    model = ReachGraspOrientationHybrid(**args)
    stats = {key: {"mean": [0.] * width, "std": [1.] * width}
             for key, width in [("emg", 8), ("imu", 24), ("position", 3)]}
    decoder = {"holding": {"low": .3, "high": .7, "persistence_s": .03},
               "pulse_s": .1,
               "events": [{"local_weight": .5, "threshold": .5, "persistence_s": .03},
                          {"local_weight": .5, "threshold": .5, "persistence_s": .03}]}
    checkpoint = tmp_path / "model.pt"
    torch.save({"format": "reach_grasp_orientation_hybrid_v1", "model_args": args,
                "state_dict": model.state_dict(), "normalization": stats,
                "preprocessing": settings(), "hybrid_event_decoder": decoder}, checkpoint)
    predictor = LiveReachGraspPredictor(checkpoint, "cpu", warmup_ms=50)
    for index in range(100):
        predictor.add_sample(10 + index / 1000, np.ones(4) * .01, np.ones(24))
    result = predictor.predict()
    assert result["valid"]
    assert set(result["position_m"]) == {"x", "y", "z"}
    assert len(result["orientation_quaternion_wxyz"]) == 4
    assert set(result["orientation_deg_zyx"]) == {"yaw", "pitch", "roll"}
    assert set(result["triggered"]) == {"grasp", "release"}


def test_live_channel_mapping_rejects_missing_sensor():
    names = [f"EMG 1_{sensor}" for sensor in SENSORS] + [
        f"{kind} {axis}_{sensor}" for sensor in SENSORS
        for kind in ["ACC", "GYRO"] for axis in "XYZ"]
    emg, imu = channel_indices(names)
    assert emg == [0, 1, 2, 3] and len(imu) == 24
    try:
        channel_indices(names[:-1])
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing channel should fail")
