import numpy as np
import torch

from emg_touch.data.reach_grasp import SENSORS, preprocess
from emg_touch.live_reach_grasp import (LiveReachGraspPredictor,
                                        LiveReachGraspPreprocessor,
                                        LiveThreeRGripperController)
from emg_touch.physics.manipulator_ik import ThreeRManipulator
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


def prediction(world, grasp=False, release=False):
    return {"valid": True,
            "position_m": {axis: float(value) for axis, value in zip("xyz", world)},
            "triggered": {"grasp": grasp, "release": release},
            "trigger_time_s": {"grasp": 1. if grasp else None,
                               "release": 2. if release else None}}


def test_live_ik_uses_model_endpoint_and_gripper_transitions():
    arm = ThreeRManipulator((.5, .6))
    angles = np.array([.3, .2, 1.0])
    base = np.array([1., 2., 3.])
    world = base + arm.forward(angles)[-1]
    controller = LiveThreeRGripperController(
        (.5, .6), np.degrees(angles), base_world=base)
    opened = controller.attach(prediction(world))
    np.testing.assert_allclose(
        opened["manipulator"]["projected_endpoint_m"], world - base, atol=1e-7)
    assert opened["gripper"]["state"] == "open"
    assert opened["manipulator"]["ik_fk_residual_cm"] < 1e-5
    closed = controller.attach(prediction(world, grasp=True))
    assert closed["gripper"]["state"] == "closed"
    assert closed["gripper"]["command"] == "close"
    assert np.isclose(np.linalg.norm(np.diff(
        closed["manipulator"]["gripper_jaw_points_m"], axis=0)), .015)
    held = controller.attach(prediction(world))
    assert held["gripper"] == {"state": "closed", "command": "hold", "width_m": .015}
    released = controller.attach(prediction(world, release=True))
    assert released["gripper"]["state"] == "open"
    assert released["gripper"]["command"] == "open"


def test_synthetic_ik_anchor_is_explicit_and_resettable():
    controller = LiveThreeRGripperController(initial_joint_deg=(0, 20, 90))
    first = controller.attach(prediction(np.array([4., -2., 1.])))
    assert first["manipulator"]["calibration"] == "synthetic initial pose"
    home = controller.arm.forward(np.radians([0, 20, 90]))[-1]
    np.testing.assert_allclose(first["manipulator"]["requested_endpoint_m"], home)
    controller.attach(prediction(np.array([4.1, -2., 1.]), grasp=True))
    controller.reset()
    assert controller.gripper == "open" and controller.first_world is None
