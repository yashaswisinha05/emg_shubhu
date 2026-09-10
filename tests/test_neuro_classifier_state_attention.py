import numpy as np
import torch

from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from emg_touch.models.reach_grasp_neuro_classifier_attention import (
    NeuroClassifierConditionedAttention,
)
from emg_touch.models.reach_grasp_state_attention_v2 import (
    StateConditionedAttentionV2,
)
from emg_touch.state_attention_inference import StateAttentionStream


def model_args(future_steps=3):
    return {"modality": "emg+imu", "width": 16, "patch": 4, "stride": 2,
            "layers": 1, "heads": 4, "dropout": 0.,
            "future_steps": future_steps, "predict_click": True,
            "predict_final_pose": False, "react_context": 10}


def stats(emg_mean=0., imu_mean=0.):
    return {
        "emg": {"mean": np.full(8, emg_mean), "std": np.full(8, 2.)},
        "imu": {"mean": np.full(24, imu_mean), "std": np.full(24, 3.)},
        "position": {"mean": np.zeros(3), "std": np.ones(3)},
    }


def make_model():
    teacher_args = model_args()
    teacher_args["predict_final_pose"] = True
    teacher = NeuromuscularFutureGripperPoseModel(**teacher_args)
    motion = StateConditionedAttentionV2(**model_args())
    return NeuroClassifierConditionedAttention(
        teacher, motion, stats(1., 2.), stats(3., 5.)), teacher_args


def test_frozen_classifier_is_authoritative_and_conditions_motion():
    model, _ = make_model()
    model.train()
    assert model.classifier.training is False
    assert all(not parameter.requires_grad for parameter in model.classifier.parameters())
    emg = torch.randn(2, 24, 16)
    imu = torch.randn(2, 24, 48)
    output = model(emg, imu)
    converted = model._classifier_inputs(emg, imu)
    with torch.no_grad():
        expected = model.classifier(*converted)["gripper_state_logits"]
    torch.testing.assert_close(output["gripper_state_logits"], expected)
    torch.testing.assert_close(output["conditioning_state_probability"],
                               expected.softmax(-1))
    assert output["position"].shape == (2, 24, 3)
    assert output["click"].shape == (2, 24, 2)
    assert output["future_position"].shape == (2, 24, 3, 3)
    assert output["student_gripper_state_logits"].shape == (2, 24, 2)


def test_hybrid_checkpoint_streams_at_widescreen_resolution(tmp_path):
    model, teacher_args = make_model()
    path = tmp_path / "hybrid.pt"
    torch.save({
        "format": "neuro_classifier_state_attention_v1",
        "state_dict": model.state_dict(),
        "model_args": model_args(),
        "classifier_model_args": teacher_args,
        "classifier_normalization": stats(1., 2.),
        "normalization": stats(3., 5.),
        "preprocessing": {"raw_rate_hz": 1000., "rate_hz": 100.,
                          "gap_s": .02},
    }, path)
    stream = StateAttentionStream(path, "cpu", warmup_ms=20., context_ms=100.)
    result = None
    for index in range(40):
        value = stream.update(index / 1000, np.ones(4), np.ones(24))
        result = value if value is not None else result
    assert result["valid"] is True
    normalized = np.asarray(result["pixel_normalized_xy"])
    np.testing.assert_allclose(result["pixel_xy"], normalized * [1920, 1080])
    assert len(result["future_positions_m"]) == 3
