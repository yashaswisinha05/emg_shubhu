import numpy as np
import torch

from emg_touch.models.reach_grasp_future_intent import ReachGraspFutureIntentModel
from scripts.train_reach_grasp_future_intent import future_probability, time_distribution


def test_future_intent_shapes_and_causality():
    torch.manual_seed(4)
    model = ReachGraspFutureIntentModel(
        width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.,
        future_horizons_ms=(100, 500, 1000)).eval()
    emg, imu = torch.randn(2, 30, 16), torch.randn(2, 30, 48)
    emg[..., 8:] = 1
    imu[..., 24:] = 1
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 20:, :8] += 50
    changed_imu[:, 20:, :24] += 50
    with torch.no_grad():
        output = model(emg, imu)
        changed = model(changed_emg, changed_imu)
    assert output["future_position"].shape == (2, 30, 3, 3)
    assert output["future_orientation_6d"].shape == (2, 30, 3, 6)
    assert output["future_event_logits"].shape == (2, 30, 2)
    assert output["intent_time_logits"].shape == (2, 30, 2, 5)
    torch.testing.assert_close(output["future_position"][:, :20],
                               changed["future_position"][:, :20])


def test_future_probability_is_high_only_when_event_is_in_horizon():
    time = np.array([0., .5, 1., 1.5, 2.])
    probability = future_probability(time, 1.5, 1., .2)
    assert probability[2] > .99
    assert probability[0] < .01
    assert probability[-1] < .01


def test_time_distribution_has_no_event_class():
    target = time_distribution(np.array([0., 3.]), [1., 2.],
                               [0., .1, .5, 1.], 1., .2)
    np.testing.assert_allclose(target.sum(-1), 1, atol=1e-6)
    assert target[1, 0, -1] == 1

