import numpy as np
import torch

from emg_touch.models.emg_grasp_onset import EMGGraspOnsetDetector
from scripts.train_emg_grasp_onset import detect, soft_grasp_target


def test_model_is_causal():
    torch.manual_seed(0)
    model = EMGGraspOnsetDetector(width=16, dropout=0., dilations=(1, 2)).eval()
    x = torch.randn(2, 40, 16)
    x[..., 8:] = 1
    changed = x.clone()
    changed[:, 25:, :8] += 100
    with torch.no_grad():
        first = model(x)["grasp_logit"]
        second = model(changed)["grasp_logit"]
    torch.testing.assert_close(first[:, :25], second[:, :25])


def test_soft_target_centres_on_grasp():
    time = np.arange(0, 1, .01)
    target = soft_grasp_target(time, .5, .2)
    assert np.argmax(target) == 50
    assert target[50] == 1
    assert target[0] == 0


def test_detector_requires_persistence():
    time = np.arange(0, .2, .01)
    probability = np.zeros_like(time)
    probability[5:10] = .9
    found = detect(time, probability, np.ones_like(time, bool), .8, .03)
    assert found == [time[5]]

