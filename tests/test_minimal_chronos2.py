from types import SimpleNamespace

import torch
from torch import nn

from minimal_emg_imu.chronos2_model import FrozenChronos2Branch


class FakeChronos2(nn.Module):
    def __init__(self, width=12, patch=4):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(d_model=width)
        self.chronos_config = SimpleNamespace(
            input_patch_size=patch, input_patch_stride=patch)

    def encode(self, context, group_ids, num_output_patches=1):
        batch, frames = context.shape
        patches = context.reshape(batch, frames // 4, 4).mean(-1, keepdim=True)
        patches = patches.expand(-1, -1, self.config.d_model)
        special = patches.new_zeros(batch, 2, self.config.d_model)
        output = SimpleNamespace(last_hidden_state=torch.cat((patches, special), 1))
        return output, None, None, patches.shape[1]


def test_chronos2_branch_is_patch_endpoint_causal():
    branch = FrozenChronos2Branch(
        FakeChronos2(), inputs=3, width=8,
        context_frames=16, chunk_frames=16).eval()
    signal = torch.randn(1, 16, 3)
    changed = signal.clone()
    changed[:, 9:] += 50
    with torch.no_grad():
        before = branch(signal)
        after = branch(changed)
    torch.testing.assert_close(before[:, :9], after[:, :9])


def test_frozen_foundation_is_not_saved_or_optimized():
    foundation = FakeChronos2()
    branch = FrozenChronos2Branch(
        foundation, inputs=3, width=8,
        context_frames=16, chunk_frames=16)
    assert not any("foundation" in name for name in branch.state_dict())
    assert all(parameter is not foundation.anchor for parameter in branch.parameters())
