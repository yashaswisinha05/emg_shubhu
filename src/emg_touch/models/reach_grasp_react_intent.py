"""ReactEMG-inspired causal masking, extended to wearable future pose.

Independent implementation; not the authors' architecture or pretrained weights.
Action labels are accepted only by the explicit auxiliary training method.
"""
import math

import torch
from torch import nn

from .reach_grasp_future_intent import ReachGraspFutureIntentModel


class CausalEMGIntent(nn.Module):
    def __init__(self, width=64, heads=4, layers=2, dropout=.1, context=100):
        super().__init__()
        self.context = context
        self.history = (context - 1) * layers
        self.signal = nn.Linear(16, width)
        self.action = nn.Embedding(3, width)  # not holding, holding, MASK
        self.mask = nn.Parameter(torch.zeros(width))
        self.modality = nn.Parameter(torch.randn(2, width) * .02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout,
                    activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.reconstruct = nn.Linear(width, 8)
        self.classify = nn.Linear(width, 2)

    def forward(self, emg, actions=None, hidden=None, block_intent_attention=False):
        """``block_intent_attention`` implements ReactEMG's self-supervised-EMG
        task: rather than only substituting the intent MASK embedding (which
        still lets every EMG query attend to a token carrying learned MASK
        content), EMG queries are architecturally forbidden from attending to
        any intent-modality key, so masked-EMG reconstruction in that mode is
        provably driven by EMG history alone."""
        batch, frames, _ = emg.shape
        if actions is None:
            actions = torch.full((batch, frames), 2, device=emg.device, dtype=torch.long)
        signal = self.signal(emg)
        if hidden is not None:
            signal = torch.where(hidden[..., None], self.mask, signal)
        # Interleave modalities. Both tokens at t may attend to each other,
        # but neither can access a token timestamp later than t.
        time = torch.arange(frames, device=emg.device)
        frequency = torch.exp(torch.arange(0, signal.shape[-1], 2,
                    device=emg.device) * (-math.log(10000.) / signal.shape[-1]))
        phase = time[:, None] * frequency
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        tokens = torch.stack((signal + self.modality[0],
                              self.action(actions) + self.modality[1]), 2)
        tokens = (tokens + position[None, :, None]).flatten(1, 2)
        # Bound attention memory for long recordings. Include the complete
        # multilayer receptive field before each output chunk.
        chunks = []
        for start in range(0, frames, self.context):
            end = min(frames, start + self.context)
            left = max(0, start - self.history)
            stamps = time[left:end].repeat_interleave(2)
            lag = stamps[:, None] - stamps[None, :]
            blocked = (lag < 0) | (lag >= self.context)
            if block_intent_attention:
                # Even local indices are EMG tokens, odd are intent tokens
                # (matching the interleave order above): forbid EMG queries
                # from attending to any intent key.
                is_intent = torch.arange(stamps.shape[0], device=emg.device) % 2 == 1
                blocked = blocked | (~is_intent[:, None] & is_intent[None, :])
            chunk = self.encoder(tokens[:, 2 * left:2 * end], mask=blocked)
            chunks.append(chunk[:, 2 * (start - left):])
        encoded = torch.cat(chunks, 1).reshape(batch, frames, 2, -1)
        return {"features": encoded[:, :, 1],
                "holding_logits": self.classify(encoded[:, :, 1]),
                "reconstruction": self.reconstruct(encoded[:, :, 0])}


class ReactFutureIntentModel(ReachGraspFutureIntentModel):
    def __init__(self, react_context=100, **kwargs):
        super().__init__(**kwargs)
        width = kwargs.get("width", 128)
        # The causal intent-alignment branch previously always received the
        # class defaults (layers=2, heads=4, context=100) no matter what depth
        # was requested for the rest of the network -- it now shares the same
        # capacity as the EMG/IMU patch encoders it feeds into.
        self.react = CausalEMGIntent(width=width // 2,
                                     heads=kwargs.get("heads", 4),
                                     layers=kwargs.get("layers", 4),
                                     dropout=kwargs.get("dropout", .1),
                                     context=react_context)
        self.react_to_context = nn.Linear(width // 2, width * 2)
        self.react_current = nn.Linear(width // 2, 3)
        self.motion_velocity = nn.Linear(width * 2, self.future_steps * 3)
        self.motion_mix = nn.Parameter(torch.tensor(-2.2))
        # Start the interaction/context correction at zero; the branch first
        # learns from its own dense holding and reconstruction supervision.
        for layer in (self.react_to_context, self.react_current):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        branch = self.react(emg)  # every action token is MASK at inference
        context = result["context_features"] + self.react_to_context(branch["features"])
        shape = context.shape[:2]
        direct = self.future_position(context).reshape(*shape, self.future_steps, 3)
        velocity = self.motion_velocity(context).reshape(*shape, self.future_steps, 3)
        times = self.future_horizons_ms / 1000
        dt = torch.diff(times, prepend=times.new_zeros(1))
        displacement = (velocity * dt[None, None, :, None]).cumsum(-2)
        integrated = result["position"].unsqueeze(-2) + displacement
        result["future_position"] = torch.lerp(direct, integrated, self.motion_mix.sigmoid())
        result["future_direct_position"] = direct
        result["future_displacement"] = displacement
        result["future_integrated_position"] = integrated
        result["future_orientation_6d"] = self.future_orientation(context).reshape(
            *shape, self.future_steps, 6)
        result["future_event_logits"] = self.future_event(context)
        result["intent_time_logits"] = self.intent_time(context).reshape(
            *shape, 2, self.intent_time_bins)
        result["logits"] = result["logits"] + self.react_current(branch["features"])
        result["react_holding_logits"] = branch["holding_logits"]
        result["future_endpoint"] = result["future_position"][..., -1, :]
        return result
