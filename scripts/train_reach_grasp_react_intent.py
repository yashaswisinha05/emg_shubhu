#!/usr/bin/env python3
"""Train causal ReactEMG-inspired holding alignment and one-second future pose."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from torch.nn import functional as F
from emg_touch.models.reach_grasp_react_intent import ReactFutureIntentModel
from scripts import train_reach_grasp_future_intent as training


def spans(shape, device, probability=.25, length=10):
    """Random contiguous blocks (100 ms at the existing processed 100 Hz)."""
    batch, frames = shape
    offset = int(torch.randint(length, (), device=device))
    blocks = torch.rand(batch, (frames + offset) // length + 1, device=device) < probability
    return blocks.repeat_interleave(length, 1)[:, offset:offset + frames]


def masked_mean(value, mask):
    mask = mask.expand_as(value)
    return value[mask].mean() if mask.any() else value.sum() * 0


class ReactTraining:
    @staticmethod
    def configure(parser):
        parser.set_defaults(output_dir=Path("runs/reach_grasp_react_intent_seed42"))
        parser.add_argument("--masked-weight", type=float, default=.15)
        parser.add_argument("--emg-holding-weight", type=float, default=.2)
        parser.add_argument("--stability-weight", type=float, default=.03)
        parser.add_argument("--motion-weight", type=float, default=.2)

    @staticmethod
    def loss(model, output, batch, usable, args):
        emg = batch["emg"]
        target = batch["labels"][..., 0].long()
        certain = batch["label_mask"][..., 0].bool() & batch["emg_usable"]
        # Always train deployment mode: no supplied labels or future signals.
        dense = masked_mean(F.cross_entropy(output["react_holding_logits"].transpose(1, 2),
                                          target, reduction="none"), certain)
        masked = dense * 0
        if args.masked_weight:
            hidden = spans(target.shape, emg.device)
            action_hidden = spans(target.shape, emg.device, probability=.6)
            mode = int(torch.randint(3, (), device=emg.device))
            if mode == 0:  # no labels, reconstruct masked EMG
                action_hidden = torch.ones_like(action_hidden)
            elif mode == 1:  # aligned masking of both modalities
                action_hidden = hidden.clone()
            # No uncertain annotation or invalid-frame label is visible.
            action_hidden |= ~certain
            actions = torch.where(action_hidden, 2, target)
            corrupted = emg.clone()
            # Apply small noise only on standardized values with valid evidence.
            corrupted[..., :8] += .03 * torch.randn_like(emg[..., :8]) * emg[..., 8:]
            aux = model.react(corrupted, actions=actions, hidden=hidden)
            reconstruction_mask = hidden[..., None] & emg[..., 8:].bool()
            reconstruction = masked_mean((aux["reconstruction"] - emg[..., :8]).square(),
                                         reconstruction_mask)
            classification = masked_mean(F.cross_entropy(aux["holding_logits"].transpose(1, 2),
                          target, reduction="none"), action_hidden & certain)
            masked = reconstruction + classification
        # Do not penalize real transitions or bridge invalid/uncertain gaps.
        pair = certain[:, 1:] & certain[:, :-1] & (target[:, 1:] == target[:, :-1])
        holding = output["logits"][..., 0].sigmoid()
        stability = masked_mean((holding[:, 1:] - holding[:, :-1]).square(), pair)
        valid_motion = (batch["future_pose_mask"] & usable[..., None]
                        & batch["pose_mask"].bool())
        displacement = batch["future_position"] - batch["pose"].unsqueeze(-2)
        motion = training.masked_smooth_l1(output["future_displacement"], displacement,
                                         valid_motion)
        motion = motion + .25 * training.masked_smooth_l1(
            output["future_direct_position"], output["future_integrated_position"], valid_motion)
        total = (args.emg_holding_weight * dense + args.masked_weight * masked
                 + args.stability_weight * stability + args.motion_weight * motion)
        return total, {"emg_holding": dense.item(), "masked_alignment": masked.item(),
                       "holding_stability": stability.item(), "motion_consistency": motion.item()}

    @staticmethod
    def evaluate(predictions, args):
        # Fixed 0.5 holding threshold. No test-set threshold search.
        flips, seconds, correct, total = 0, 0., 0, 0
        for item in predictions:
            trial = item["trial"]
            valid = item["valid"] & trial["holding_certain"].astype(bool)
            pred = item["prob"][:, 0] >= .5
            truth = trial["holding"].astype(bool)
            dt = np.diff(trial["time"])
            pair = valid[1:] & valid[:-1] & (truth[1:] == truth[:-1]) & (dt > 0) & (dt < .03)
            flips += int(((pred[1:] != pred[:-1]) & pair).sum())
            seconds += float(dt[pair].sum())
            correct += int(((pred == truth) & valid).sum())
            total += int(valid.sum())
        return {"split": "test", "holding_threshold": .5,
                "boundary_excluded_holding_accuracy": correct / max(1, total),
                "maintenance_flips": flips, "maintenance_seconds": seconds,
                "maintenance_flips_per_minute": 60 * flips / seconds if seconds else None,
                "note": "Descriptive stability metric; not ReactEMG's exact transition accuracy."}


if __name__ == "__main__":
    training.main(ReactFutureIntentModel, "reach_grasp_react_intent_v1", ReactTraining)
