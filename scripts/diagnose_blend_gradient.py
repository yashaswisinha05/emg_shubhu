#!/usr/bin/env python3
"""Diagnose why a grid_fusion_physics(3) blend weight isn't moving.

physics_blend has been observed frozen at its ~0.018 sigmoid(-4) init across
every run so far, even at 10x the physics loss weight. That's consistent with
two different explanations that call for different fixes:

  (a) No consistent incentive: physics_pred is sometimes closer to target and
      sometimes farther, so the gradient into raw_blend averages toward zero
      across trials/batches - the branch genuinely isn't accurate enough yet,
      full stop.
  (b) A real but numerically suppressed incentive: at blend~0.018, the local
      sigmoid slope blend*(1-blend) is ~14x smaller than at blend=0.5, and
      global gradient-norm clipping (training.gradient_clip_norm) could squash
      an already-tiny raw_blend gradient toward the clipping threshold along
      with everything else.

This reads raw_blend.grad directly off real batches - before any clipping -
and decomposes it per-trial to tell the two apart, instead of guessing from
the training curve.
"""
from __future__ import annotations

import argparse

import torch

from emg_touch.config import load_config
from emg_touch.data.grid_trajectory import build_grid_trajectory_loaders
from emg_touch.grid_training import grid_point_loss
from emg_touch.models.grid_point import build_grid_model
from emg_touch.utils import choose_device, move_batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hill_fusion.yaml")
    parser.add_argument("--kind", choices=["grid_fusion_physics", "grid_fusion_physics3"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(args.device)
    train_loader, _, _ = build_grid_trajectory_loaders(
        config, config["paths"]["split_file"], config["paths"]["scaler"]
    )

    model = build_grid_model(args.kind, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.train()

    clip_norm = float(config["training"]["gradient_clip_norm"])
    print(f"kind={args.kind}  checkpoint epoch={checkpoint.get('epoch', '?')}")
    print(f"blend at load: {torch.sigmoid(model.raw_blend).item():.5f}")
    print(f"config gradient_clip_norm: {clip_norm}")
    print()

    it = iter(train_loader)
    raw_grads = []
    same_sign_fractions = []
    global_norms = []
    for step in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = move_batch_to_device(batch, device)

        outputs = model(batch)
        prediction = outputs["prediction"]
        prediction.retain_grad()
        losses = grid_point_loss(outputs, batch, config)
        loss = losses["loss"]

        model.zero_grad(set_to_none=True)
        loss.backward()

        raw_grad = model.raw_blend.grad.item()
        raw_grads.append(raw_grad)

        # Global gradient norm training would compute, to see how much
        # clipping would rescale this batch's raw_blend gradient.
        total_sq = sum(
            p.grad.detach().pow(2).sum().item()
            for p in model.parameters()
            if p.grad is not None
        )
        global_norm = total_sq**0.5
        global_norms.append(global_norm)
        clip_scale = min(1.0, clip_norm / max(global_norm, 1e-12))

        # Per-trial sign breakdown: does each trial's own contribution agree
        # with the batch's net gradient, or are trials pulling opposite ways?
        grad_pred = prediction.grad.detach()  # (batch, 2), d(loss)/d(blended prediction)
        blend = torch.sigmoid(model.raw_blend).detach()
        physics_pred = outputs["physics_prediction"].detach().clamp(0.0, 1.0)
        fusion_pred = outputs["fusion_prediction"].detach()
        delta = physics_pred - fusion_pred  # (batch, 2)
        per_trial = (grad_pred * delta).sum(dim=-1) * (blend * (1.0 - blend))
        net_sign = 1.0 if raw_grad >= 0 else -1.0
        same_sign = ((per_trial * net_sign) > 0).float().mean().item()
        same_sign_fractions.append(same_sign)

        print(
            f"batch {step}: raw_blend.grad={raw_grad:+.3e}  "
            f"global_grad_norm={global_norm:.2f}  clip_scale={clip_scale:.3f}  "
            f"trials_agreeing_with_net_sign={same_sign*100:.0f}%"
        )

    mean_grad = sum(raw_grads) / len(raw_grads)
    grad_sign_flips = sum(
        1 for a, b in zip(raw_grads, raw_grads[1:]) if (a >= 0) != (b >= 0)
    )
    print()
    print(f"mean raw_blend.grad across {args.batches} batches: {mean_grad:+.3e}")
    print(f"sign flips between consecutive batches: {grad_sign_flips}/{args.batches - 1}")
    print(f"mean per-trial agreement with each batch's own net sign: "
          f"{sum(same_sign_fractions) / len(same_sign_fractions) * 100:.0f}%")
    print(f"mean global gradient norm: {sum(global_norms) / len(global_norms):.2f} "
          f"(clip threshold: {clip_norm})")
    print()
    print("Reading this:")
    print("- If raw_blend.grad flips sign often across batches and per-trial")
    print("  agreement is near 50%: explanation (a) - no consistent incentive,")
    print("  physics isn't accurate enough yet, full stop.")
    print("- If raw_blend.grad is consistently signed (few flips, high per-trial")
    print("  agreement) but tiny in magnitude, and global_grad_norm is well above")
    print("  the clip threshold: explanation (b) - a real incentive is being")
    print("  numerically suppressed, both by the sigmoid tail and by clipping.")


if __name__ == "__main__":
    main()
