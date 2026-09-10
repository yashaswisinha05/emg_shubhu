#!/usr/bin/env python3
"""Render the paper pipeline for the frozen-classifier attention model."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

COLORS = {
    "emg": "#D95F59", "imu": "#3478B8", "frozen": "#667085",
    "state": "#8B5CF6", "fusion": "#0F9D82", "head": "#E6952A",
    "loss": "#B42318", "light": "#F7F9FC", "ink": "#17202A",
}


def box(ax, xy, wh, text, color, *, fill=0.10, dashed=False, fontsize=9,
        linewidth=1.6, radius=.018):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth, edgecolor=color,
        facecolor=(*plt.matplotlib.colors.to_rgb(color), fill),
        linestyle="--" if dashed else "-", zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=COLORS["ink"], linespacing=1.25,
            fontweight="semibold", zorder=3)
    return patch


def arrow(ax, start, end, color="#475467", *, dashed=False, width=1.5,
          connection="arc3", label=None, label_offset=(0, 0)):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=width,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=connection, shrinkA=2, shrinkB=2, zorder=1)
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha="center", va="center", fontsize=7.5, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.8, 8.6), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(.03, .965, "Causal neuromuscular state-conditioned attention",
            fontsize=18, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(.03, .925,
            "Frozen state reliability + trainable EMG–IMU motion fusion + 200 ms intent supervision",
            fontsize=10.5, color="#475467", va="top")

    # Input and preprocessing.
    box(ax, (.025, .70), (.105, .105), "4-site EMG\nS0, S4, S8, S12",
        COLORS["emg"], fontsize=9.5)
    box(ax, (.025, .51), (.105, .105), "4-site IMU\n24 channels",
        COLORS["imu"], fontsize=9.5)
    box(ax, (.155, .60), (.12, .125),
        "Causal preprocessing\nmedian / RMS / validity\n100 Hz resampling",
        "#475467", fontsize=8.5)
    arrow(ax, (.13, .752), (.155, .68), COLORS["emg"])
    arrow(ax, (.13, .562), (.155, .64), COLORS["imu"])

    # Frozen branch.
    frozen_region = FancyBboxPatch(
        (.30, .725), .425, .17, boxstyle="round,pad=0.012,rounding_size=.02",
        facecolor="#F2F4F7", edgecolor=COLORS["frozen"], linewidth=1.5,
        linestyle="--", zorder=0)
    ax.add_patch(frozen_region)
    ax.text(.31, .875, "FROZEN DEPLOYED STATE PATH", fontsize=8,
            color=COLORS["frozen"], fontweight="bold")
    box(ax, (.32, .765), (.12, .08), "Classifier-specific\nnormalization", COLORS["frozen"],
        fill=.07, fontsize=8)
    box(ax, (.47, .755), (.15, .10), "Proven neuromuscular\nclassifier (EMG+IMU)",
        COLORS["frozen"], fill=.07, fontsize=8.5)
    box(ax, (.65, .765), (.055, .08), "Softmax\n$p_t$", COLORS["state"], fontsize=9)
    arrow(ax, (.275, .68), (.32, .805), COLORS["frozen"], connection="arc3,rad=-.12")
    arrow(ax, (.44, .805), (.47, .805), COLORS["frozen"])
    arrow(ax, (.62, .805), (.65, .805), COLORS["state"])
    box(ax, (.77, .765), (.12, .08), "OPEN / CLOSE\nclassification", COLORS["state"], fontsize=9)
    arrow(ax, (.705, .805), (.77, .805), COLORS["state"])

    # Trainable encoders.
    ax.text(.31, .675, "TRAINABLE CAUSAL MOTION PATH", fontsize=8,
            color=COLORS["fusion"], fontweight="bold")
    box(ax, (.31, .535), (.14, .105), "EMG channel–time\nattention + causal\npatch encoder",
        COLORS["emg"], fontsize=8.5)
    box(ax, (.31, .36), (.14, .105), "IMU causal\npatch Transformer\n(motion backbone)",
        COLORS["imu"], fontsize=8.5)
    arrow(ax, (.275, .66), (.31, .59), COLORS["emg"])
    arrow(ax, (.275, .625), (.31, .412), COLORS["imu"], connection="arc3,rad=.12")

    box(ax, (.49, .59), (.12, .075), "State embedding\n$e_t=f(p_t)$",
        COLORS["state"], fontsize=8.5)
    arrow(ax, (.678, .765), (.55, .665), COLORS["state"], connection="arc3,rad=.13",
          label="detached", label_offset=(.015, .015))
    box(ax, (.50, .425), (.145, .105), "Causal EMG$\\rightarrow$IMU\ncross-attention\n$Q=h_t^{I}+e_t$, $K,V=h_{\\leq t}^{E}$",
        COLORS["fusion"], fontsize=8.5)
    arrow(ax, (.45, .587), (.50, .49), COLORS["emg"])
    arrow(ax, (.45, .412), (.50, .46), COLORS["imu"])
    arrow(ax, (.55, .59), (.565, .53), COLORS["state"])

    box(ax, (.68, .425), (.13, .105), "Zero-initialized\ngated EMG residual\nover IMU backbone",
        COLORS["fusion"], fontsize=8.5)
    arrow(ax, (.645, .477), (.68, .477), COLORS["fusion"])
    arrow(ax, (.45, .40), (.68, .445), COLORS["imu"], connection="arc3,rad=-.10",
          label="base", label_offset=(0, -.022))
    box(ax, (.84, .425), (.13, .105), "Soft-state FiLM\n$\\gamma(e_t),\\,\\beta(e_t)$\n+ LayerNorm",
        COLORS["state"], fontsize=8.5)
    arrow(ax, (.81, .477), (.84, .477), COLORS["fusion"])
    arrow(ax, (.61, .625), (.89, .53), COLORS["state"], connection="arc3,rad=-.10")

    # Heads.
    box(ax, (.38, .19), (.13, .095), "Current XYZ head\n$\\hat{\\mathbf{x}}_t$",
        COLORS["head"], fontsize=9)
    box(ax, (.55, .19), (.15, .095),
        "Pixel head (1920$\\times$1080)\ndirect + detached-XYZ blend",
        COLORS["head"], fontsize=8.4)
    box(ax, (.74, .19), (.16, .095),
        "Future XYZ decoder\n$\\hat{\\mathbf{x}}_{t+10:t+200\\,ms}$",
        COLORS["head"], fontsize=8.7)
    arrow(ax, (.905, .425), (.445, .285), COLORS["head"], connection="arc3,rad=-.08")
    arrow(ax, (.91, .425), (.625, .285), COLORS["head"], connection="arc3,rad=-.03")
    arrow(ax, (.92, .425), (.82, .285), COLORS["head"], connection="arc3,rad=.05")

    # Training-only objectives.
    ax.plot([.03, .97], [.135, .135], color="#D0D5DD", linewidth=1)
    ax.text(.03, .11, "TRAINING-ONLY SUPERVISION", fontsize=8,
            color=COLORS["loss"], fontweight="bold")
    box(ax, (.18, .045), (.16, .07), "Auxiliary state +\nteacher distillation",
        COLORS["loss"], fill=.04, dashed=True, fontsize=7.8, linewidth=1.2)
    box(ax, (.39, .045), (.12, .07), "Current XYZ\nSmooth-$L_1$",
        COLORS["loss"], fill=.04, dashed=True, fontsize=7.8, linewidth=1.2)
    box(ax, (.56, .045), (.14, .07), "Axis-aware pixel loss\n$x/1920$, $y/1080$",
        COLORS["loss"], fill=.04, dashed=True, fontsize=7.8, linewidth=1.2)
    box(ax, (.75, .045), (.18, .07), "Withheld-future reconstruction\n+ arrival consistency",
        COLORS["loss"], fill=.04, dashed=True, fontsize=7.8, linewidth=1.2)
    arrow(ax, (.38, .535), (.26, .115), COLORS["loss"], dashed=True,
          connection="arc3,rad=.16")
    arrow(ax, (.445, .19), (.45, .115), COLORS["loss"], dashed=True)
    arrow(ax, (.625, .19), (.63, .115), COLORS["loss"], dashed=True)
    arrow(ax, (.82, .19), (.84, .115), COLORS["loss"], dashed=True)

    ax.text(.965, .015, "Solid: inference path   ·   Dashed red: training only   ·   Gray: frozen",
            fontsize=7.5, color="#667085", ha="right")

    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"neuro_classifier_state_attention_pipeline.{suffix}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
