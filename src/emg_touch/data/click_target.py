"""Attach a trial's on-screen click target for pixel-space supervision.

The target is a property of the whole trial, not of a timestep: the
participant reaches to one button and clicks it once, and the click columns
carry that single coordinate on every row. It is stored here as one (x, y)
pair in normalized canvas units, and the trainer broadcasts it across the
trial's timesteps rather than the loader materializing a constant array.

Normalization matches the recorder's own convention -- click_x_norm is
click_x_pos / canvas_width_px and click_y_norm is click_y_pos /
canvas_height_px, note the CANVAS height, not the screen height. The canvas
size is kept alongside so error can be reported back in pixels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reach_grasp import constant

CANVAS_COLUMNS = ("canvas_width_px", "canvas_height_px")
NORMALIZED_COLUMNS = ("click_x_norm", "click_y_norm")
PIXEL_COLUMNS = ("click_x_pos", "click_y_pos")


def add_click_target(path, trial):
    frame = pd.read_csv(path, usecols=lambda name: name in
                        set(CANVAS_COLUMNS + NORMALIZED_COLUMNS + PIXEL_COLUMNS))
    canvas = [constant(frame, name) for name in CANVAS_COLUMNS]
    if any(value is None or value <= 0 for value in canvas):
        raise ValueError("missing or nonpositive canvas_width_px/canvas_height_px")
    normalized = [constant(frame, name) for name in NORMALIZED_COLUMNS]
    if any(value is None for value in normalized):
        pixels = [constant(frame, name) for name in PIXEL_COLUMNS]
        if any(value is None for value in pixels):
            raise ValueError("missing click target: need click_x_norm/click_y_norm or "
                             "click_x_pos/click_y_pos alongside the canvas size")
        normalized = [pixel / size for pixel, size in zip(pixels, canvas)]
    target = np.asarray(normalized, dtype="float32")
    if not np.isfinite(target).all():
        raise ValueError("nonfinite click target")
    trial["click_target"] = target
    trial["canvas_px"] = np.asarray(canvas, dtype="float32")
    trial["audit"]["click_target_norm"] = target.tolist()
    trial["audit"]["click_target_px"] = (target * trial["canvas_px"]).tolist()
    return trial
