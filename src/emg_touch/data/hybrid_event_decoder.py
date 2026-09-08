"""Validation-calibrated causal fusion of local heads and holding transitions."""
from __future__ import annotations

import numpy as np

from .annotation_uncertainty import holding_transitions


def transition_pulses(item, holding_parameters, pulse_s):
    """Convert confirmed holding transitions into causal, forward-only pulses."""
    time, valid = item["trial"]["time"], item["valid"]
    found = holding_transitions(time, item["prob"][:, 0], valid,
                                **holding_parameters)
    pulses = np.zeros((len(time), 2), dtype=float)
    for event, stamps in enumerate(found):
        for stamp in stamps:
            pulses[:, event] = np.maximum(
                pulses[:, event], ((time >= stamp) & (time < stamp + pulse_s)))
    pulses[~valid] = np.nan
    return pulses


def stable_detect(time, probability, valid, threshold, persistence_s,
                  low_ratio=.5, refractory_s=.4):
    """Causal hysteresis/debounce; a trigger is timestamped when confirmed."""
    armed, since, last, previous = True, None, -np.inf, None
    found = []
    for stamp, value, good in zip(time, probability, valid):
        if previous is not None and stamp - previous > .05:
            since = None
        previous = stamp
        if not good or not np.isfinite(value):
            since = None
            continue
        if value < threshold * low_ratio:
            armed, since = True, None
        if value < threshold or not armed or stamp - last < refractory_s:
            if value < threshold:
                since = None
            continue
        if since is None:
            since = stamp
        if stamp - since + 1e-9 >= persistence_s:
            found.append(float(stamp))
            armed, since, last = False, None, stamp
    return found


def apply_decoder(items, decoder):
    """Apply parameters frozen on validation to prediction dictionaries."""
    result = []
    for item in items:
        pulses = transition_pulses(item, decoder["holding"], decoder["pulse_s"])
        combined = item["prob"].copy()
        detections = []
        for event, parameters in enumerate(decoder["events"]):
            alpha = parameters["local_weight"]
            combined[:, event + 1] = (alpha * item["prob"][:, event + 1]
                                      + (1 - alpha) * pulses[:, event])
            detections.append(stable_detect(
                item["trial"]["time"], combined[:, event + 1], item["valid"],
                parameters["threshold"], parameters["persistence_s"]))
        result.append(dict(item, prob=combined, detections=detections))
    return result


def calibrate_decoder(items, holding_parameters, event_score):
    """Choose each event's causal blend solely by event-level validation F1."""
    best = None
    for pulse_s in [.05, .1, .2]:
        pulses = [transition_pulses(item, holding_parameters, pulse_s) for item in items]
        selected = []
        for event in range(2):
            winner = None
            for alpha in [0., .25, .5, .75, 1.]:
                for threshold in [.2, .35, .5, .65, .8]:
                    for persistence in [0., .03, .06]:
                        candidate = []
                        for item, pulse in zip(items, pulses):
                            probability = (alpha * item["prob"][:, event + 1]
                                           + (1 - alpha) * pulse[:, event])
                            candidate.append(dict(item, detections={event:
                                stable_detect(item["trial"]["time"], probability,
                                              item["valid"], threshold, persistence)}))
                        score = event_score(candidate, event)
                        parameters = {"local_weight": alpha, "threshold": threshold,
                                      "persistence_s": persistence}
                        if winner is None or score > winner[0]:
                            winner = (score, parameters)
            selected.append(winner)
        score = sum(x[0] for x in selected) / 2
        if best is None or score > best[0]:
            best = (score, {"holding": holding_parameters, "pulse_s": pulse_s,
                            "events": [x[1] for x in selected],
                            "validation_event_macro_f1": score})
    return best[1]
