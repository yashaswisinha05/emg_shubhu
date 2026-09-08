"""Supervision and causal decoding for manually annotated interaction events."""
import numpy as np


def soften_events(trial, uncertainty_s):
    if uncertainty_s <= 0:
        raise ValueError("annotation uncertainty must be positive")
    distance = np.abs(trial["time"][:, None] - trial["events"])
    trial["event_labels"] = np.where(distance <= uncertainty_s,
        np.exp(-.5 * (distance / (uncertainty_s / 2))**2), 0).astype("float32")
    trial["holding_certain"] = (distance > uncertainty_s).all(1)
    trial["annotation_uncertainty_s"] = float(uncertainty_s)
    return trial


def holding_transitions(times, probability, valid, low=.35, high=.65, persistence_s=.05):
    """Hysteresis with timestamp-based persistence; emits at confirmation time.

    State starts UNKNOWN. Establishing the initial state emits no event, so a
    recording beginning mid-hold is not automatically treated as a new grasp.
    Missing inputs reset pending evidence, not the last confirmed state.
    """
    if not 0 <= low < high <= 1 or persistence_s < 0:
        raise ValueError("invalid hysteresis/persistence settings")
    state, candidate, since, previous = None, None, None, None
    events = [[], []]
    for t, p, good in zip(times, probability, valid):
        good = good and np.isfinite(p)
        if not good or (previous is not None and t - previous > .05):
            candidate, since = None, None
        previous = t
        if not good:
            continue
        wanted = True if p >= high else False if p <= low else None
        if wanted is None or wanted == state:
            candidate, since = None, None
            continue
        if wanted != candidate:
            candidate, since = wanted, t
        if t - since + 1e-9 >= persistence_s:
            if state is not None:
                events[0 if wanted else 1].append(float(t))
            state, candidate, since = wanted, None, None
    return events
