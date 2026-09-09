"""Trial timing, segmentation policy and label windows -- single source of truth.

Every other module imports these constants. Change them here, re-run
segmentation; you never have to re-record.

Trial structure: start -> act immediately -> stop
--------------------------------------------------
The hand is PRE-POSITIONED, open, around the object without touching it (or
around the same empty space, for AIR). On the GO beep the subject closes
immediately and holds until STOP. There is no reach phase, so AIR and GRASP
differ only in whether an object is inside the closing hand -- reach kinematics
cannot leak into the label.

    0.0  cue text appears  ("BOTTLE - firm" / "AIR - hard" / "NOTHING")
         subject positions the open hand, NOT touching the object
    2.0  beep GO      -> close immediately and hold
    5.0  beep STOP    -> open, relax
    6.5  trial ends

The labelled window is NOT at a fixed offset. Reaction time varies, so the
window is anchored to the EMG onset detected inside the GO..STOP span:

    onset             first sustained crossing of the baseline threshold
    onset + SETTLE_S  window starts (skips the closing transient)
    + HOLD_S          window ends, clipped at STOP

NOTHING trials have no onset by construction: the whole GO..STOP span is
labelled REST, and a detected onset means the subject moved and the trial is
dropped. That is a real data-quality check, not a formality.
"""

from __future__ import annotations

LABELS = ("REST", "AIR", "GRASP")
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

# -- trial timing, seconds from trial onset ---------------------------------
CUE_S = 0.0          # cue text appears; subject positions the open hand
GO_S = 2.0           # GO beep -- close immediately
STOP_S = 5.0         # STOP beep -- open, relax
TRIAL_DURATION_S = 6.5

# -- segmentation policy ----------------------------------------------------
SETTLE_S = 0.30      # skipped after the detected onset (closing transient)
HOLD_S = 2.00        # length of the labelled hold window
MIN_HOLD_S = 1.00    # shorter than this and the trial is dropped
EDGE_S = 0.20        # trimmed from both ends of a NOTHING trial
REST_TAIL_S = 1.00   # REST harvested from the post-STOP relaxation

# Onset = envelope exceeds baseline_mean + ONSET_K * baseline_std, sustained
# for ONSET_PERSISTENCE_S. The baseline comes from the session's own rest
# recording, so this rescales automatically across sessions and electrode
# re-applications.
ONSET_K = 4.0
ONSET_PERSISTENCE_S = 0.10
ONSET_SEARCH_PAD_S = 0.20   # onset may precede GO slightly (anticipation)

BEEP_TONES = {"go": 880.0, "stop": 523.0}

CLASS_OF_FAMILY = {
    "AIR": "AIR",
    "GRASP": "GRASP",
    "NOTHING": "REST",
}
