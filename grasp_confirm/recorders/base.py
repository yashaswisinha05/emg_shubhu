"""Recorder interface.

To support a new EMG device you implement three methods. Nothing else in this
package knows anything about hardware.

    class MyRecorder(Recorder):
        def start(self): ...
        def poll(self): return t, x     # t: (n,) seconds, x: (n, C)
        def stop(self): ...

``poll()`` returns every sample acquired since the previous call and may return
an empty batch. Timestamps must be on the same clock as ``time.perf_counter()``
in the recording process, because that is the clock the cue events are stamped
with and the two are aligned by subtraction, not by wall time.

If your device gives you a sample counter rather than host timestamps, derive
    t = t0_perf_counter + index / sample_rate
and say so in ``clock_note`` -- that estimate drifts, and the segmenter prints
``clock_note`` so whoever reads the data later knows what they are holding.
"""

from __future__ import annotations

import abc

import numpy as np


class Recorder(abc.ABC):
    """Streaming multi-channel EMG source."""

    #: Human-readable channel names, length C, in column order.
    channel_names: list[str] = []
    #: Nominal sampling rate in Hz.
    sample_rate: float = 0.0
    #: True when samples are not from real hardware. Propagated into session.json
    #: and checked by qc.py, which refuses to issue a verdict on fake data.
    synthetic: bool = False
    #: How ``poll()`` timestamps were obtained; stored with the session.
    clock_note: str = "host perf_counter at poll time"

    @abc.abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""

    @abc.abstractmethod
    def poll(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t, x)`` for all samples acquired since the last call."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop streaming and release the device."""

    def __enter__(self) -> "Recorder":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
