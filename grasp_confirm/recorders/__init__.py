"""Pluggable EMG sources. See base.py for the three-method interface."""

from __future__ import annotations

import importlib

from .base import Recorder


def make_recorder(spec: str, **kwargs) -> Recorder:
    """Build a recorder from a CLI spec.

    ``synthetic``            fake data, for dry runs (see synthetic.py)
    ``delsys``               Delsys Trigno via the AeroPy SDK (unverified)
    ``pkg.module:ClassName`` anything else implementing Recorder
    """
    if spec == "synthetic":
        from .synthetic import SyntheticRecorder

        return SyntheticRecorder(**kwargs)
    if spec == "delsys":
        from .delsys import DelsysRecorder

        return DelsysRecorder(**kwargs)
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)(**kwargs)
    raise SystemExit(
        f"unknown recorder {spec!r}; use 'synthetic', 'delsys', or 'module:Class'"
    )


__all__ = ["Recorder", "make_recorder"]
