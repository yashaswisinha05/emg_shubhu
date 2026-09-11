#!/usr/bin/env python3
"""Inference entry point for the shared-encoder residual-GRU model.

Accepts either --trial-csv for an unseen recorded trial or newline-delimited
raw samples on stdin. The argument interface and JSON output are identical to
infer_neuromuscular_residual_gru.py.
"""
from __future__ import annotations

from infer_neuromuscular_residual_gru import main


if __name__ == "__main__":
    main()
