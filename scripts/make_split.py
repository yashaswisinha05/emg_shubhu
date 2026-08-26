#!/usr/bin/env python3
from __future__ import annotations

import argparse

from emg_touch.config import load_config
from emg_touch.data.manifest import load_manifest
from emg_touch.data.splits import make_subject_split
from emg_touch.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a participant-disjoint train/val/test split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test-subject", required=True)
    parser.add_argument("--val-subject", required=True)
    parser.add_argument("--configuration", help="Optional configuration such as a1 or mix3")
    parser.add_argument("--output", help="Override split JSON path")
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_manifest(config["paths"]["manifest"])
    split = make_subject_split(
        frame,
        test_subject=args.test_subject,
        val_subject=args.val_subject,
        configuration=args.configuration,
    )
    output = args.output or config["paths"]["split_file"]
    save_json(split, output)
    print(f"Wrote {output}: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")


if __name__ == "__main__":
    main()

