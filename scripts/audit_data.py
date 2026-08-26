#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emg_touch.config import load_config
from emg_touch.data.audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of the EMG/IMU dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    config = load_config(args.config)
    report = audit_dataset(config["paths"]["data_root"])
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
