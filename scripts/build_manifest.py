#!/usr/bin/env python3
from __future__ import annotations

import argparse

from emg_touch.config import load_config
from emg_touch.data.manifest import build_manifest_rows, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a trial-level manifest from session summaries")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = build_manifest_rows(
        data_root=config["paths"]["data_root"],
        cache_dir=config["paths"]["cache_dir"],
        subject_aliases=config["data"].get("subject_aliases", {}),
    )
    frame = write_manifest(rows, config["paths"]["manifest"])
    print(f"Wrote {len(frame)} trials to {config['paths']['manifest']}")
    if "touch_time_s" in frame:
        available = int(frame["touch_time_s"].notna().sum())
        print(f"Touch-aligned timestamps: {available}/{len(frame)}")
    print(frame.groupby(["configuration", "subject"]).size().to_string())


if __name__ == "__main__":
    main()
