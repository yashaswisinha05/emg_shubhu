#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from emg_touch.config import load_config
from emg_touch.data.schema import configuration_from_participant_id


def planned_moves(data_root: Path) -> list[tuple[Path, Path]]:
    moves: dict[Path, Path] = {}
    for summary_path in data_root.rglob("session_summary.json"):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        participant_dir = summary_path.parent.parent
        participant_id = participant_dir.name
        configuration = configuration_from_participant_id(participant_id)
        destination = data_root / configuration / participant_dir.name
        if participant_dir.resolve() != destination.resolve():
            moves[participant_dir.resolve()] = destination.resolve()
    return sorted(moves.items(), key=lambda item: str(item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move logically misplaced participant directories; dry-run unless --apply is given"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["paths"]["data_root"])
    moves = planned_moves(root)
    for source, destination in moves:
        print(f"{source} -> {destination}")
    if not args.apply:
        print(f"Dry run: {len(moves)} move(s). Pass --apply to perform them.")
        return
    for source, destination in moves:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    print(f"Applied {len(moves)} move(s). Rebuild the manifest and cache afterwards.")


if __name__ == "__main__":
    main()
