from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = _load_mapping(path, seen=set())
    config = deepcopy(config)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent)
    _resolve_paths(config)
    return config


def _load_mapping(path: Path, seen: set[Path]) -> dict[str, Any]:
    """Load a YAML mapping, optionally deep-merging a relative base_config."""
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"Circular base_config chain: {chain}")
    seen = {*seen, path}
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    base_value = config.pop("base_config", None)
    if base_value is None:
        return config
    base_path = Path(base_value).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    inherited = _load_mapping(base_path.resolve(), seen)
    return _deep_merge(inherited, config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_paths(config: dict[str, Any]) -> None:
    project_root = Path(config["_project_root"])
    for key, value in config.get("paths", {}).items():
        if value is None:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        config["paths"][key] = str(candidate.resolve())


def save_config(config: dict[str, Any], path: str | Path) -> None:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False)


def override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[parts[-1]] = value
