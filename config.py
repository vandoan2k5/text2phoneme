from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_runtime_config(path: str) -> Dict[str, Any]:
    runtime_path = Path(path)
    runtime_config = load_yaml(str(runtime_path))
    model_config_path = runtime_config.get("model_config")
    if not model_config_path:
        return runtime_config

    resolved_model_config = Path(model_config_path)
    if not resolved_model_config.is_absolute():
        resolved_model_config = (runtime_path.parent / resolved_model_config).resolve()

    model_config = load_yaml(str(resolved_model_config))
    merged = deep_merge_dicts(model_config, runtime_config)
    merged["model_config"] = str(resolved_model_config)
    return merged
