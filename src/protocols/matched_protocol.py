"""Load and validate the manuscript-matched formal protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "matched_baseline_formal.yaml"


def load_matched_protocol(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Protocol must be a mapping: {config_path}")
    required = {"datasets", "seeds", "matched_train_faults", "test_faults", "validation_faults", "training"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Missing protocol keys {missing}: {config_path}")
    train_faults = list(data["matched_train_faults"])
    test_faults = list(data["test_faults"])
    if "random_missing_20" in train_faults:
        raise ValueError("RM20 is test-only and must not appear in matched_train_faults")
    if "random_missing_20" not in test_faults:
        raise ValueError("RM20 must be present in test_faults")
    if len(train_faults) != len(set(train_faults)):
        raise ValueError("matched_train_faults contains duplicates")
    return data


PROTOCOL = load_matched_protocol()
DATASETS = tuple(str(value) for value in PROTOCOL["datasets"])
SEEDS = tuple(int(value) for value in PROTOCOL["seeds"])
MATCHED_TRAIN_FAULTS = tuple(str(value) for value in PROTOCOL["matched_train_faults"])
TEST_FAULTS = tuple(str(value) for value in PROTOCOL["test_faults"])
VALIDATION_FAULTS = tuple(str(value) for value in PROTOCOL["validation_faults"])


def training_fault_for_step(step: int) -> str:
    """Return the deterministic fault label used at a global batch step."""

    return MATCHED_TRAIN_FAULTS[step % len(MATCHED_TRAIN_FAULTS)]


def training_fault_seed(seed: int, step: int) -> int:
    return int(seed) + int(step)


def test_fault_seed(seed: int, fault_index: int) -> int:
    return int(seed) + int(fault_index)

