"""Run fault simulation tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corruptions.faults import (  # noqa: E402
    continuous_outage,
    gaussian_noise,
    linear_drift,
    random_missing,
    stuck_at_last_value,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke checks only.")
    return parser


def _assert_fault_result(original: np.ndarray, corrupted: np.ndarray, mask: np.ndarray, metadata: dict) -> None:
    if corrupted.shape != original.shape:
        raise AssertionError(f"Shape changed from {original.shape} to {corrupted.shape}")
    if mask.shape != original.shape:
        raise AssertionError(f"Mask shape changed from {original.shape} to {mask.shape}")
    if mask.dtype != np.bool_:
        raise AssertionError("Mask must be boolean")
    if not isinstance(metadata, dict) or "fault" not in metadata:
        raise AssertionError("Metadata must include fault name")


def run_tests(seed: int) -> None:
    x3 = np.arange(12 * 5 * 1, dtype=float).reshape(12, 5, 1)
    x4 = np.stack([x3, x3 + 100.0], axis=0)
    cases = [
        lambda arr: random_missing(arr, rate=0.2, seed=seed),
        lambda arr: continuous_outage(arr, length=6, seed=seed),
        lambda arr: gaussian_noise(arr, severity="medium", train_std=2.0, seed=seed),
        lambda arr: linear_drift(arr, severity="medium", train_std=2.0, seed=seed),
        lambda arr: stuck_at_last_value(arr, severity="medium", seed=seed),
    ]
    for arr in (x3, x4):
        original = arr.copy()
        for case in cases:
            corrupted, mask, metadata = case(arr)
            _assert_fault_result(arr, corrupted, mask, metadata)
            if not np.array_equal(arr, original):
                raise AssertionError(f"Input mutated by {metadata['fault']}")
            if not mask.any():
                raise AssertionError(f"Fault mask is empty for {metadata['fault']}")
    print("fault tests passed")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_tests(args.seed)


if __name__ == "__main__":
    main()
