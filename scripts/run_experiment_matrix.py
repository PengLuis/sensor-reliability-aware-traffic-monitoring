"""Run the configured experiment matrix."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_matrix.yaml")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs only.")
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    print("run_experiment_matrix skeleton only; no experiment run executed.")


if __name__ == "__main__":
    main()
