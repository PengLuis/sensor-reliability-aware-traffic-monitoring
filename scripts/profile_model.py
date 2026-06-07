"""Profile model parameter count and latency."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="SRAF-GRU")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    print("profile_model skeleton only; no profiling executed.")


if __name__ == "__main__":
    main()
