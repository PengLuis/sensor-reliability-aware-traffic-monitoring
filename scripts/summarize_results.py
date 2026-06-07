"""Summarize experiment results from saved artifacts."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--output-dir", default="output")
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    print("summarize_results skeleton only; no summaries generated.")


if __name__ == "__main__":
    main()
