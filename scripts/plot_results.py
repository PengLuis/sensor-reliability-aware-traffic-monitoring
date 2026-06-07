"""Plot tables and figures from traceable result artifacts."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="output")
    parser.add_argument("--figures-dir", default="paper/figures")
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    print("plot_results skeleton only; no figures generated.")


if __name__ == "__main__":
    main()
