"""Audit official-style STID identity time features for METR-LA splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_strong_clean_backbone_integration import load_split, stid_time_feature_audit  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-official-style-stid-code-repair")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x, _ = load_split(data_dir, "train")
    val_x, _ = load_split(data_dir, "val")
    test_x, _ = load_split(data_dir, "test")
    audit = stid_time_feature_audit(
        train_x,
        val_x,
        test_x,
        train_start=0,
        val_start=train_x.shape[0],
        test_start=train_x.shape[0] + val_x.shape[0],
    )
    valid = all(
        split["tod_min"] == 0
        and split["tod_max"] == 287
        and split["tod_unique_count"] == 288
        and split["dow_min"] == 0
        and split["dow_max"] == 6
        and split["dow_unique_count"] == 7
        for split in audit["splits"]
    )
    audit["status"] = "PASS" if valid else "FAIL"
    (out_dir / "time_feature_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "output": str(out_dir / "time_feature_audit.json")}, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
