"""Discover and audit raw PEMS-BAY files without generating data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pems_bay_utils import (  # noqa: E402
    ADJ_NAMES,
    SPEED_NAMES,
    candidate_dirs,
    discover_file,
    frame_stats,
    load_adjacency,
    load_speed_frame,
    write_blocker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", default="data")
    parser.add_argument("--output-dir", default="experiments/pems-bay-data-import")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = ROOT / "data" / "processed" / "pems-bay"

    speed_path = discover_file(args.search_root, SPEED_NAMES, "speed")
    adj_path = discover_file(args.search_root, ADJ_NAMES, "adj")
    report: dict[str, object] = {
        "searched_directories": [str(p) for p in candidate_dirs(args.search_root)],
        "speed_file": None,
        "adjacency_file": None,
        "status": "PASS",
    }

    if speed_path is None:
        details = {"searched_directories": report["searched_directories"], "accepted_names": SPEED_NAMES}
        report["status"] = "BLOCKED"
        report["blocker"] = "raw_speed_file_missing"
        (out_dir / "raw_data_discovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "raw_data_discovery_summary.md").write_text(
            "# Raw Data Discovery Summary\n\n- Status: `BLOCKED`\n- Raw PEMS-BAY speed file was not found.\n",
            encoding="utf-8",
        )
        write_blocker(processed_dir, out_dir, "raw PEMS-BAY speed file missing", details)
        print(json.dumps(report, indent=2))
        raise SystemExit(2)

    frame, meta = load_speed_frame(speed_path)
    stats = frame_stats(frame)
    report["speed_file"] = {
        "path": str(speed_path),
        "size_bytes": speed_path.stat().st_size,
        "extension": speed_path.suffix.lower(),
        "timestamp_status": meta["timestamp_status"],
        "time_steps": stats["time_steps"],
        "sensor_count": stats["sensor_count"],
        "stats": stats,
    }
    if adj_path is None:
        report["status"] = "BLOCKED"
        report["blocker"] = "adjacency_file_missing"
        details = {"speed_file": str(speed_path), "accepted_adjacency_names": ADJ_NAMES}
        write_blocker(processed_dir, out_dir, "raw PEMS-BAY adjacency file missing", details)
    else:
        adj, adj_meta = load_adjacency(adj_path)
        report["adjacency_file"] = {
            "path": str(adj_path),
            "size_bytes": adj_path.stat().st_size,
            "extension": adj_path.suffix.lower(),
            "metadata": adj_meta,
            "shape_matches_speed_n": bool(adj.shape[0] == stats["sensor_count"]),
        }
        if adj.shape[0] != stats["sensor_count"]:
            report["status"] = "BLOCKED"
            report["blocker"] = "adjacency_shape_mismatch"
            write_blocker(
                processed_dir,
                out_dir,
                "raw PEMS-BAY adjacency shape mismatches speed sensor count",
                {"speed_sensor_count": stats["sensor_count"], "adjacency_shape": list(adj.shape)},
            )

    (out_dir / "raw_data_discovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = [
        "# Raw Data Discovery Summary",
        "",
        f"- Status: `{report['status']}`",
        f"- Speed file: `{speed_path}`",
        f"- Time steps: `{stats['time_steps']}`",
        f"- Sensors: `{stats['sensor_count']}`",
        f"- Timestamp status: `{meta['timestamp_status']}`",
        f"- NaN ratio: `{stats['nan_ratio']}`",
        f"- Zero ratio: `{stats['zero_ratio']}`",
        f"- Adjacency file: `{adj_path if adj_path else 'MISSING'}`",
    ]
    (out_dir / "raw_data_discovery_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if report["status"] == "PASS":
        for stale in [processed_dir / "DATA_BLOCKER_REPORT.md", out_dir / "DATA_BLOCKER_REPORT.md"]:
            if stale.exists():
                stale.unlink()
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
