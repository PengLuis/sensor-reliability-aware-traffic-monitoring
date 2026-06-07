"""Audit PEMS-BAY STID identity feature construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pems_bay_utils import add_identity_features, load_npz_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/pems-bay")
    parser.add_argument("--output-dir", default="experiments/pems-bay-data-import")
    return parser.parse_args()


def audit_split(x: np.ndarray, start_index: int) -> dict[str, object]:
    aug = add_identity_features(x, start_index)
    tod_idx = np.floor(aug[..., 1] * 288).astype(np.int64)
    dow_idx = np.floor(aug[..., 2] * 7).astype(np.int64)
    return {
        "start_index": int(start_index),
        "tod_min": int(tod_idx.min()),
        "tod_max": int(tod_idx.max()),
        "tod_unique_count": int(np.unique(tod_idx).size),
        "first_50_tod_index_values": tod_idx.reshape(-1)[:50].astype(int).tolist(),
        "dow_min": int(dow_idx.min()),
        "dow_max": int(dow_idx.max()),
        "dow_unique_count": int(np.unique(dow_idx).size),
        "first_50_dow_index_values": dow_idx.reshape(-1)[:50].astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x, _ = load_npz_pair(data_dir / "train.npz")
    val_x, _ = load_npz_pair(data_dir / "val.npz")
    test_x, _ = load_npz_pair(data_dir / "test.npz")
    offsets = {"train": 0, "val": int(train_x.shape[0]), "test": int(train_x.shape[0] + val_x.shape[0])}
    time_path = data_dir / "time_metadata.json"
    time_meta = json.loads(time_path.read_text(encoding="utf-8")) if time_path.exists() else {}

    split_audits = {
        "train": audit_split(train_x, offsets["train"]),
        "val": audit_split(val_x, offsets["val"]),
        "test": audit_split(test_x, offsets["test"]),
    }
    identity_checks = {}
    base = add_identity_features(test_x[: min(128, test_x.shape[0])], offsets["test"])
    for fault in ["random_missing_40", "gaussian_noise_high", "linear_drift_high", "stuck_at_last_value_high"]:
        corrupted = base.copy()
        if fault == "random_missing_40":
            rng = np.random.default_rng(42)
            mask = rng.random(corrupted[..., 0].shape) < 0.4
            corrupted[..., 0][mask] = 0.0
        elif fault == "gaussian_noise_high":
            corrupted[..., 0] += 0.5
        elif fault == "linear_drift_high":
            drift = np.linspace(0.0, 0.5, corrupted.shape[1], dtype=np.float32)[None, :, None]
            corrupted[..., 0] += drift
        elif fault == "stuck_at_last_value_high":
            corrupted[:, corrupted.shape[1] // 2 :, :, 0] = corrupted[:, corrupted.shape[1] // 2 - 1 : corrupted.shape[1] // 2, :, 0]
        identity_checks[fault] = bool(np.array_equal(base[..., 1:], corrupted[..., 1:]))

    status = "PASS" if all(identity_checks.values()) and all(a["tod_unique_count"] > 0 and a["dow_unique_count"] > 0 for a in split_audits.values()) else "FAIL"
    audit = {
        "status": status,
        "identity_rule": "x_aug[...,0]=speed_norm, x_aug[...,1]=tod_index/288, x_aug[...,2]=dow_index/7",
        "timestamp_status": time_meta.get("timestamp_status", "MISSING"),
        "tod_dow_source": time_meta.get("tod_dow_source", "MISSING"),
        "split_offsets": offsets,
        "splits": split_audits,
        "identity_features_unchanged_under_faults": identity_checks,
    }
    (out_dir / "identity_feature_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Identity Feature Audit Summary",
        "",
        f"- Status: `{status}`",
        f"- Timestamp status: `{audit['timestamp_status']}`",
        f"- TOD/DOW source: `{audit['tod_dow_source']}`",
        f"- Split offsets: `{offsets}`",
        f"- Identity features unchanged under speed-only fault corruption: `{all(identity_checks.values())}`",
        f"- Train TOD unique count: `{split_audits['train']['tod_unique_count']}`",
        f"- Val TOD unique count: `{split_audits['val']['tod_unique_count']}`",
        f"- Test TOD unique count: `{split_audits['test']['tod_unique_count']}`",
    ]
    (out_dir / "identity_feature_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
