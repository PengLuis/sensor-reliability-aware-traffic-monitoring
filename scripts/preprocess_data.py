"""Preprocess traffic sensor datasets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.traffic_dataset import (  # noqa: E402
    ForecastingSpec,
    load_h5_series,
    load_adjacency_pickle,
    load_csv_series,
    make_windows,
    normalize_splits,
    save_processed_dataset,
    split_windows,
    synthetic_series,
    write_missing_data_status,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="METR-LA", help="Dataset name.")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory.")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed output directory.")
    parser.add_argument("--input-length", type=int, default=12, help="Input window L.")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon H.")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Create synthetic smoke-test data only.")
    parser.add_argument("--synthetic-steps", type=int, default=96, help="Synthetic time steps.")
    parser.add_argument("--synthetic-sensors", type=int, default=8, help="Synthetic sensor count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def _raw_path(raw_dir: str | Path, dataset: str) -> Path:
    candidates = [
        Path(raw_dir) / f"{dataset}.h5",
        Path(raw_dir) / f"{dataset.lower()}.h5",
        Path(raw_dir) / f"{dataset.replace('-', '_').lower()}.h5",
        Path(raw_dir) / f"{dataset}.csv",
        Path(raw_dir) / f"{dataset.upper()}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _adjacency_path(raw_dir: str | Path, dataset: str) -> Path | None:
    candidates = [
        Path(raw_dir) / f"adj_mx_{dataset}.pkl",
        Path(raw_dir) / f"adj_mx_{dataset.upper()}.pkl",
        Path(raw_dir) / f"adj_mx_{dataset.replace('-', '_')}.pkl",
        Path(raw_dir) / f"adj_mx_{dataset.replace('-', '_').upper()}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def preprocess(args: argparse.Namespace) -> dict[str, object]:
    spec = ForecastingSpec(input_length=args.input_length, horizon=args.horizon)
    dataset_name = "synthetic_smoke" if args.synthetic_smoke else args.dataset
    if args.synthetic_smoke:
        series = synthetic_series(args.synthetic_steps, args.synthetic_sensors, spec.feature_dim, args.seed)
        source = "synthetic_smoke"
        status = "processed"
    else:
        raw_path = _raw_path(args.raw_dir, args.dataset)
        if not raw_path.exists():
            metadata_path = write_missing_data_status(args.processed_dir, args.dataset, raw_path, spec)
            return {
                "dataset": args.dataset,
                "status": "raw_file_missing",
                "metadata": str(metadata_path),
            }
        if raw_path.suffix.lower() == ".csv":
            series = load_csv_series(raw_path)
        else:
            series = load_h5_series(raw_path)
        source = str(raw_path)
        status = "processed"

    adjacency = None
    adjacency_metadata = None
    if not args.synthetic_smoke:
        adj_path = _adjacency_path(args.raw_dir, args.dataset)
        if adj_path is not None:
            adjacency, adjacency_metadata = load_adjacency_pickle(adj_path)
            if adjacency.shape[0] != series.shape[1]:
                raise ValueError(
                    f"Adjacency sensors {adjacency.shape[0]} do not match data sensors {series.shape[1]}"
                )

    x, y = make_windows(series, spec)
    splits = split_windows(x, y)
    normalized, stats = normalize_splits(splits)
    metadata = {
        "dataset": dataset_name,
        "status": status,
        "source": source,
        "spec": asdict(spec),
        "raw_shape": list(series.shape),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "splits": {
            name: {"x_shape": list(pair[0].shape), "y_shape": list(pair[1].shape)}
            for name, pair in normalized.items()
        },
        "seed": args.seed,
        "synthetic_smoke_only": args.synthetic_smoke,
        "adjacency_loaded": adjacency is not None,
    }
    written = save_processed_dataset(
        normalized,
        stats,
        metadata,
        args.processed_dir,
        dataset_name,
        adjacency=adjacency,
        adjacency_metadata=adjacency_metadata,
    )
    return {
        "dataset": dataset_name,
        "status": status,
        "written": written,
        "metadata": metadata,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = preprocess(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
