"""Step115: multi-date training preflight and frozen spatiotemporal split.

This step does not train a model. It validates Step114, preserves the approved
point-level spatial split, freezes date-level weather splits, creates explicit
sample partitions, checks CUDA availability, and estimates later GPU workload.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SCRIPT_VERSION = "1.0.0"

PARTITION_MAP = {
    ("train", "train"): "train",
    ("validation", "train"): "validation_unseen_points_seen_weather",
    ("train", "validation"): "validation_seen_points_unseen_weather",
    ("validation", "validation"): "validation_unseen_points_unseen_weather",
    ("test", "train"): "test_unseen_points_seen_weather",
    ("train", "test"): "test_seen_points_unseen_weather",
    ("test", "test"): "test_unseen_points_unseen_weather",
    ("test", "validation"): "reserved_test_points_validation_weather",
    ("validation", "test"): "reserved_validation_points_test_weather",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "data_pipeline" / "configs" / "multiweather" / "step115_spatiotemporal_preflight.yaml",
    )
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def configure_logging(record_dir: Path) -> logging.Logger:
    record_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step115")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(record_dir / "step115.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    total = max(1, math.ceil(path.stat().st_size / chunk_size))
    with path.open("rb") as handle, tqdm(
        total=total, desc=f"Hashing {path.name}", unit="chunk", dynamic_ncols=True, leave=False
    ) as progress:
        while block := handle.read(chunk_size):
            digest.update(block)
            progress.update(1)
    return digest.hexdigest()


def qc(category: str, check: str, status: str, value: Any, expected: Any, details: str = "") -> dict[str, Any]:
    return {
        "category": category,
        "check": check,
        "status": status,
        "value": value,
        "expected": expected,
        "details": details,
    }


def weather_metadata(dynamic: pd.DataFrame, metrics_path: Path, split_by_date: dict[str, str]) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path, dtype={"date": str})
    summary = (
        dynamic.groupby("date", as_index=False)
        .agg(
            iso_week=("datetime_local", lambda s: dt.date.fromisoformat(str(s.iloc[0])[:10]).isocalendar().week),
            mean_temperature_c=("air_temperature", "mean"),
            maximum_temperature_c=("air_temperature", "max"),
            mean_relative_humidity_percent=("relative_humidity", "mean"),
            mean_wind_speed_m_s=("wind_speed", "mean"),
            mean_global_shortwave_w_m2=("global_shortwave_radiation", "mean"),
            maximum_global_shortwave_w_m2=("global_shortwave_radiation", "max"),
        )
    )
    regime_lookup = dict(zip(metrics["date"], metrics.get("weather_regime", pd.Series(dtype=str))))
    summary["weather_regime"] = summary["date"].map(regime_lookup).fillna("hot_clear_reference")
    summary["weather_split"] = summary["date"].map(split_by_date)
    summary["is_frozen_test"] = summary["weather_split"].eq("test")
    return summary.sort_values("date").reset_index(drop=True)


def create_split_manifest(
    manifest_path: Path,
    output_path: Path,
    split_by_date: dict[str, str],
    chunk_rows: int,
) -> tuple[Counter, int, int]:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    counts: Counter = Counter()
    row_count = 0
    mismatch_count = 0
    first = True
    for chunk in tqdm(
        pd.read_csv(manifest_path, dtype={"point_id": str, "date": str}, chunksize=chunk_rows),
        total=math.ceil(1_633_450 / chunk_rows),
        desc="Partitioning samples",
        unit="chunk",
        dynamic_ncols=True,
    ):
        chunk["weather_split"] = chunk["date"].map(split_by_date)
        mismatch_count += int(chunk["weather_split"].isna().sum())
        keys = list(zip(chunk["dataset_split"], chunk["weather_split"]))
        chunk["spatiotemporal_partition"] = [PARTITION_MAP.get(key, "UNMAPPED") for key in keys]
        mismatch_count += int(chunk["spatiotemporal_partition"].eq("UNMAPPED").sum())
        counts.update(chunk["spatiotemporal_partition"])
        chunk.to_csv(temporary, index=False, encoding="utf-8-sig", mode="w" if first else "a", header=first)
        first = False
        row_count += len(chunk)
    temporary.replace(output_path)
    return counts, row_count, mismatch_count


def audit_alignment(
    manifest_path: Path,
    labels_path: Path,
    dynamic: pd.DataFrame,
    chunk_rows: int,
) -> dict[str, int]:
    manifest_reader = pd.read_csv(
        manifest_path,
        usecols=["sample_id", "point_id", "date", "hour", "dynamic_condition_row", "label_row"],
        dtype={"point_id": str, "date": str},
        chunksize=chunk_rows,
    )
    label_reader = pd.read_csv(
        labels_path,
        usecols=["point_id", "date", "hour", "shade_rate", "tmrt_mean", "utci_mean"],
        dtype={"point_id": str, "date": str},
        chunksize=chunk_rows,
    )
    row_count = 0
    mismatch = 0
    nonfinite = 0
    duplicate_ids = 0
    seen: set[str] = set()
    dynamic_index = {(str(row.date), int(row.hour)): index for index, row in dynamic.iterrows()}
    total = math.ceil(1_633_450 / chunk_rows)
    for manifest, labels in tqdm(
        zip(manifest_reader, label_reader), total=total, desc="Auditing cross-file alignment", unit="chunk", dynamic_ncols=True
    ):
        size = len(manifest)
        if size != len(labels):
            mismatch += abs(size - len(labels)) + 1
        mismatch += int((manifest["point_id"].to_numpy() != labels["point_id"].to_numpy()).sum())
        mismatch += int((manifest["date"].to_numpy() != labels["date"].to_numpy()).sum())
        mismatch += int((manifest["hour"].to_numpy() != labels["hour"].to_numpy()).sum())
        mismatch += int((manifest["label_row"].to_numpy() != np.arange(row_count, row_count + size)).sum())
        expected_dynamic = np.asarray(
            [dynamic_index[(str(date), int(hour))] for date, hour in zip(manifest["date"], manifest["hour"])],
            dtype=np.int64,
        )
        mismatch += int((manifest["dynamic_condition_row"].to_numpy() != expected_dynamic).sum())
        targets = labels[["shade_rate", "tmrt_mean", "utci_mean"]].to_numpy(dtype=float)
        nonfinite += int((~np.isfinite(targets)).sum())
        ids = manifest["sample_id"].astype(str).tolist()
        duplicate_ids += sum(sample_id in seen for sample_id in ids)
        seen.update(ids)
        row_count += size
    return {
        "rows": row_count,
        "alignment_mismatches": mismatch,
        "nonfinite_target_cells": nonfinite,
        "unique_sample_ids": len(seen),
        "duplicate_sample_ids": duplicate_ids,
    }


def feature_ranges(dynamic: pd.DataFrame, split_by_date: dict[str, str], fields: list[str]) -> list[dict[str, Any]]:
    table = dynamic.copy()
    table["weather_split"] = table["date"].map(split_by_date)
    rows: list[dict[str, Any]] = []
    for split in tqdm(["train", "validation", "test"], desc="Profiling weather features", unit="split", dynamic_ncols=True):
        subset = table[table["weather_split"].eq(split)]
        for field in fields:
            values = subset[field].astype(float)
            rows.append(
                {
                    "weather_split": split,
                    "feature": field,
                    "count": len(values),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "mean": values.mean(),
                    "std": values.std(ddof=0),
                }
            )
    return rows


def gpu_estimate(config: dict[str, Any], partition_counts: Counter) -> dict[str, Any]:
    runtime = config["runtime"]
    train_rows = int(partition_counts["train"])
    scale = train_rows / int(runtime["original_train_rows"])
    minutes_per_seed = float(runtime["original_minutes_per_seed"]) * scale
    primary_seed_count = len(runtime["primary_seeds"])
    full_seed_count = len(runtime["full_model_seeds"])
    ablations = int(runtime["primary_ablation_count"])
    cuda_available = torch.cuda.is_available()
    return {
        "cuda_required": str(runtime["device"]).lower() == "cuda",
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3) if cuda_available else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "batch_size": int(runtime["batch_size"]),
        "mixed_precision": bool(runtime["mixed_precision"]),
        "train_rows": train_rows,
        "batches_per_epoch": math.ceil(train_rows / int(runtime["batch_size"])),
        "dataset_scale_vs_single_date": scale,
        "estimated_minutes_per_seed_at_original_epoch_count": minutes_per_seed,
        "estimated_full_model_5_seed_gpu_hours": minutes_per_seed * full_seed_count / 60,
        "estimated_primary_11_ablation_3_seed_gpu_hours": minutes_per_seed * ablations * primary_seed_count / 60,
        "estimate_note": "Linear estimate from prior Phase C runtime; Step116 smoke benchmark must replace it before formal training.",
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    root = Path(config["project_root"])
    output_dir = resolve(root, config["outputs"]["data_directory"])
    record_dir = resolve(root, config["outputs"]["record_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(record_dir)
    started = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    logger.info("Step115 started | mode=%s | version=%s", args.mode, SCRIPT_VERSION)
    failed_rows: list[dict[str, Any]] = []
    try:
        inputs = {name: resolve(root, value) for name, value in config["inputs"].items()}
        missing = [str(path) for path in inputs.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing inputs: " + "; ".join(missing))

        expected = config["expected"]
        train_dates = [str(value) for value in config["temporal_split"]["train_dates"]]
        validation_dates = [str(value) for value in config["temporal_split"]["validation_dates"]]
        test_dates = [str(value) for value in config["temporal_split"]["test_dates"]]
        all_dates = train_dates + validation_dates + test_dates
        split_by_date = {date: split for split, dates in (("train", train_dates), ("validation", validation_dates), ("test", test_dates)) for date in dates}
        if len(split_by_date) != len(all_dates):
            raise ValueError("Temporal split contains duplicate dates")

        dynamic = pd.read_csv(inputs["dynamic_conditions"], dtype={"date": str})
        input_spec = load_yaml(inputs["model_input_spec"])
        dynamic_fields = list(input_spec["dynamic_query_fields"])
        spatial = pd.read_csv(inputs["spatial_split"], dtype={"point_id": str})
        weather = weather_metadata(dynamic, inputs["date_metrics"], split_by_date)

        split_manifest = output_dir / "training_manifest_spatiotemporal.csv"
        if split_manifest.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {split_manifest}")
        counts, manifest_rows, unmapped = create_split_manifest(
            inputs["manifest"], split_manifest, split_by_date, int(config["runtime"]["chunk_rows"])
        )
        alignment = audit_alignment(
            inputs["manifest"], inputs["labels"], dynamic, int(config["runtime"]["chunk_rows"])
        )

        weather.to_csv(output_dir / "weather_date_split.csv", index=False, encoding="utf-8-sig")
        range_rows = feature_ranges(dynamic, split_by_date, dynamic_fields)
        atomic_csv(
            output_dir / "dynamic_feature_ranges_by_weather_split.csv",
            range_rows,
            ["weather_split", "feature", "count", "minimum", "maximum", "mean", "std"],
        )
        partition_rows = [
            {"spatiotemporal_partition": name, "sample_count": int(count), "percentage": count / manifest_rows}
            for name, count in sorted(counts.items())
        ]
        atomic_csv(
            output_dir / "partition_summary.csv",
            partition_rows,
            ["spatiotemporal_partition", "sample_count", "percentage"],
        )

        static_rows = len(pd.read_csv(inputs["semantic_features"], usecols=["point_id"]))
        dino_mean_rows = len(pd.read_csv(inputs["dino_mean_features"], usecols=["point_id"]))
        dino_shape = list(np.load(inputs["dino_window_features"], mmap_mode="r").shape)
        spatial_counts = spatial["dataset_split"].value_counts().to_dict()
        expected_rows = int(expected["rows"])
        expected_dates = set(str(value) for value in dynamic["date"].unique())
        qcs = [
            qc("temporal", "all dates assigned exactly once", "PASS" if set(all_dates) == expected_dates and len(all_dates) == len(set(all_dates)) else "FAIL", len(all_dates), expected["dates"]),
            qc("temporal", "weather split sizes", "PASS" if (len(train_dates), len(validation_dates), len(test_dates)) == (10, 2, 2) else "FAIL", f"{len(train_dates)}/{len(validation_dates)}/{len(test_dates)}", "10/2/2"),
            qc("spatial", "point count", "PASS" if len(spatial) == int(expected["points"]) else "FAIL", len(spatial), expected["points"]),
            qc("spatial", "original split counts preserved", "PASS" if all(int(spatial_counts.get(k, 0)) == int(v) for k, v in expected["spatial_points"].items()) else "FAIL", json.dumps(spatial_counts), json.dumps(expected["spatial_points"])),
            qc("manifest", "row count", "PASS" if manifest_rows == expected_rows else "FAIL", manifest_rows, expected_rows),
            qc("manifest", "all rows mapped", "PASS" if unmapped == 0 else "FAIL", unmapped, 0),
            qc("alignment", "cross-file row references", "PASS" if alignment["alignment_mismatches"] == 0 else "FAIL", alignment["alignment_mismatches"], 0),
            qc("alignment", "unique sample ids", "PASS" if alignment["unique_sample_ids"] == expected_rows and alignment["duplicate_sample_ids"] == 0 else "FAIL", alignment["unique_sample_ids"], expected_rows),
            qc("targets", "finite shade/Tmrt/UTCI", "PASS" if alignment["nonfinite_target_cells"] == 0 else "FAIL", alignment["nonfinite_target_cells"], 0),
            qc("static", "semantic rows", "PASS" if static_rows == int(expected["points"]) else "FAIL", static_rows, expected["points"]),
            qc("static", "DINO mean rows", "PASS" if dino_mean_rows == int(expected["points"]) else "FAIL", dino_mean_rows, expected["points"]),
            qc("static", "directional DINO shape", "PASS" if dino_shape == [int(expected["points"]), 8, 768] else "FAIL", str(dino_shape), f"[{expected['points']}, 8, 768]"),
        ]
        estimate = gpu_estimate(config, counts)
        qcs.append(qc("runtime", "CUDA available", "PASS" if estimate["cuda_available"] else "FAIL", estimate["gpu_name"], "CUDA GPU"))
        atomic_csv(record_dir / "preflight_qc.csv", qcs, ["category", "check", "status", "value", "expected", "details"])
        atomic_json(record_dir / "gpu_workload_estimate.json", estimate)

        input_hashes = {name: sha256(path) for name, path in tqdm(inputs.items(), desc="Fingerprinting inputs", unit="file", dynamic_ncols=True) if name in {"manifest", "labels", "dynamic_conditions", "spatial_split", "model_input_spec"}}
        status = "PASS" if all(row["status"] == "PASS" for row in qcs) else "FAIL"
        ended = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
        summary = {
            "step": 115,
            "status": status,
            "mode": args.mode,
            "script_version": SCRIPT_VERSION,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "elapsed_seconds": (ended - started).total_seconds(),
            "dates": {"train": train_dates, "validation": validation_dates, "test": test_dates},
            "spatial_point_counts": {key: int(value) for key, value in spatial_counts.items()},
            "sample_partition_counts": {key: int(value) for key, value in sorted(counts.items())},
            "manifest_rows": manifest_rows,
            "alignment": alignment,
            "gpu_estimate": estimate,
            "input_sha256": input_hashes,
            "formal_training_started": False,
            "next_step": "Step116 multi-date CUDA smoke benchmark; requires user approval before formal training.",
        }
        atomic_json(output_dir / "summary.json", summary)
        atomic_csv(record_dir / "failed_files.csv", [], ["point_id", "filename", "error_message"])
        logger.info("Step115 ended | status=%s | rows=%d | elapsed=%.2fs", status, manifest_rows, summary["elapsed_seconds"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if status == "PASS" else 2
    except Exception as error:
        logger.exception("Step115 failed")
        failed_rows.append({"point_id": "", "filename": str(args.config), "error_message": str(error)})
        atomic_csv(record_dir / "failed_files.csv", failed_rows, ["point_id", "filename", "error_message"])
        atomic_json(record_dir / "failure.json", {"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
