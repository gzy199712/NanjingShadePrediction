"""Step116: CUDA smoke benchmark for the frozen multi-date split.

The script fits scalers exclusively on the full Train-point x Train-weather
partition, runs bounded CUDA forward/backward benchmarks, and performs a small
two-epoch smoke fit. It never saves a formal checkpoint and never evaluates a
frozen Test partition.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.src.data_loading import (  # noqa: E402
    FormalData,
    ManifestDataset,
    TrainingPaths,
    load_specs,
    load_yaml,
)
from training.src.losses import MultitaskSmoothL1Loss  # noqa: E402
from training.src.model import SolarDirectionalMultitaskModel  # noqa: E402
from training.src.reproducibility import seed_everything  # noqa: E402
from training.src.trainer import TrainingScalers, prepare_batch  # noqa: E402


SCRIPT_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "training" / "configs" / "multidate_smoke.yaml",
    )
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def configure_logger(record_dir: Path) -> logging.Logger:
    record_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step116")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(record_dir / "step116.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def select_points(spatial_path: Path, split: str, count: int, seed: int) -> set[str]:
    spatial = pd.read_csv(spatial_path, dtype={"point_id": str})
    candidates = spatial.loc[spatial["dataset_split"].eq(split), "point_id"].to_numpy()
    generator = np.random.default_rng(seed)
    return set(generator.choice(candidates, size=count, replace=False).astype(str))


def collect_smoke_manifests(
    manifest_path: Path,
    train_points: set[str],
    validation_points: set[str],
    chunk_rows: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    train_chunks: list[pd.DataFrame] = []
    validation_chunks: list[pd.DataFrame] = []
    all_train_static_rows: set[int] = set()
    reader = pd.read_csv(manifest_path, dtype={"point_id": str, "date": str}, chunksize=chunk_rows)
    for chunk in tqdm(reader, total=math.ceil(1_633_450 / chunk_rows), desc="Selecting smoke samples", unit="chunk", dynamic_ncols=True):
        full_train_mask = chunk["spatiotemporal_partition"].eq("train")
        all_train_static_rows.update(
            chunk.loc[full_train_mask, "static_feature_row"].astype(int).tolist()
        )
        train = chunk[full_train_mask & chunk["point_id"].isin(train_points)].copy()
        validation = chunk[
            chunk["spatiotemporal_partition"].eq("validation_unseen_points_unseen_weather")
            & chunk["point_id"].isin(validation_points)
        ].copy()
        if not train.empty:
            train_chunks.append(train)
        if not validation.empty:
            validation_chunks.append(validation)
    if not train_chunks or not validation_chunks:
        raise RuntimeError("Smoke subset selection returned no records")
    train = pd.concat(train_chunks, ignore_index=True).sort_values(["point_id", "date", "hour"]).reset_index(drop=True)
    validation = pd.concat(validation_chunks, ignore_index=True).sort_values(["point_id", "date", "hour"]).reset_index(drop=True)
    return train, validation, all_train_static_rows


def stream_train_tmrt_moments(
    manifest_path: Path,
    labels_path: Path,
    chunk_rows: int = 100_000,
) -> tuple[float, float, int, int]:
    count = 0
    total = 0.0
    total_square = 0.0
    leakage = 0
    manifest_reader = pd.read_csv(
        manifest_path,
        usecols=["spatiotemporal_partition", "weather_split", "dataset_split"],
        chunksize=chunk_rows,
    )
    label_reader = pd.read_csv(labels_path, usecols=["tmrt_mean"], chunksize=chunk_rows)
    total_chunks = math.ceil(1_633_450 / chunk_rows)
    for manifest, labels in tqdm(
        zip(manifest_reader, label_reader), total=total_chunks, desc="Fitting Train-only Tmrt scaler", unit="chunk", dynamic_ncols=True
    ):
        mask = manifest["spatiotemporal_partition"].eq("train").to_numpy()
        leakage += int((mask & (~manifest["weather_split"].eq("train").to_numpy())).sum())
        leakage += int((mask & (~manifest["dataset_split"].eq("train").to_numpy())).sum())
        values = labels.loc[mask, "tmrt_mean"].to_numpy(dtype=np.float64)
        count += len(values)
        total += float(values.sum())
        total_square += float(np.square(values).sum())
    mean = total / count
    variance = max(0.0, total_square / count - mean * mean)
    return mean, math.sqrt(variance), count, leakage


def load_compact_labels(labels_path: Path, manifests: list[pd.DataFrame], chunk_rows: int = 100_000) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    required = sorted({int(value) for manifest in manifests for value in manifest["label_row"]})
    required_set = set(required)
    rows: list[pd.DataFrame] = []
    offset = 0
    for chunk in tqdm(
        pd.read_csv(labels_path, usecols=["shade_rate", "tmrt_mean", "utci_mean"], chunksize=chunk_rows),
        total=math.ceil(1_633_450 / chunk_rows),
        desc="Loading compact smoke labels",
        unit="chunk",
        dynamic_ncols=True,
    ):
        global_rows = np.arange(offset, offset + len(chunk))
        mask = np.fromiter((int(value) in required_set for value in global_rows), dtype=bool, count=len(global_rows))
        if mask.any():
            selected = chunk.loc[mask].copy()
            selected["global_label_row"] = global_rows[mask]
            rows.append(selected)
        offset += len(chunk)
    compact = pd.concat(rows, ignore_index=True).sort_values("global_label_row").reset_index(drop=True)
    mapping = {int(value): index for index, value in enumerate(compact["global_label_row"])}
    compact = compact[["shade_rate", "tmrt_mean", "utci_mean"]]
    remapped: list[pd.DataFrame] = []
    for manifest in manifests:
        result = manifest.copy()
        result["label_row"] = result["label_row"].astype(int).map(mapping)
        if result["label_row"].isna().any():
            raise RuntimeError("Compact label remapping failed")
        remapped.append(result)
    return compact, remapped


def fit_scalers(
    semantic: pd.DataFrame,
    dynamic: pd.DataFrame,
    specs: Any,
    train_static_rows: set[int],
    tmrt_mean: float,
    tmrt_scale: float,
) -> TrainingScalers:
    train_dates = {
        "2024-07-05", "2024-07-08", "2024-07-21", "2024-07-28", "2024-07-29",
        "2024-08-06", "2024-08-11", "2024-08-12", "2024-08-22", "2024-08-24",
    }
    semantic_values = semantic.iloc[sorted(train_static_rows)][specs.semantic_fields].to_numpy(dtype=np.float64)
    dynamic_values = dynamic.loc[dynamic["date"].astype(str).isin(train_dates), specs.dynamic_fields].to_numpy(dtype=np.float64)

    def moments(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        return torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(scale, dtype=torch.float32)

    semantic_mean, semantic_scale = moments(semantic_values)
    dynamic_mean, dynamic_scale = moments(dynamic_values)
    return TrainingScalers(
        semantic_mean=semantic_mean,
        semantic_scale=semantic_scale,
        dynamic_mean=dynamic_mean,
        dynamic_scale=dynamic_scale,
        tmrt_mean=torch.tensor([tmrt_mean], dtype=torch.float32),
        tmrt_scale=torch.tensor([tmrt_scale], dtype=torch.float32),
    )


def make_model(config: dict[str, Any], specs: Any, device: torch.device) -> SolarDirectionalMultitaskModel:
    return SolarDirectionalMultitaskModel(
        len(specs.semantic_fields),
        len(specs.dino_mean_fields),
        len(specs.dynamic_fields),
        specs.window_azimuth,
        config["model"],
        float(config["training"]["dropout"]),
    ).to(device)


def make_loader(dataset: ManifestDataset, batch_size: int, workers: int, pin_memory: bool, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def train_step(model: Any, batch: dict[str, Any], device: torch.device, scalers: TrainingScalers, criterion: Any, optimizer: Any, amp_scaler: Any, use_amp: bool, gradient_clip: float) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    prepared = prepare_batch(batch, device, scalers, True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        output = model(
            prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"]
        )
        losses = criterion(
            output["shade_prediction"], prepared["shade_target"], output["tmrt_standardized"], prepared["tmrt_target_standardized"]
        )
    amp_scaler.scale(losses.total).backward()
    amp_scaler.unscale_(optimizer)
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip))
    amp_scaler.step(optimizer)
    amp_scaler.update()
    return float(losses.total.detach()), gradient_norm


def benchmark_batch_size(config: dict[str, Any], specs: Any, dataset: ManifestDataset, scalers: TrainingScalers, batch_size: int, device: torch.device) -> dict[str, Any]:
    smoke = config["smoke_test"]
    seed_everything(int(smoke["seed"]))
    loader = make_loader(dataset, batch_size, int(smoke["num_workers"]), True, True, int(smoke["seed"]))
    model = make_model(config, specs, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(smoke["learning_rate"]), weight_decay=float(smoke["weight_decay"]))
    criterion = MultitaskSmoothL1Loss(float(config["training"]["lambda_shade"]), float(config["training"]["lambda_tmrt"]))
    amp_scaler = torch.amp.GradScaler("cuda", enabled=True)
    gpu_scalers = scalers.to(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []; gradients: list[float] = []
    warmup = int(smoke["warmup_batches"]); timed = int(smoke["timed_batches"])
    iterator = iter(loader)
    for _ in tqdm(range(warmup), desc=f"Warmup bs={batch_size}", unit="batch", dynamic_ncols=True):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        train_step(model, batch, device, gpu_scalers, criterion, optimizer, amp_scaler, True, float(smoke["gradient_clip_norm"]))
    torch.cuda.synchronize(); start = time.perf_counter()
    for _ in tqdm(range(timed), desc=f"Benchmark bs={batch_size}", unit="batch", dynamic_ncols=True):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        loss, gradient = train_step(model, batch, device, gpu_scalers, criterion, optimizer, amp_scaler, True, float(smoke["gradient_clip_norm"]))
        losses.append(loss); gradients.append(gradient)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    result = {
        "batch_size": batch_size,
        "timed_batches": timed,
        "elapsed_seconds": elapsed,
        "samples_per_second": batch_size * timed / elapsed,
        "milliseconds_per_batch": elapsed * 1000 / timed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "mean_loss": float(np.mean(losses)),
        "maximum_gradient_norm": float(np.max(gradients)),
        "finite": bool(np.isfinite(losses).all() and np.isfinite(gradients).all()),
    }
    del model, optimizer, criterion, amp_scaler, loader, iterator
    torch.cuda.empty_cache()
    return result


def smoke_epochs(config: dict[str, Any], specs: Any, train_dataset: ManifestDataset, validation_dataset: ManifestDataset, scalers: TrainingScalers, batch_size: int, device: torch.device) -> list[dict[str, Any]]:
    smoke = config["smoke_test"]
    seed_everything(int(smoke["seed"]))
    train_loader = make_loader(train_dataset, batch_size, int(smoke["num_workers"]), True, True, int(smoke["seed"]))
    validation_loader = make_loader(validation_dataset, batch_size, int(smoke["num_workers"]), True, False, int(smoke["seed"]))
    model = make_model(config, specs, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(smoke["learning_rate"]), weight_decay=float(smoke["weight_decay"]))
    criterion = MultitaskSmoothL1Loss(float(config["training"]["lambda_shade"]), float(config["training"]["lambda_tmrt"]))
    amp_scaler = torch.amp.GradScaler("cuda", enabled=True)
    gpu_scalers = scalers.to(device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(smoke["smoke_epochs"]) + 1):
        model.train(); train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Smoke train {epoch}", unit="batch", dynamic_ncols=True):
            loss, _ = train_step(model, batch, device, gpu_scalers, criterion, optimizer, amp_scaler, True, float(smoke["gradient_clip_norm"]))
            train_losses.append(loss)
        model.eval(); validation_losses: list[float] = []
        with torch.inference_mode():
            for batch in tqdm(validation_loader, desc=f"Smoke validation {epoch}", unit="batch", dynamic_ncols=True):
                prepared = prepare_batch(batch, device, gpu_scalers, True)
                with torch.amp.autocast("cuda", enabled=True):
                    output = model(prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"])
                    losses = criterion(output["shade_prediction"], prepared["shade_target"], output["tmrt_standardized"], prepared["tmrt_target_standardized"])
                validation_losses.append(float(losses.total))
        history.append({"epoch": epoch, "train_total_loss": float(np.mean(train_losses)), "validation_total_loss": float(np.mean(validation_losses)), "finite": bool(np.isfinite(train_losses).all() and np.isfinite(validation_losses).all())})
    del model, optimizer, criterion, amp_scaler, train_loader, validation_loader
    torch.cuda.empty_cache()
    return history


def main() -> int:
    args = parse_args()
    if args.mode == "run" and not args.approved_by_user:
        raise PermissionError("Step116 run requires --approved-by-user")
    config_path = args.config.resolve(); config = load_yaml(config_path)
    paths = TrainingPaths.from_config(config_path, config)
    smoke = config["smoke_test"]
    report_dir = project_path(paths.project_root, smoke["report_dir"])
    metrics_dir = project_path(paths.project_root, smoke["metrics_dir"])
    record_dir = project_path(paths.project_root, smoke["record_dir"])
    for directory in (report_dir, metrics_dir, record_dir): directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(record_dir)
    started = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    logger.info("Step116 started | mode=%s | version=%s", args.mode, SCRIPT_VERSION)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        if not args.overwrite and (report_dir / "summary.json").exists():
            raise FileExistsError("Step116 output exists; use --overwrite")
        specs = load_specs(paths)
        train_points = select_points(paths.spatial_split, "train", int(smoke["train_point_count"]), int(smoke["seed"]))
        validation_points = select_points(paths.spatial_split, "validation", int(smoke["validation_point_count"]), int(smoke["seed"]) + 1)
        train_manifest, validation_manifest, train_static_rows = collect_smoke_manifests(paths.manifest, train_points, validation_points)
        tmrt_mean, tmrt_scale, scaler_rows, leakage = stream_train_tmrt_moments(paths.manifest, paths.labels)
        semantic = pd.read_csv(paths.semantic_features)
        dino_mean = pd.read_csv(paths.dino_mean_features)
        dynamic = pd.read_csv(paths.dynamic_conditions, dtype={"date": str})
        directional = np.load(paths.dino_window_features, mmap_mode="r")
        compact_labels, remapped = load_compact_labels(paths.labels, [train_manifest, validation_manifest])
        train_manifest, validation_manifest = remapped
        data = FormalData(
            manifest=pd.DataFrame(), semantic=semantic, dino_mean=dino_mean,
            dino_window=directional, dino_index=pd.DataFrame(), dynamic=dynamic,
            labels=compact_labels, split=pd.DataFrame(),
        )
        scalers = fit_scalers(semantic, dynamic, specs, train_static_rows, tmrt_mean, tmrt_scale)
        train_dataset = ManifestDataset(train_manifest, semantic, dino_mean, directional, dynamic, compact_labels, specs)
        validation_dataset = ManifestDataset(validation_manifest, semantic, dino_mean, directional, dynamic, compact_labels, specs)
        device = torch.device("cuda:0")
        benchmark_rows = [benchmark_batch_size(config, specs, train_dataset, scalers, int(size), device) for size in tqdm(smoke["benchmark_batch_sizes"], desc="Benchmarking batch sizes", unit="size", dynamic_ncols=True)]
        valid_benchmarks = [row for row in benchmark_rows if row["finite"] and row["peak_reserved_mb"] < torch.cuda.get_device_properties(0).total_memory / 1024**2 * 0.9]
        if not valid_benchmarks:
            raise RuntimeError("No candidate batch size passed finite/memory checks")
        recommended = int(max(valid_benchmarks, key=lambda row: row["samples_per_second"])["batch_size"])
        history = smoke_epochs(config, specs, train_dataset, validation_dataset, scalers, recommended, device)
        atomic_csv(metrics_dir / "batch_size_benchmark.csv", benchmark_rows, list(benchmark_rows[0]))
        atomic_csv(metrics_dir / "smoke_history.csv", history, list(history[0]))
        scaler_provenance = {
            "fit_partition": "train = Train points x Train weather only",
            "fit_rows": scaler_rows,
            "expected_fit_rows": 816660,
            "leakage_rows": leakage,
            "tmrt_mean": tmrt_mean,
            "tmrt_scale": tmrt_scale,
            "semantic_fit_rows": len(train_static_rows),
            "dynamic_fit_rows": 130,
        }
        atomic_json(metrics_dir / "scaler_provenance.json", scaler_provenance)
        train_rows = 816660
        batches_per_epoch = math.ceil(train_rows / recommended)
        observed = max(row for row in benchmark_rows if int(row["batch_size"]) == recommended)
        seconds_per_epoch = train_rows / observed["samples_per_second"]
        estimated_hours_per_seed_14_epochs = seconds_per_epoch * 14 / 3600
        status = "PASS" if leakage == 0 and scaler_rows == 816660 and all(row["finite"] for row in benchmark_rows) and all(row["finite"] for row in history) else "FAIL"
        ended = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
        summary = {
            "step": 116,
            "status": status,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "elapsed_seconds": (ended - started).total_seconds(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "train_smoke_points": len(train_points),
            "train_smoke_records": len(train_manifest),
            "validation_smoke_points": len(validation_points),
            "validation_smoke_records": len(validation_manifest),
            "validation_partition": "validation_unseen_points_unseen_weather",
            "scaler_provenance": scaler_provenance,
            "benchmarks": benchmark_rows,
            "recommended_batch_size": recommended,
            "formal_batches_per_epoch": batches_per_epoch,
            "estimated_gpu_hours_per_seed_at_14_epochs": estimated_hours_per_seed_14_epochs,
            "estimated_gpu_hours_5_seeds": estimated_hours_per_seed_14_epochs * 5,
            "history": history,
            "formal_checkpoint_written": False,
            "frozen_test_evaluated": False,
            "next_step": "Step117 formal multi-date training; requires explicit approval.",
        }
        atomic_json(report_dir / "summary.json", summary)
        atomic_csv(record_dir / "failed_files.csv", [], ["point_id", "filename", "error_message"])
        logger.info("Step116 ended | status=%s | recommended_bs=%d | elapsed=%.2fs", status, recommended, summary["elapsed_seconds"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if status == "PASS" else 2
    except Exception as error:
        logger.exception("Step116 failed")
        atomic_csv(record_dir / "failed_files.csv", [{"point_id": "", "filename": str(config_path), "error_message": str(error)}], ["point_id", "filename", "error_message"])
        atomic_json(record_dir / "failure.json", {"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
