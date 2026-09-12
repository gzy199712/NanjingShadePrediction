"""Frozen-model inference helpers shared by Phase F preflight and run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from training.scripts.Step19_evaluate_model import (
    infer_split,
    load_frozen_models,
    make_loader,
)
from training.src.trainer import TrainingScalers, prepare_batch
from training.src.utci import calculate_utci


@dataclass
class FrozenModelBundle:
    models: dict[int, torch.nn.Module]
    scalers: TrainingScalers
    checkpoint_records: list[dict[str, Any]]
    device: torch.device


def load_frozen_bundle(
    config: dict[str, Any],
    specs: Any,
) -> FrozenModelBundle:
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen inference requires CUDA; CPU fallback is forbidden")
    device = torch.device("cuda")
    models, scalers, checkpoint_records = load_frozen_models(
        config, specs, device
    )
    return FrozenModelBundle(
        models=models,
        scalers=scalers,
        checkpoint_records=checkpoint_records,
        device=device,
    )


def infer_frozen_subset(
    *,
    manifest: pd.DataFrame,
    data: Any,
    specs: Any,
    config: dict[str, Any],
    bundle: FrozenModelBundle,
    description: str,
) -> tuple[pd.DataFrame, Any]:
    """Use the exact Step19 inference implementation on a fixed subset."""
    return infer_split(
        description,
        manifest,
        data,
        specs,
        config,
        bundle.models,
        bundle.scalers,
        bundle.device,
    )


def benchmark_frozen_models(
    *,
    manifest: pd.DataFrame,
    data: Any,
    specs: Any,
    config: dict[str, Any],
    bundle: FrozenModelBundle,
) -> list[dict[str, Any]]:
    """Measure per-seed forward throughput on fixed records without saving output."""
    loader = make_loader(manifest, data, specs, config)
    prepared_batches: list[dict[str, Any]] = []
    for batch in tqdm(
        loader,
        desc="Preparing GPU benchmark batches",
        unit="batch",
        dynamic_ncols=True,
    ):
        prepared_batches.append(
            prepare_batch(batch, bundle.device, bundle.scalers, True)
        )
    records: list[dict[str, Any]] = []
    for seed, model in tqdm(
        bundle.models.items(),
        desc="Benchmarking frozen seeds",
        unit="seed",
        dynamic_ncols=True,
    ):
        torch.cuda.synchronize()
        started = time.perf_counter()
        record_count = 0
        finite = True
        with torch.inference_mode():
            for prepared in prepared_batches:
                with torch.amp.autocast("cuda", enabled=True):
                    output = model(
                        prepared["semantic_features"],
                        prepared["dino_mean"],
                        prepared["dino_windows"],
                        prepared["dynamic_features"],
                    )
                record_count += int(output["shade_prediction"].shape[0])
                finite &= bool(
                    torch.isfinite(output["shade_prediction"]).all()
                    and torch.isfinite(output["tmrt_standardized"]).all()
                    and torch.isfinite(output["attention_weights"]).all()
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        records.append(
            {
                "seed": seed,
                "records": record_count,
                "batches": len(prepared_batches),
                "elapsed_seconds": elapsed,
                "records_per_second": record_count / elapsed,
                "finite_outputs": finite,
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            }
        )
    return records


def infer_single_seed(
    *,
    seed: int,
    model: torch.nn.Module,
    manifest: pd.DataFrame,
    data: Any,
    specs: Any,
    config: dict[str, Any],
    scalers: TrainingScalers,
    device: torch.device,
    mixed_precision: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Run one frozen seed in manifest order without exposing labels to the model."""
    loader = make_loader(manifest, data, specs, config)
    storage: dict[str, list[Any]] = {
        "sample_id": [],
        "point_id": [],
        "date": [],
        "hour": [],
        "datetime_local": [],
        "dataset_split": [],
        "shade_pred": [],
        "tmrt_pred": [],
    }
    attention: list[np.ndarray] = []
    failed: list[dict[str, Any]] = []
    dynamic_positions = {
        field: specs.dynamic_fields.index(field)
        for field in ("air_temperature", "relative_humidity", "wind_speed")
    }
    dynamic_values = {field: [] for field in dynamic_positions}
    started = time.perf_counter()
    nan_count = 0
    progress = tqdm(
        loader,
        desc=f"Full inference seed {seed}",
        unit="batch",
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(progress):
            batch_size = len(batch["sample_id"])
            try:
                prepared = prepare_batch(batch, device, scalers, True)
                with torch.amp.autocast("cuda", enabled=mixed_precision):
                    output = model(
                        prepared["semantic_features"],
                        prepared["dino_mean"],
                        prepared["dino_windows"],
                        prepared["dynamic_features"],
                    )
                shade = output["shade_prediction"].float().cpu().numpy()
                tmrt = (
                    output["tmrt_standardized"] * scalers.tmrt_scale
                    + scalers.tmrt_mean
                ).float().cpu().numpy()
                weights = output["attention_weights"].float().cpu().numpy()
                if shade.shape != (batch_size,) or tmrt.shape != (batch_size,):
                    raise RuntimeError(
                        f"prediction shape mismatch: {shade.shape}, {tmrt.shape}"
                    )
                if weights.shape != (batch_size, len(specs.window_azimuth)):
                    raise RuntimeError(
                        f"attention shape mismatch: {weights.shape}"
                    )
            except Exception as error:
                shade = np.full(batch_size, np.nan, dtype=np.float32)
                tmrt = np.full(batch_size, np.nan, dtype=np.float32)
                weights = np.full(
                    (batch_size, len(specs.window_azimuth)),
                    np.nan,
                    dtype=np.float32,
                )
                for sample_id, point_id in zip(
                    batch["sample_id"], batch["point_id"], strict=True
                ):
                    failed.append(
                        {
                            "seed": seed,
                            "batch_index": batch_index,
                            "sample_id": str(sample_id),
                            "point_id": str(point_id),
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
            sample_ids = [str(value) for value in batch["sample_id"]]
            storage["sample_id"].extend(sample_ids)
            storage["point_id"].extend(
                [str(value) for value in batch["point_id"]]
            )
            storage["hour"].extend(batch["hour"].numpy().astype(int).tolist())
            storage["dataset_split"].extend(list(batch["dataset_split"]))
            positions = batch["manifest_row"].numpy().astype(int)
            storage["date"].extend(
                manifest.iloc[positions]["date"].astype(str).tolist()
            )
            dynamic_rows = manifest.iloc[positions][
                "dynamic_condition_row"
            ].to_numpy(dtype=int)
            storage["datetime_local"].extend(
                data.dynamic.iloc[dynamic_rows]["datetime_local"]
                .astype(str)
                .tolist()
            )
            dynamic = batch["dynamic_features"].numpy()
            for field, position in dynamic_positions.items():
                dynamic_values[field].extend(
                    dynamic[:, position].astype(float).tolist()
                )
            storage["shade_pred"].extend(shade.astype(float).tolist())
            storage["tmrt_pred"].extend(tmrt.astype(float).tolist())
            attention.append(weights)
            nan_count += int(
                (~np.isfinite(shade)).sum()
                + (~np.isfinite(tmrt)).sum()
                + (~np.isfinite(weights)).sum()
            )
            progress.set_postfix(
                records=len(storage["sample_id"]),
                total=len(manifest),
                gpu=f"{torch.cuda.memory_allocated()/1024**2:.0f}MB",
                nan=nan_count,
                failed=len(failed),
            )
    frame = pd.DataFrame(storage)
    prediction, utci_qc = calculate_utci(
        np.asarray(dynamic_values["air_temperature"], dtype=float),
        frame["tmrt_pred"].to_numpy(dtype=float),
        np.asarray(dynamic_values["wind_speed"], dtype=float),
        np.asarray(dynamic_values["relative_humidity"], dtype=float),
        description=f"Full inference UTCI seed {seed}",
    )
    frame["utci_pred"] = prediction
    attention_matrix = np.concatenate(attention, axis=0)
    for index, azimuth in enumerate(specs.window_azimuth):
        frame[f"attention_{int(azimuth)}"] = attention_matrix[:, index]
    elapsed = time.perf_counter() - started
    stats = {
        "seed": seed,
        "record_count": len(frame),
        "batch_count": len(loader),
        "elapsed_seconds": elapsed,
        "records_per_second": len(frame) / elapsed,
        "failed_count": len(failed),
        "nonfinite_count": int(
            (~np.isfinite(frame[["shade_pred", "tmrt_pred", "utci_pred"]]))
            .to_numpy()
            .sum()
        ),
        "utci_qc": utci_qc.as_dict(),
    }
    return frame, failed, stats
