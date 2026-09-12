"""Reusable CUDA trainer for the formal Phase C multitask experiment."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.src.data_loading import FormalData, FormalSpecs, ManifestDataset
from training.src.logging_utils import atomic_csv, atomic_json
from training.src.losses import MultitaskSmoothL1Loss
from training.src.metrics import regression_metrics
from training.src.model import SolarDirectionalMultitaskModel
from training.src.reproducibility import seed_everything


@dataclass
class TrainingScalers:
    semantic_mean: torch.Tensor
    semantic_scale: torch.Tensor
    dynamic_mean: torch.Tensor
    dynamic_scale: torch.Tensor
    tmrt_mean: torch.Tensor
    tmrt_scale: torch.Tensor

    def to(self, device: torch.device) -> "TrainingScalers":
        return TrainingScalers(
            **{key: value.to(device) for key, value in self.__dict__.items()}
        )

    def state_dict(self) -> dict[str, list[float]]:
        return {
            key: value.detach().cpu().numpy().reshape(-1).tolist()
            for key, value in self.__dict__.items()
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "TrainingScalers":
        return cls(
            **{
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in state.items()
            }
        )


@dataclass(frozen=True)
class TrainerSettings:
    seed: int
    batch_size: int
    num_workers: int
    max_epochs: int
    patience: int
    min_delta: float
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    lambda_shade: float
    lambda_tmrt: float
    mixed_precision: bool
    pin_memory: bool
    dropout: float


def _row_indices(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} contains invalid row indices")
    return values.astype(np.int64)


def fit_train_scalers(
    train_manifest: pd.DataFrame,
    data: FormalData,
    specs: FormalSpecs,
) -> tuple[TrainingScalers, dict[str, Any]]:
    """Fit every scaler exclusively on rows referenced by the Train split."""
    static_rows = np.unique(
        _row_indices(train_manifest["static_feature_row"], "static_feature_row")
    )
    dynamic_rows = _row_indices(
        train_manifest["dynamic_condition_row"], "dynamic_condition_row"
    )
    label_rows = _row_indices(train_manifest["label_row"], "label_row")
    semantic = data.semantic.iloc[static_rows][specs.semantic_fields].to_numpy(
        dtype=np.float64
    )
    dynamic = data.dynamic.iloc[dynamic_rows][specs.dynamic_fields].to_numpy(
        dtype=np.float64
    )
    tmrt = data.labels.iloc[label_rows][specs.tmrt_target].to_numpy(
        dtype=np.float64
    )

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        return mean, scale

    semantic_mean, semantic_scale = moments(semantic)
    dynamic_mean, dynamic_scale = moments(dynamic)
    tmrt_mean = np.asarray([tmrt.mean()], dtype=np.float64)
    tmrt_scale = np.asarray([tmrt.std()], dtype=np.float64)
    tmrt_scale[tmrt_scale == 0] = 1.0

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32)

    scalers = TrainingScalers(
        semantic_mean=tensor(semantic_mean),
        semantic_scale=tensor(semantic_scale),
        dynamic_mean=tensor(dynamic_mean),
        dynamic_scale=tensor(dynamic_scale),
        tmrt_mean=tensor(tmrt_mean),
        tmrt_scale=tensor(tmrt_scale),
    )
    provenance = {
        "fit_split": "Train",
        "train_records": int(len(train_manifest)),
        "train_unique_points": int(train_manifest["point_id"].nunique()),
        "semantic_unique_rows": int(len(static_rows)),
        "dynamic_rows": int(len(dynamic_rows)),
        "tmrt_rows": int(len(label_rows)),
        "semantic_fields": specs.semantic_fields,
        "dynamic_fields": specs.dynamic_fields,
        "tmrt_field": specs.tmrt_target,
        "tmrt_mean": float(tmrt_mean[0]),
        "tmrt_scale": float(tmrt_scale[0]),
    }
    return scalers, provenance


def save_scalers(
    path: Path, scalers: TrainingScalers, provenance: dict[str, Any]
) -> None:
    atomic_json(path, {"scalers": scalers.state_dict(), "provenance": provenance})


def prepare_batch(
    batch: dict[str, Any],
    device: torch.device,
    scalers: TrainingScalers,
    non_blocking: bool,
) -> dict[str, Any]:
    semantic = batch["semantic_features"].to(device, non_blocking=non_blocking)
    dynamic = batch["dynamic_features"].to(device, non_blocking=non_blocking)
    tmrt = batch["tmrt_target"].to(device, non_blocking=non_blocking)
    return {
        "semantic_features": (semantic - scalers.semantic_mean)
        / scalers.semantic_scale,
        "dino_mean": batch["dino_mean"].to(device, non_blocking=non_blocking),
        "dino_windows": batch["dino_windows"].to(
            device, non_blocking=non_blocking
        ),
        "dynamic_features": (dynamic - scalers.dynamic_mean)
        / scalers.dynamic_scale,
        "shade_target": batch["shade_target"].to(
            device, non_blocking=non_blocking
        ),
        "tmrt_target_standardized": (tmrt - scalers.tmrt_mean)
        / scalers.tmrt_scale,
        "tmrt_target_celsius": tmrt,
    }


def make_loaders(
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    data: FormalData,
    specs: FormalSpecs,
    settings: TrainerSettings,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = ManifestDataset(
        train_manifest,
        data.semantic,
        data.dino_mean,
        data.dino_window,
        data.dynamic,
        data.labels,
        specs,
    )
    validation_dataset = ManifestDataset(
        validation_manifest,
        data.semantic,
        data.dino_mean,
        data.dino_window,
        data.dynamic,
        data.labels,
        specs,
    )
    generator = torch.Generator()
    generator.manual_seed(settings.seed)
    common = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": settings.pin_memory,
        "persistent_workers": settings.num_workers > 0,
    }
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            drop_last=False,
            **common,
        ),
        DataLoader(validation_dataset, shuffle=False, drop_last=False, **common),
    )


def gpu_memory_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0


def validate_epoch(
    model: SolarDirectionalMultitaskModel,
    loader: DataLoader,
    device: torch.device,
    scalers: TrainingScalers,
    criterion: MultitaskSmoothL1Loss,
    use_amp: bool,
    epoch: int,
    max_epochs: int,
) -> dict[str, Any]:
    model.eval()
    sums = {"total": 0.0, "shade": 0.0, "tmrt": 0.0, "count": 0}
    shade_truth: list[np.ndarray] = []
    shade_prediction: list[np.ndarray] = []
    tmrt_truth: list[np.ndarray] = []
    tmrt_prediction: list[np.ndarray] = []
    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=f"Validation {epoch}/{max_epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            prepared = prepare_batch(batch, device, scalers, True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(
                    prepared["semantic_features"],
                    prepared["dino_mean"],
                    prepared["dino_windows"],
                    prepared["dynamic_features"],
                )
                losses = criterion(
                    output["shade_prediction"],
                    prepared["shade_target"],
                    output["tmrt_standardized"],
                    prepared["tmrt_target_standardized"],
                )
            count = int(prepared["shade_target"].shape[0])
            sums["total"] += float(losses.total) * count
            sums["shade"] += float(losses.shade) * count
            sums["tmrt"] += float(losses.tmrt) * count
            sums["count"] += count
            tmrt_celsius = (
                output["tmrt_standardized"] * scalers.tmrt_scale
                + scalers.tmrt_mean
            )
            shade_truth.append(prepared["shade_target"].float().cpu().numpy())
            shade_prediction.append(
                output["shade_prediction"].float().cpu().numpy()
            )
            tmrt_truth.append(
                prepared["tmrt_target_celsius"].float().cpu().numpy()
            )
            tmrt_prediction.append(tmrt_celsius.float().cpu().numpy())
            progress.set_postfix(
                total=f"{float(losses.total):.4f}",
                shade=f"{float(losses.shade):.4f}",
                tmrt=f"{float(losses.tmrt):.4f}",
                gpu=f"{gpu_memory_mb():.0f}MB",
            )
    if sums["count"] == 0:
        raise RuntimeError("Validation loader is empty")
    shade_true = np.concatenate(shade_truth)
    shade_pred = np.concatenate(shade_prediction)
    tmrt_true = np.concatenate(tmrt_truth)
    tmrt_pred = np.concatenate(tmrt_prediction)
    metrics: dict[str, Any] = {
        "epoch": epoch,
        "validation_total_loss": sums["total"] / sums["count"],
        "validation_shade_loss": sums["shade"] / sums["count"],
        "validation_tmrt_loss": sums["tmrt"] / sums["count"],
        "validation_records": sums["count"],
    }
    metrics.update(regression_metrics(shade_true, shade_pred, "shade"))
    metrics.update(regression_metrics(tmrt_true, tmrt_pred, "tmrt_celsius"))
    return metrics


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_seed(
    *,
    data: FormalData,
    specs: FormalSpecs,
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    scalers_cpu: TrainingScalers,
    settings: TrainerSettings,
    model_config: dict[str, Any],
    checkpoint_dir: Path,
    metrics_dir: Path,
    config: dict[str, Any],
    config_fingerprint: str,
    input_fingerprints: dict[str, str],
    git_state: dict[str, Any],
    environment: dict[str, Any],
    resume: bool,
    logger: Any,
    status_callback: Any,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase C forbids CPU training; CUDA is unavailable")
    device = torch.device("cuda")
    seed_everything(settings.seed)
    train_loader, validation_loader = make_loaders(
        train_manifest, validation_manifest, data, specs, settings
    )
    if model_factory is None:
        model = SolarDirectionalMultitaskModel(
            len(specs.semantic_fields),
            len(specs.dino_mean_fields),
            len(specs.dynamic_fields),
            specs.window_azimuth,
            model_config,
            settings.dropout,
        ).to(device)
    else:
        model = model_factory(settings.dropout).to(device)
    criterion = MultitaskSmoothL1Loss(
        settings.lambda_shade, settings.lambda_tmrt
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5, min_lr=1.0e-7
    )
    amp_scaler = torch.amp.GradScaler(
        "cuda", enabled=settings.mixed_precision
    )
    scalers = scalers_cpu.to(device)
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    patience_count = 0
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"

    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint missing: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["seed"] != settings.seed:
            raise RuntimeError("resume seed mismatch")
        if checkpoint["config_fingerprint"] != config_fingerprint:
            raise RuntimeError("resume config fingerprint mismatch")
        if checkpoint["input_fingerprints"] != input_fingerprints:
            raise RuntimeError("resume input fingerprint mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        amp_scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        history = list(checkpoint["history"])
        validation_history = list(checkpoint["validation_history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_total_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_count = int(checkpoint["patience_count"])
        logger.info("Resuming seed %s at epoch %s", settings.seed, start_epoch)

    started = time.perf_counter()
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, settings.max_epochs + 1):
        model.train()
        sums = {"total": 0.0, "shade": 0.0, "tmrt": 0.0, "count": 0}
        report_interval = max(1, int(np.ceil(len(train_loader) * 0.05)))
        progress = tqdm(
            train_loader,
            desc=f"Seed {settings.seed} train {epoch}/{settings.max_epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress, start=1):
            prepared = prepare_batch(batch, device, scalers, True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=settings.mixed_precision):
                output = model(
                    prepared["semantic_features"],
                    prepared["dino_mean"],
                    prepared["dino_windows"],
                    prepared["dynamic_features"],
                )
                losses = criterion(
                    output["shade_prediction"],
                    prepared["shade_target"],
                    output["tmrt_standardized"],
                    prepared["tmrt_target_standardized"],
                )
            if not torch.isfinite(losses.total):
                raise FloatingPointError(
                    f"non-finite loss at seed={settings.seed}, epoch={epoch}, "
                    f"batch={batch_index}"
                )
            amp_scaler.scale(losses.total).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_clip_norm
            )
            amp_scaler.step(optimizer)
            amp_scaler.update()
            count = int(prepared["shade_target"].shape[0])
            sums["total"] += float(losses.total.detach()) * count
            sums["shade"] += float(losses.shade.detach()) * count
            sums["tmrt"] += float(losses.tmrt.detach()) * count
            sums["count"] += count
            progress.set_postfix(
                total=f"{sums['total']/sums['count']:.4f}",
                shade=f"{sums['shade']/sums['count']:.4f}",
                tmrt=f"{sums['tmrt']/sums['count']:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                gpu=f"{gpu_memory_mb():.0f}MB",
            )
            if batch_index % report_interval == 0 or batch_index == len(train_loader):
                tqdm.write(
                    f"seed={settings.seed} epoch={epoch} "
                    f"{batch_index}/{len(train_loader)} "
                    f"loss={sums['total']/sums['count']:.6f}"
                )
        validation = validate_epoch(
            model,
            validation_loader,
            device,
            scalers,
            criterion,
            settings.mixed_precision,
            epoch,
            settings.max_epochs,
        )
        scheduler.step(validation["validation_total_loss"])
        train_record = {
            "epoch": epoch,
            "train_total_loss": sums["total"] / sums["count"],
            "train_shade_loss": sums["shade"] / sums["count"],
            "train_tmrt_loss": sums["tmrt"] / sums["count"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            **validation,
        }
        history.append(train_record)
        validation_history.append(validation)
        improved = (
            validation["validation_total_loss"]
            < best_loss - settings.min_delta
        )
        if improved:
            best_loss = float(validation["validation_total_loss"])
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
        payload = {
            "checkpoint_type": "phase_c_formal",
            "seed": settings.seed,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_total_loss": best_loss,
            "patience_count": patience_count,
            "completed": False,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "amp_scaler_state_dict": amp_scaler.state_dict(),
            "feature_scalers": scalers_cpu.state_dict(),
            "settings": asdict(settings),
            "config": config,
            "config_fingerprint": config_fingerprint,
            "input_fingerprints": input_fingerprints,
            "git": git_state,
            "environment": environment,
            "history": history,
            "validation_history": validation_history,
        }
        atomic_torch_save(last_path, payload)
        if improved:
            atomic_torch_save(best_path, payload)
        atomic_csv(metrics_dir / "training_history.csv", history)
        atomic_csv(
            metrics_dir / "validation_metrics.csv", validation_history
        )
        logger.info(
            "seed=%s epoch=%s train=%.6f val=%.6f best_epoch=%s patience=%s",
            settings.seed,
            epoch,
            train_record["train_total_loss"],
            validation["validation_total_loss"],
            best_epoch,
            patience_count,
        )
        status_callback(
            {
                "seed": settings.seed,
                "epoch": epoch,
                "max_epochs": settings.max_epochs,
                "best_epoch": best_epoch,
                "best_validation_total_loss": best_loss,
                "patience_count": patience_count,
                "state": "training",
            }
        )
        if patience_count >= settings.patience:
            stop_reason = "early_stopping"
            break

    final_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    final_checkpoint["completed"] = True
    final_checkpoint["stop_reason"] = stop_reason
    atomic_torch_save(last_path, final_checkpoint)
    result = {
        "seed": settings.seed,
        "completed": True,
        "stop_reason": stop_reason,
        "epochs_completed": int(history[-1]["epoch"]),
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": model.parameter_count,
    }
    status_callback({**result, "state": "complete"})
    del model, optimizer, scheduler, amp_scaler, train_loader, validation_loader
    torch.cuda.empty_cache()
    return result
