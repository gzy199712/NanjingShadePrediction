"""Step117: CUDA-only five-seed formal multi-date model training.

The original single-day checkpoints remain untouched. Model selection uses only
the frozen Validation-points x Validation-weather partition. All Test weather
and Test point partitions remain unread for evaluation until Step118.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.src.data_loading import TrainingPaths, load_formal_data, load_specs, load_yaml  # noqa: E402
from training.src.logging_utils import atomic_json  # noqa: E402
from training.src.model import SolarDirectionalMultitaskModel  # noqa: E402
from training.src.reproducibility import collect_environment, collect_git_state, seed_everything  # noqa: E402
from training.src.trainer import (  # noqa: E402
    TrainerSettings,
    fit_train_scalers,
    make_loaders,
    prepare_batch,
    save_scalers,
    train_seed,
)


SCRIPT_VERSION = "1.0.0"
EXPECTED_SEEDS = [42, 52, 62, 72, 82]


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "training" / "configs" / "multidate_formal.yaml",
    )
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    total = max(1, (path.stat().st_size + chunk_bytes - 1) // chunk_bytes)
    with path.open("rb") as handle, tqdm(total=total, desc=f"Hashing {path.name}", unit="chunk", dynamic_ncols=True, leave=False) as progress:
        while block := handle.read(chunk_bytes):
            digest.update(block)
            progress.update(1)
    return digest.hexdigest()


def setup_logger(path: Path, name: str, append: bool = False) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(); stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler); logger.addHandler(stream_handler)
    return logger


def settings_for(config: dict[str, Any], seed: int) -> TrainerSettings:
    training = config["training"]; runtime = config["runtime"]
    return TrainerSettings(
        seed=seed,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["early_stopping_patience"]),
        min_delta=float(training["early_stopping_min_delta"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        lambda_shade=float(training["lambda_shade"]),
        lambda_tmrt=float(training["lambda_tmrt"]),
        mixed_precision=bool(runtime["mixed_precision"]),
        pin_memory=bool(runtime["pin_memory"]),
        dropout=float(training["dropout"]),
    )


def split_frames(manifest: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    field = "spatiotemporal_partition"
    train_name = str(config["training"]["train_partition"])
    validation_name = str(config["training"]["model_selection_partition"])
    train = manifest.loc[manifest[field].eq(train_name)].copy()
    validation = manifest.loc[manifest[field].eq(validation_name)].copy()
    if len(train) != 816_660:
        raise RuntimeError(f"Train row count mismatch: {len(train)}")
    if len(validation) != 35_022:
        raise RuntimeError(f"Validation row count mismatch: {len(validation)}")
    if set(train["dataset_split"].astype(str).str.lower()) != {"train"} or set(train["weather_split"]) != {"train"}:
        raise RuntimeError("Train partition contains spatial or weather leakage")
    if set(validation["dataset_split"].astype(str).str.lower()) != {"validation"} or set(validation["weather_split"]) != {"validation"}:
        raise RuntimeError("Validation partition is not Validation-points x Validation-weather")
    test_names = set(str(value) for value in config["training"]["frozen_test_partitions"])
    if set(train[field]).intersection(test_names) or set(validation[field]).intersection(test_names):
        raise RuntimeError("Frozen Test partition leakage detected")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def input_hashes(paths: TrainingPaths) -> dict[str, str]:
    selected = {
        "training_manifest": paths.manifest,
        "semantic_features": paths.semantic_features,
        "dinov2_mean_features": paths.dino_mean_features,
        "dinov2_window_features": paths.dino_window_features,
        "dynamic_conditions": paths.dynamic_conditions,
        "point_hour_labels": paths.labels,
        "spatial_split_assignments": paths.spatial_split,
        "model_input_spec": paths.model_input_spec,
        "model_target_spec": paths.model_target_spec,
    }
    return {name: file_sha256(path) for name, path in tqdm(selected.items(), desc="Fingerprinting formal inputs", unit="file", dynamic_ncols=True)}


def smoke_summary() -> dict[str, Any]:
    path = PROJECT_ROOT / "training/reports/multidate_smoke/summary.json"
    if not path.exists():
        raise FileNotFoundError("Step116 summary is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or value.get("frozen_test_evaluated") is not False:
        raise RuntimeError("Step116 did not provide a clean PASS")
    return value


def check_phase(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    smoke = smoke_summary()
    if str(config["runtime"]["device"]).lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Step117 requires CUDA; CPU fallback is forbidden")
    seeds = [int(value) for value in config["training"]["seeds"]]
    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"Formal seeds must be {EXPECTED_SEEDS}")
    if int(config["training"]["batch_size"]) != int(smoke["recommended_batch_size"]):
        raise RuntimeError("Formal batch size differs from Step116 recommendation")
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths)
    specs = load_specs(paths)
    train, validation = split_frames(data.manifest, config)
    scalers_cpu, provenance = fit_train_scalers(train, data, specs)
    if provenance["train_records"] != 816_660 or provenance["train_unique_points"] != 6_282:
        raise RuntimeError("Train-only scaler provenance mismatch")
    settings = settings_for(config, seeds[0])
    train_loader, validation_loader = make_loaders(train, validation, data, specs, settings)
    seed_everything(settings.seed)
    device = torch.device("cuda:0")
    model = SolarDirectionalMultitaskModel(
        len(specs.semantic_fields), len(specs.dino_mean_fields), len(specs.dynamic_fields),
        specs.window_azimuth, config["model"], settings.dropout,
    ).to(device)
    batch = next(iter(train_loader)); prepared = prepare_batch(batch, device, scalers_cpu.to(device), True)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        output = model(prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"])
    if output["shade_prediction"].shape != prepared["shade_target"].shape:
        raise RuntimeError("Output shape mismatch")
    report = {
        "status": "PASS",
        "ready_for_formal_training": True,
        "checked_at": now_text(),
        "script_version": SCRIPT_VERSION,
        "gpu": torch.cuda.get_device_name(0),
        "seeds": seeds,
        "train_records": len(train),
        "train_points": int(train["point_id"].nunique()),
        "train_weather_dates": int(train["date"].nunique()),
        "validation_records": len(validation),
        "validation_points": int(validation["point_id"].nunique()),
        "validation_weather_dates": int(validation["date"].nunique()),
        "model_selection_partition": config["training"]["model_selection_partition"],
        "test_records_used": 0,
        "parameter_count": model.parameter_count,
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "mixed_precision": settings.mixed_precision,
        "scaler_provenance": provenance,
        "config_fingerprint": fingerprint(config),
        "input_fingerprints": input_hashes(paths),
        "no_training_performed": True,
    }
    del model, train_loader, validation_loader, data
    torch.cuda.empty_cache()
    return report


def validate_output_policy(seeds: list[int], checkpoint_root: Path, metrics_root: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    existing = [path for seed in seeds for path in (checkpoint_root / f"seed_{seed}/best.pt", checkpoint_root / f"seed_{seed}/last.pt", metrics_root / f"seed_{seed}/training_history.csv") if path.exists()]
    if existing and not (overwrite or resume):
        raise FileExistsError("Formal outputs exist; use --resume or --overwrite")
    if overwrite:
        for seed in seeds:
            for path in (checkpoint_root / f"seed_{seed}", metrics_root / f"seed_{seed}"):
                if path.exists(): shutil.rmtree(path)


def run_training(config_path: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.approved_by_user:
        raise PermissionError("--approved-by-user is required")
    check_path = PROJECT_ROOT / "training/reports/multidate_training/step117_check.json"
    if not check_path.exists():
        raise RuntimeError("Run Step117 --mode check first")
    check = json.loads(check_path.read_text(encoding="utf-8"))
    if not check.get("ready_for_formal_training") or check["config_fingerprint"] != fingerprint(config):
        raise RuntimeError("Step117 check is missing, failed, or stale")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    configured = [int(value) for value in config["training"]["seeds"]]
    seeds = args.seeds or configured
    if any(seed not in configured for seed in seeds):
        raise ValueError("Requested seed is not frozen in config")
    training = config["training"]
    checkpoint_root = project_path(training["checkpoint_root"])
    metrics_root = project_path(training["metrics_root"])
    scaler_path = project_path(training["scaler_path"])
    log_root = project_path(training["log_root"])
    validate_output_policy(seeds, checkpoint_root, metrics_root, args.overwrite, args.resume)
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths); specs = load_specs(paths)
    train, validation = split_frames(data.manifest, config)
    scalers, scaler_provenance = fit_train_scalers(train, data, specs)
    save_scalers(scaler_path, scalers, scaler_provenance)
    config_hash = fingerprint(config)
    hashes = check["input_fingerprints"]
    environment = collect_environment(); git_state = collect_git_state(PROJECT_ROOT, 50).as_dict()
    status_path = metrics_root / "formal_training_status.json"
    summary_path = metrics_root / "formal_training_summary.json"
    status: dict[str, Any] = {
        "step": 117, "state": "running", "started_at": now_text(), "updated_at": now_text(),
        "active_seed": None, "completed_seeds": [], "requested_seeds": seeds,
        "train_records": len(train), "validation_records": len(validation), "test_records_used": 0, "results": [],
    }
    atomic_json(status_path, status)
    results: list[dict[str, Any]] = []
    for seed in tqdm(seeds, desc="Formal multi-date seeds", unit="seed", dynamic_ncols=True):
        seed_checkpoint = checkpoint_root / f"seed_{seed}"
        seed_metrics = metrics_root / f"seed_{seed}"
        seed_logger = setup_logger(log_root / f"seed_{seed}/training.log", f"multidate_seed_{seed}", append=args.resume)
        last_path = seed_checkpoint / "last.pt"
        if args.resume and last_path.exists():
            saved = torch.load(last_path, map_location="cpu", weights_only=False)
            if saved.get("completed"):
                result = {
                    "seed": seed, "completed": True, "stop_reason": saved.get("stop_reason", "previously_complete"),
                    "epochs_completed": int(saved["epoch"]), "best_epoch": int(saved["best_epoch"]),
                    "best_validation_total_loss": float(saved["best_validation_total_loss"]),
                    "best_checkpoint": str(seed_checkpoint / "best.pt"), "last_checkpoint": str(last_path),
                    "elapsed_seconds": 0.0, "resumed_completed": True,
                }
                results.append(result); status["completed_seeds"].append(seed); status["results"] = results; status["updated_at"] = now_text(); atomic_json(status_path, status)
                continue
        status["active_seed"] = seed; status["updated_at"] = now_text(); atomic_json(status_path, status)

        def update_seed(seed_status: dict[str, Any]) -> None:
            status["seed_status"] = seed_status; status["updated_at"] = now_text(); atomic_json(status_path, status)

        result = train_seed(
            data=data, specs=specs, train_manifest=train, validation_manifest=validation,
            scalers_cpu=scalers, settings=settings_for(config, seed), model_config=config["model"],
            checkpoint_dir=seed_checkpoint, metrics_dir=seed_metrics, config=config,
            config_fingerprint=config_hash, input_fingerprints=hashes, git_state=git_state,
            environment=environment, resume=args.resume, logger=seed_logger, status_callback=update_seed,
        )
        results.append(result); status["completed_seeds"].append(seed); status["results"] = results; status["updated_at"] = now_text(); atomic_json(status_path, status)
    summary = {
        "step": 117, "status": "COMPLETE", "started_at": status["started_at"], "ended_at": now_text(),
        "script": SCRIPT_PATH.name, "script_version": SCRIPT_VERSION, "seeds": seeds,
        "completed_seed_count": len(results), "train_records": len(train), "validation_records": len(validation),
        "model_selection_partition": config["training"]["model_selection_partition"], "test_records_used": 0,
        "config_fingerprint": config_hash, "input_fingerprints": hashes,
        "scaler_provenance": scaler_provenance, "environment": environment, "git": git_state, "results": results,
    }
    atomic_json(summary_path, summary)
    status.update({"state": "complete", "active_seed": None, "updated_at": now_text(), "summary_path": str(summary_path)})
    atomic_json(status_path, status)
    return summary


def main() -> int:
    args = parse_args(); config_path = args.config.resolve(); config = load_yaml(config_path)
    report_dir = PROJECT_ROOT / "training/reports/multidate_training"; report_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(PROJECT_ROOT / "training/logs/multidate_full/step117.log", "step117", append=args.resume)
    try:
        if args.mode == "check":
            report = check_phase(config_path, config)
            output = report_dir / "step117_check.json"
            if output.exists() and not args.overwrite: raise FileExistsError("Step117 check exists; use --overwrite")
            atomic_json(output, report); log.info("Step117 check PASS"); print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            summary = run_training(config_path, config, args); log.info("Step117 complete with %d seeds", len(summary["results"])); print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        log.exception("Step117 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
