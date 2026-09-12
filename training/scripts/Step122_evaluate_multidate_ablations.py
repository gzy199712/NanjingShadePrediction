"""Step122: frozen three-seed Test evaluation of formal ablations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.scripts.Step118_evaluate_multidate_test import EXPECTED, manifests_for_views  # noqa: E402
from training.scripts.Step19_evaluate_model import make_loader  # noqa: E402
from training.src.ablation_model import AblationSolarDirectionalMultitaskModel, AblationSpec  # noqa: E402
from training.src.data_loading import TrainingPaths, load_formal_data, load_specs, load_yaml  # noqa: E402
from training.src.logging_utils import atomic_csv, atomic_json  # noqa: E402
from training.src.metrics import regression_metrics  # noqa: E402
from training.src.trainer import TrainingScalers, prepare_batch  # noqa: E402
from training.src.utci import calculate_utci, implementation_metadata  # noqa: E402


SCRIPT_VERSION = "1.0.0"
SEEDS = [42, 52, 62]


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_ablation_evaluation.yaml")
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def variants(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(group, item) for group in ("visual_semantic", "weather", "fusion_multitask") for item in config["groups"][group]]


def spec_for(item: dict[str, Any]) -> AblationSpec:
    return AblationSpec(
        variant_id=str(item["id"]), zero_dynamic_fields=tuple(item.get("zero_dynamic_fields", [])),
        use_global_dinov2=bool(item.get("use_global_dinov2", True)),
        use_directional_dinov2=bool(item.get("use_directional_dinov2", True)),
        use_semantic_structure=bool(item.get("use_semantic_structure", True)),
        use_cross_attention=bool(item.get("use_cross_attention", True)),
        use_azimuth_encoding=bool(item.get("use_azimuth_encoding", True)),
        connect_shade_to_tmrt=bool(item.get("connect_shade_to_tmrt", True)),
    )


def step121_summary() -> dict[str, Any]:
    path = PROJECT_ROOT / "training/metrics/multidate_ablation/formal_training_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or int(summary.get("completed_run_count", -1)) != 30:
        raise RuntimeError("Step121 formal training is incomplete")
    if int(summary.get("test_records_used", -1)) != 0:
        raise RuntimeError("Step121 accessed frozen Test")
    return summary


def checkpoint_path(config: dict[str, Any], variant_id: str, seed: int) -> Path:
    return project_path(config["training"]["checkpoint_root"]) / variant_id / f"seed_{seed}" / f"{config['evaluation']['checkpoint_kind']}.pt"


def load_models(config: dict[str, Any], specs: Any, item: dict[str, Any], device: torch.device) -> tuple[dict[int, torch.nn.Module], TrainingScalers, list[dict[str, Any]]]:
    model_spec = spec_for(item)
    models: dict[int, torch.nn.Module] = {}
    scaler_state = None
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = checkpoint_path(config, model_spec.variant_id, seed)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if int(checkpoint["seed"]) != seed or not checkpoint.get("completed", False) and path.name == "last.pt":
            raise RuntimeError(f"Invalid frozen checkpoint: {path}")
        if checkpoint["config"].get("active_ablation", {}).get("id") != model_spec.variant_id:
            raise RuntimeError(f"Ablation identity mismatch: {path}")
        if scaler_state is None:
            scaler_state = checkpoint["feature_scalers"]
        elif scaler_state != checkpoint["feature_scalers"]:
            raise RuntimeError("Scaler mismatch across ablation seeds")
        model = AblationSolarDirectionalMultitaskModel(
            len(specs.semantic_fields), len(specs.dino_mean_fields), specs.dynamic_fields,
            specs.window_azimuth, config["model"], float(config["training"]["dropout"]), model_spec,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        models[seed] = model
        records.append({
            "variant_id": model_spec.variant_id, "seed": seed, "checkpoint": str(path),
            "checkpoint_epoch": int(checkpoint["epoch"]), "best_epoch": int(checkpoint["best_epoch"]),
            "best_validation_total_loss": float(checkpoint["best_validation_total_loss"]), "strict_load": True,
        })
    if scaler_state is None:
        raise RuntimeError("No scaler in checkpoints")
    return models, TrainingScalers.from_state_dict(scaler_state).to(device), records


def check_phase(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available() or str(config["runtime"]["device"]).lower() != "cuda":
        raise RuntimeError("Step122 requires CUDA; CPU fallback is forbidden")
    if version("utci") != str(config["evaluation"]["utci_version"]):
        raise RuntimeError("UTCI package version mismatch")
    step121_summary()
    matrix = variants(config)
    if len(matrix) != 10 or len({item["id"] for _, item in matrix}) != 10:
        raise RuntimeError("Expected 10 unique ablations")
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths); specs = load_specs(paths)
    validation = data.manifest.loc[data.manifest["spatiotemporal_partition"].eq("validation_unseen_points_unseen_weather")].iloc[:512].copy()
    device = torch.device("cuda:0")
    checkpoint_records: list[dict[str, Any]] = []
    finite: dict[str, bool] = {}
    for _, item in tqdm(matrix, desc="Checking frozen ablations", unit="variant", dynamic_ncols=True):
        models, scalers, records = load_models(config, specs, item, device)
        checkpoint_records.extend(records)
        batch = next(iter(make_loader(validation, data, specs, config)))
        prepared = prepare_batch(batch, device, scalers, True)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
            output = models[SEEDS[0]](prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"])
        finite[item["id"]] = bool(torch.isfinite(output["shade_prediction"]).all() and torch.isfinite(output["tmrt_standardized"]).all())
        del models, scalers
        torch.cuda.empty_cache()
    return {
        "step": 122, "status": "PASS" if all(finite.values()) else "FAIL", "checked_at": now_text(),
        "ready_for_frozen_test_evaluation": all(finite.values()), "gpu": torch.cuda.get_device_name(0),
        "variant_count": len(matrix), "checkpoint_count": len(checkpoint_records), "seeds": SEEDS,
        "checkpoint_records": checkpoint_records, "validation_preview_finite": finite,
        "test_predictions_computed": False, "training_performed": False, "checkpoint_selection_changed": False,
    }


def infer_variant(view: str, manifest: pd.DataFrame, data: Any, specs: Any, config: dict[str, Any], models: dict[int, torch.nn.Module], scalers: TrainingScalers, device: torch.device) -> pd.DataFrame:
    loader = make_loader(manifest, data, specs, config)
    shade: dict[int, list[np.ndarray]] = {seed: [] for seed in SEEDS}
    tmrt: dict[int, list[np.ndarray]] = {seed: [] for seed in SEEDS}
    truth_shade: list[np.ndarray] = []; truth_tmrt: list[np.ndarray] = []; truth_utci: list[np.ndarray] = []
    air: list[np.ndarray] = []; humidity: list[np.ndarray] = []; wind: list[np.ndarray] = []
    dynamic_positions = {field: specs.dynamic_fields.index(field) for field in ("air_temperature", "relative_humidity", "wind_speed")}
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{view} inference", unit="batch", dynamic_ncols=True, leave=False):
            prepared = prepare_batch(batch, device, scalers, True)
            for seed, model in models.items():
                with torch.amp.autocast("cuda", enabled=True):
                    output = model(prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"])
                shade[seed].append(output["shade_prediction"].float().cpu().numpy())
                tmrt[seed].append((output["tmrt_standardized"] * scalers.tmrt_scale + scalers.tmrt_mean).float().cpu().numpy())
            truth_shade.append(batch["shade_target"].numpy()); truth_tmrt.append(batch["tmrt_target"].numpy()); truth_utci.append(batch["utci_target"].numpy())
            dynamic = batch["dynamic_features"].numpy()
            air.append(dynamic[:, dynamic_positions["air_temperature"]]); humidity.append(dynamic[:, dynamic_positions["relative_humidity"]]); wind.append(dynamic[:, dynamic_positions["wind_speed"]])
    frame = pd.DataFrame({
        "sample_id": manifest["sample_id"].astype(str).to_numpy(), "point_id": manifest["point_id"].astype(str).to_numpy(),
        "date": manifest["date"].astype(str).to_numpy(), "hour": manifest["hour"].astype(int).to_numpy(),
        "shade_true": np.concatenate(truth_shade), "tmrt_true": np.concatenate(truth_tmrt), "utci_true": np.concatenate(truth_utci),
        "air_temperature": np.concatenate(air), "relative_humidity": np.concatenate(humidity), "wind_speed": np.concatenate(wind),
    })
    shade_matrix = np.column_stack([np.concatenate(shade[seed]) for seed in SEEDS])
    tmrt_matrix = np.column_stack([np.concatenate(tmrt[seed]) for seed in SEEDS])
    utci_matrix = np.empty_like(tmrt_matrix, dtype=np.float64)
    for index, seed in enumerate(SEEDS):
        frame[f"shade_pred_seed_{seed}"] = shade_matrix[:, index]
        frame[f"tmrt_pred_seed_{seed}"] = tmrt_matrix[:, index]
        utci_prediction, _ = calculate_utci(
            frame["air_temperature"].to_numpy(), tmrt_matrix[:, index], frame["wind_speed"].to_numpy(), frame["relative_humidity"].to_numpy(),
            description=f"{view} UTCI seed {seed}",
        )
        utci_matrix[:, index] = utci_prediction
        frame[f"utci_pred_seed_{seed}"] = utci_prediction
    frame["shade_pred_mean"] = shade_matrix.mean(axis=1)
    frame["tmrt_pred_mean"] = tmrt_matrix.mean(axis=1)
    frame["utci_pred_mean"] = np.nanmean(utci_matrix, axis=1)
    return frame


def baseline_frame(view: str) -> pd.DataFrame:
    path = PROJECT_ROOT / f"training/predictions/multidate_test/{view}_predictions.csv"
    columns = ["sample_id", "point_id", "date", "hour", "shade_true", "tmrt_true", "utci_true"]
    for target in ("shade", "tmrt", "utci"):
        columns.extend([f"{target}_pred_seed_{seed}" for seed in SEEDS])
    frame = pd.read_csv(path, usecols=columns, dtype={"sample_id": str, "point_id": str, "date": str})
    for target in ("shade", "tmrt", "utci"):
        frame[f"{target}_pred_mean"] = frame[[f"{target}_pred_seed_{seed}" for seed in SEEDS]].mean(axis=1)
    return frame


def metric_rows(variant_id: str, group: str, view: str, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall: list[dict[str, Any]] = []; seeds: list[dict[str, Any]] = []; daily: list[dict[str, Any]] = []; hourly: list[dict[str, Any]] = []
    for target, truth, prediction in (("shade", "shade_true", "shade_pred_mean"), ("tmrt_celsius", "tmrt_true", "tmrt_pred_mean"), ("utci_celsius", "utci_true", "utci_pred_mean")):
        valid = np.isfinite(frame[truth]) & np.isfinite(frame[prediction])
        overall.append({"variant_id": variant_id, "group": group, "test_view": view, "target": target, "record_count": len(frame), "valid_count": int(valid.sum()), **regression_metrics(frame.loc[valid, truth].to_numpy(), frame.loc[valid, prediction].to_numpy(), target)})
        stem = target.split("_")[0]
        for seed in SEEDS:
            column = f"{stem}_pred_seed_{seed}"; valid_seed = np.isfinite(frame[truth]) & np.isfinite(frame[column])
            seeds.append({"variant_id": variant_id, "group": group, "test_view": view, "seed": seed, "target": target, "valid_count": int(valid_seed.sum()), **regression_metrics(frame.loc[valid_seed, truth].to_numpy(), frame.loc[valid_seed, column].to_numpy(), target)})
        for date, subset in frame.groupby("date", sort=True):
            keep = np.isfinite(subset[truth]) & np.isfinite(subset[prediction])
            daily.append({"variant_id": variant_id, "group": group, "test_view": view, "date": str(date), "target": target, "record_count": len(subset), **regression_metrics(subset.loc[keep, truth].to_numpy(), subset.loc[keep, prediction].to_numpy(), target)})
        for hour, subset in frame.groupby("hour", sort=True):
            keep = np.isfinite(subset[truth]) & np.isfinite(subset[prediction])
            hourly.append({"variant_id": variant_id, "group": group, "test_view": view, "hour": int(hour), "target": target, "record_count": len(subset), **regression_metrics(subset.loc[keep, truth].to_numpy(), subset.loc[keep, prediction].to_numpy(), target)})
    return overall, seeds, daily, hourly


def atomic_npz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrays = {column: frame[column].to_numpy() for column in frame.columns}
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def degradation_table(overall: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = overall.loc[overall["variant_id"].eq("full_model_3seed")]
    for _, row in overall.loc[~overall["variant_id"].eq("full_model_3seed")].iterrows():
        match = baseline.loc[(baseline["test_view"] == row["test_view"]) & (baseline["target"] == row["target"])].iloc[0]
        prefix = str(row["target"])
        for metric in ("mae", "rmse", "r2"):
            field = f"{prefix}_{metric}"; base_value = float(match[field]); ablated = float(row[field])
            degradation = ablated - base_value if metric in ("mae", "rmse") else base_value - ablated
            rows.append({
                "variant_id": row["variant_id"], "group": row["group"], "test_view": row["test_view"], "target": row["target"],
                "metric": metric, "baseline_value": base_value, "ablation_value": ablated,
                "degradation_positive_is_worse": degradation,
                "relative_degradation_percent": degradation / abs(base_value) * 100.0 if base_value != 0 else np.nan,
            })
    return rows


def run_evaluation(config: dict[str, Any], config_path: Path, overwrite: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    started = time.perf_counter(); started_at = now_text()
    metrics_dir = project_path(config["evaluation"]["metrics_dir"]); prediction_dir = project_path(config["evaluation"]["prediction_dir"])
    summary_path = metrics_dir / "evaluation_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError("Step122 outputs exist; use --overwrite")
    paths = TrainingPaths.from_config(config_path, config); data = load_formal_data(paths); specs = load_specs(paths)
    views = manifests_for_views(data.manifest, config)
    all_overall: list[dict[str, Any]] = []; all_seeds: list[dict[str, Any]] = []; all_daily: list[dict[str, Any]] = []; all_hourly: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    for view in tqdm(list(views), desc="Three-seed baseline views", unit="view", dynamic_ncols=True):
        frame = baseline_frame(view)
        tables = metric_rows("full_model_3seed", "baseline", view, frame)
        for destination, source in zip((all_overall, all_seeds, all_daily, all_hourly), tables, strict=True): destination.extend(source)
        atomic_npz(prediction_dir / "full_model_3seed" / f"{view}.npz", frame)
    device = torch.device("cuda:0")
    for group, item in tqdm(variants(config), desc="Frozen ablation evaluation", unit="variant", dynamic_ncols=True):
        models, scalers, records = load_models(config, specs, item, device); checkpoint_records.extend(records)
        for view, manifest in views.items():
            frame = infer_variant(view, manifest, data, specs, config, models, scalers, device)
            tables = metric_rows(item["id"], group, view, frame)
            for destination, source in zip((all_overall, all_seeds, all_daily, all_hourly), tables, strict=True): destination.extend(source)
            atomic_npz(prediction_dir / item["id"] / f"{view}.npz", frame)
        del models, scalers
        torch.cuda.empty_cache()
    overall_frame = pd.DataFrame(all_overall)
    atomic_csv(metrics_dir / "overall_metrics.csv", all_overall)
    atomic_csv(metrics_dir / "seed_metrics.csv", all_seeds)
    atomic_csv(metrics_dir / "daily_metrics.csv", all_daily)
    atomic_csv(metrics_dir / "hourly_metrics.csv", all_hourly)
    deltas = degradation_table(overall_frame)
    atomic_csv(metrics_dir / "degradation_vs_full_model.csv", deltas)
    atomic_csv(metrics_dir / "checkpoint_registry.csv", checkpoint_records)
    strict = config["evaluation"]["strict_view"]
    strict_rows = [row for row in deltas if row["test_view"] == strict]
    ranking = sorted(strict_rows, key=lambda row: float(row["relative_degradation_percent"]), reverse=True)
    atomic_csv(metrics_dir / "strict_test_degradation_ranking.csv", ranking)
    summary = {
        "step": 122, "status": "COMPLETE", "started_at": started_at, "ended_at": now_text(), "elapsed_seconds": time.perf_counter() - started,
        "script": SCRIPT_PATH.name, "script_version": SCRIPT_VERSION, "variant_count": 10, "baseline": "full_model_3seed",
        "seeds": SEEDS, "test_views": {view: len(frame) for view, frame in views.items()}, "checkpoint_count": len(checkpoint_records),
        "training_performed": False, "checkpoint_selection_changed": False, "scaler_refit": False,
        "utci": implementation_metadata(), "metrics_dir": str(metrics_dir), "prediction_dir": str(prediction_dir),
    }
    atomic_json(summary_path, summary)
    return summary


def main() -> int:
    args = parse_args(); config_path = args.config.resolve(); config = load_yaml(config_path)
    report_dir = project_path(config["evaluation"]["report_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
    from training.scripts.Step19_evaluate_model import logger_for
    log = logger_for(project_path(config["evaluation"]["log_path"]), args.overwrite)
    try:
        if args.mode == "check":
            report = check_phase(config, config_path)
            output = report_dir / "step122_check.json"
            if output.exists() and not args.overwrite: raise FileExistsError("Step122 check exists; use --overwrite")
            atomic_json(output, report); log.info("Step122 check PASS"); print(json.dumps(report, ensure_ascii=True, indent=2))
        else:
            if not args.approved_by_user: raise PermissionError("--approved-by-user is required")
            check_path = report_dir / "step122_check.json"
            if not check_path.exists() or not json.loads(check_path.read_text(encoding="utf-8")).get("ready_for_frozen_test_evaluation"):
                raise RuntimeError("Step122 check is missing or failed")
            summary = run_evaluation(config, config_path, args.overwrite)
            log.info("Step122 complete"); print(json.dumps(summary, ensure_ascii=True, indent=2))
        return 0
    except Exception as error:
        atomic_json(report_dir / "step122_failure.json", {"status": "FAIL", "time": now_text(), "error": str(error), "traceback": traceback.format_exc()})
        log.exception("Step122 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
