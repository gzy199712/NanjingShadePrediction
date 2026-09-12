"""Step118: first frozen Test evaluation for the multi-date ensemble."""

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

from training.scripts.Step19_evaluate_model import (  # noqa: E402
    SEEDS,
    atomic_numpy,
    evaluation_tables,
    infer_split,
    load_frozen_models,
    logger_for,
    make_loader,
)
from training.src.data_loading import TrainingPaths, load_formal_data, load_specs, load_yaml  # noqa: E402
from training.src.logging_utils import atomic_csv, atomic_json  # noqa: E402
from training.src.metrics import regression_metrics  # noqa: E402
from training.src.reproducibility import collect_environment, collect_git_state  # noqa: E402
from training.src.trainer import prepare_batch  # noqa: E402
from training.src.utci import implementation_metadata  # noqa: E402


SCRIPT_VERSION = "1.0.0"
EXPECTED = {
    "unseen_points_seen_weather": {"rows": 174_980, "points": 1_346, "dates": 10, "space": "test", "weather": "train"},
    "seen_points_unseen_weather": {"rows": 163_332, "points": 6_282, "dates": 2, "space": "train", "weather": "test"},
    "unseen_points_unseen_weather": {"rows": 34_996, "points": 1_346, "dates": 2, "space": "test", "weather": "test"},
}


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_formal.yaml")
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def manifests_for_views(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    views: dict[str, pd.DataFrame] = {}
    for view, partition in config["evaluation"]["test_views"].items():
        frame = manifest.loc[manifest["spatiotemporal_partition"].eq(partition)].copy().reset_index(drop=True)
        expected = EXPECTED[view]
        if len(frame) != expected["rows"]:
            raise RuntimeError(f"{view} row count mismatch: {len(frame)}")
        if frame["point_id"].nunique() != expected["points"] or frame["date"].nunique() != expected["dates"]:
            raise RuntimeError(f"{view} point/date coverage mismatch")
        if set(frame["dataset_split"].astype(str).str.lower()) != {expected["space"]}:
            raise RuntimeError(f"{view} spatial split mismatch")
        if set(frame["weather_split"].astype(str).str.lower()) != {expected["weather"]}:
            raise RuntimeError(f"{view} weather split mismatch")
        views[view] = frame
    sample_sets = [set(frame["sample_id"].astype(str)) for frame in views.values()]
    if any(sample_sets[i].intersection(sample_sets[j]) for i in range(len(sample_sets)) for j in range(i + 1, len(sample_sets))):
        raise RuntimeError("Test views overlap at sample_id level")
    return views


def step117_summary() -> dict[str, Any]:
    path = PROJECT_ROOT / "training/metrics/multidate_full/formal_training_summary.json"
    if not path.exists():
        raise FileNotFoundError("Step117 formal summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or summary.get("seeds") != SEEDS or summary.get("completed_seed_count") != 5:
        raise RuntimeError("Step117 is not complete")
    if int(summary.get("test_records_used", -1)) != 0:
        raise RuntimeError("Frozen Test was accessed before Step118")
    return summary


def check_phase(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Step118 requires CUDA; CPU fallback is forbidden")
    if version("utci") != str(config["evaluation"]["utci_version"]):
        raise RuntimeError("Configured UTCI package version mismatch")
    training_summary = step117_summary()
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths); specs = load_specs(paths)
    views = manifests_for_views(data.manifest, config)
    device = torch.device("cuda:0")
    models, scalers, checkpoints = load_frozen_models(config, specs, device)
    if any(record["config_fingerprint"] != training_summary["config_fingerprint"] for record in checkpoints):
        raise RuntimeError("Checkpoint config fingerprint mismatch")
    previews: dict[str, Any] = {}
    for view, manifest in views.items():
        loader = make_loader(manifest.iloc[:512].copy(), data, specs, config)
        batch = next(iter(loader)); prepared = prepare_batch(batch, device, scalers, True)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
            outputs = {seed: model(prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"]) for seed, model in models.items()}
        finite = all(bool(torch.isfinite(output["shade_prediction"]).all() and torch.isfinite(output["tmrt_standardized"]).all()) for output in outputs.values())
        previews[view] = {"records": len(manifest), "points": int(manifest["point_id"].nunique()), "dates": sorted(manifest["date"].astype(str).unique()), "preview_finite": finite}
    report = {
        "status": "PASS" if all(value["preview_finite"] for value in previews.values()) else "FAIL",
        "ready_for_first_frozen_test_evaluation": all(value["preview_finite"] for value in previews.values()),
        "checked_at": now_text(), "gpu": torch.cuda.get_device_name(0), "seeds": SEEDS,
        "checkpoint_kind": config["evaluation"]["checkpoint_kind"], "checkpoint_checks": checkpoints,
        "test_views": previews, "total_test_records": sum(len(frame) for frame in views.values()),
        "training_performed": False, "metrics_computed": False, "predictions_saved": False,
    }
    del models, data
    torch.cuda.empty_cache()
    return report


def by_date_metrics(predictions: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view, frame in predictions.items():
        for date, group in frame.groupby("date", sort=True):
            for target, truth, prediction in (
                ("shade", "shade_true", "shade_pred_mean"),
                ("tmrt_celsius", "tmrt_true", "tmrt_pred_mean"),
                ("utci_celsius", "utci_true", "utci_pred_mean"),
            ):
                valid = np.isfinite(group[truth]) & np.isfinite(group[prediction])
                metrics = regression_metrics(group.loc[valid, truth].to_numpy(), group.loc[valid, prediction].to_numpy(), target)
                rows.append({"test_view": view, "date": str(date), "target": target, "record_count": len(group), "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()), **metrics})
    return rows


def run_evaluation(config: dict[str, Any], config_path: Path, overwrite: bool) -> dict[str, Any]:
    started_at = now_text(); started = time.perf_counter()
    evaluation = config["evaluation"]
    prediction_root = project_path(evaluation["prediction_root"])
    metrics_dir = project_path(evaluation["metrics_dir"])
    outputs = {
        "overall": metrics_dir / "test_view_overall_metrics.csv",
        "hourly": metrics_dir / "test_view_hourly_metrics.csv",
        "daily": metrics_dir / "test_view_daily_metrics.csv",
        "seed": metrics_dir / "test_view_seed_metrics.csv",
        "quantile": metrics_dir / "test_view_tmrt_quantile_metrics.csv",
        "uncertainty": metrics_dir / "test_view_ensemble_uncertainty.csv",
        "confusion": metrics_dir / "test_view_utci_stress_confusion.csv",
        "utci_qc": metrics_dir / "utci_input_qc.csv",
        "attention": project_path(evaluation["attention_path"]),
        "attention_index": project_path(evaluation["attention_index_path"]),
        "summary": metrics_dir / "evaluation_summary.json",
    }
    prediction_paths = {view: prediction_root / f"{view}_predictions.csv" for view in EXPECTED}
    existing = [str(path) for path in [*outputs.values(), *prediction_paths.values()] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step118 outputs exist; use --overwrite")
    for path in [*outputs.values(), *prediction_paths.values()]: path.parent.mkdir(parents=True, exist_ok=True)
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths); specs = load_specs(paths)
    manifests = manifests_for_views(data.manifest, config)
    device = torch.device("cuda:0")
    models, scalers, checkpoint_records = load_frozen_models(config, specs, device)
    predictions: dict[str, pd.DataFrame] = {}; attentions: dict[str, np.ndarray] = {}
    for view in tqdm(list(EXPECTED), desc="Frozen Test views", unit="view", dynamic_ncols=True):
        frame, attention = infer_split(view, manifests[view], data, specs, config, models, scalers, device)
        frame.insert(2, "date", manifests[view]["date"].astype(str).to_numpy())
        frame.insert(3, "weather_split", manifests[view]["weather_split"].astype(str).to_numpy())
        frame.insert(4, "spatiotemporal_partition", manifests[view]["spatiotemporal_partition"].astype(str).to_numpy())
        predictions[view] = frame; attentions[view] = attention
    tables = evaluation_tables(predictions, config)
    for view, frame in predictions.items(): atomic_csv(prediction_paths[view], frame.to_dict("records"))
    atomic_csv(outputs["overall"], tables["overall"])
    atomic_csv(outputs["hourly"], tables["hourly"])
    atomic_csv(outputs["daily"], by_date_metrics(predictions))
    atomic_csv(outputs["seed"], tables["seed"])
    atomic_csv(outputs["quantile"], tables["quantile"])
    atomic_csv(outputs["uncertainty"], tables["uncertainty"])
    atomic_csv(outputs["confusion"], tables["confusion"])
    utci_qc = [row for frame in predictions.values() for row in frame.attrs["utci_qc_rows"]]
    atomic_csv(outputs["utci_qc"], utci_qc)
    combined_attention = np.concatenate([attentions[view] for view in EXPECTED], axis=0)
    atomic_numpy(outputs["attention"], combined_attention)
    attention_frames: list[pd.DataFrame] = []
    for view in EXPECTED:
        index = predictions[view][["sample_id", "point_id", "date", "hour", "spatiotemporal_partition"]].copy()
        index.insert(0, "test_view", view)
        for position, azimuth in enumerate(specs.window_azimuth): index[f"attention_azimuth_{int(azimuth)}"] = attentions[view][:, position]
        attention_frames.append(index)
    atomic_csv(outputs["attention_index"], pd.concat(attention_frames, ignore_index=True).to_dict("records"))
    summary = {
        "step": 118, "status": "COMPLETE", "started_at": started_at, "ended_at": now_text(),
        "elapsed_seconds": time.perf_counter() - started, "script": SCRIPT_PATH.name,
        "script_version": SCRIPT_VERSION, "seeds": SEEDS, "checkpoint_kind": evaluation["checkpoint_kind"],
        "checkpoint_records": checkpoint_records,
        "test_view_records": {view: len(frame) for view, frame in predictions.items()},
        "total_test_records": sum(len(frame) for frame in predictions.values()),
        "training_performed": False, "checkpoint_selection_changed": False, "scaler_refit": False,
        "utci": implementation_metadata(), "utci_input_qc": utci_qc,
        "metrics": tables["overall"], "attention_shape": list(combined_attention.shape),
        "environment": collect_environment(), "git": collect_git_state(PROJECT_ROOT, 50).as_dict(),
        "outputs": {**{key: str(path) for key, path in outputs.items()}, **{f"predictions_{key}": str(path) for key, path in prediction_paths.items()}},
    }
    atomic_json(outputs["summary"], summary)
    return summary


def main() -> int:
    args = parse_args(); config_path = args.config.resolve(); config = load_yaml(config_path)
    report_dir = project_path(config["evaluation"]["report_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
    log = logger_for(PROJECT_ROOT / "training/logs/multidate_full/step118.log", args.overwrite)
    failure_path = report_dir / "step118_failure.json"
    try:
        if args.mode == "check":
            report = check_phase(config, config_path)
            output = report_dir / "step118_check.json"
            if output.exists() and not args.overwrite: raise FileExistsError("Step118 check exists; use --overwrite")
            atomic_json(output, report); log.info("Step118 check PASS"); print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            if not args.approved_by_user: raise PermissionError("--approved-by-user is required for first frozen Test evaluation")
            check_path = report_dir / "step118_check.json"
            if not check_path.exists() or not json.loads(check_path.read_text(encoding="utf-8")).get("ready_for_first_frozen_test_evaluation"):
                raise RuntimeError("Step118 check did not approve evaluation")
            summary = run_evaluation(config, config_path, args.overwrite)
            log.info("Step118 frozen Test evaluation complete")
            if failure_path.exists(): failure_path.unlink()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        atomic_json(failure_path, {"status": "FAIL", "time": now_text(), "error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc()})
        log.exception("Step118 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
