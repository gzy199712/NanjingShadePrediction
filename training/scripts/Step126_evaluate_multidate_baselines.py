"""Step126: one frozen evaluation of every baseline on the three Test views."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.scripts.Step125_train_multidate_baselines import normalized_inputs, resolve, rows, tree_features  # noqa: E402
from training.src.baseline_models import make_baseline_model  # noqa: E402
from training.src.data_loading import TrainingPaths, load_formal_data, load_specs, load_yaml  # noqa: E402
from training.src.logging_utils import atomic_csv, atomic_json  # noqa: E402
from training.src.metrics import regression_metrics  # noqa: E402
from training.src.utci import calculate_utci, implementation_metadata  # noqa: E402


TARGETS = {
    "shade": ("shade_true", "shade_pred"),
    "tmrt_celsius": ("tmrt_true", "tmrt_pred"),
    "utci_celsius": ("utci_true", "utci_pred"),
}
TREE_MODELS = ("weather_hgbr", "static_hgbr", "weather_xgboost", "static_xgboost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_baselines.yaml")
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--models", nargs="+", choices=TREE_MODELS)
    return parser.parse_args()


def test_views(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    output = {}
    for view, partition in config["partitions"]["test_views"].items():
        frame = manifest.loc[manifest["spatiotemporal_partition"].eq(partition)].copy().reset_index(drop=True)
        expected_space = "test" if view.startswith("unseen_points") else "train"
        expected_weather = "test" if view.endswith("unseen_weather") else "train"
        if frame.empty or set(frame["dataset_split"].str.lower()) != {expected_space} or set(frame["weather_split"].str.lower()) != {expected_weather}:
            raise RuntimeError(f"Test partition identity mismatch: {view}")
        output[view] = frame
    return output


def base_frame(manifest: pd.DataFrame, data: Any, specs: Any, view: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    label_rows = rows(manifest, "label_row")
    dynamic_rows = rows(manifest, "dynamic_condition_row")
    weather = {name: data.dynamic.iloc[dynamic_rows][name].to_numpy(float) for name in ("air_temperature", "relative_humidity", "wind_speed")}
    tmrt = data.labels.iloc[label_rows][specs.tmrt_target].to_numpy(float)
    utci, _ = calculate_utci(weather["air_temperature"], tmrt, weather["wind_speed"], weather["relative_humidity"], description=f"{view} true UTCI")
    frame = pd.DataFrame({
        "sample_id": manifest["sample_id"].astype(str), "point_id": manifest["point_id"].astype(str),
        "date": manifest["date"].astype(str), "hour": manifest["hour"].astype(int),
        "shade_true": data.labels.iloc[label_rows][specs.shade_target].to_numpy(float),
        "tmrt_true": tmrt, "utci_true": utci,
    })
    return frame, weather


def add_utci(base: pd.DataFrame, shade: np.ndarray, tmrt: np.ndarray, weather: dict[str, np.ndarray], description: str) -> pd.DataFrame:
    utci, _ = calculate_utci(weather["air_temperature"], tmrt, weather["wind_speed"], weather["relative_humidity"], description=description)
    frame = base.copy()
    frame["shade_pred"] = np.clip(shade, 0.0, 1.0)
    frame["tmrt_pred"] = tmrt
    frame["utci_pred"] = utci
    return frame


def atomic_prediction(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def metric_rows(model_id: str, seed: str, view: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for target, (truth, prediction) in TARGETS.items():
        valid = np.isfinite(frame[truth]) & np.isfinite(frame[prediction])
        values = regression_metrics(frame.loc[valid, truth].to_numpy(), frame.loc[valid, prediction].to_numpy(), "metric")
        output.append({
            "model_id": model_id, "seed": seed, "test_view": view, "target": target,
            "record_count": len(frame), "valid_count": int(valid.sum()),
            "MAE": values["metric_mae"], "RMSE": values["metric_rmse"], "R2": values["metric_r2"],
            "bias": values["metric_bias"], "Pearson_correlation": values["metric_pearson"],
        })
    return output


def load_neural(model_id: str, seed: int, data: Any, specs: Any, config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...], float, float]:
    path = resolve(config["training"]["checkpoint_root"]) / model_id / f"seed_{seed}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    scalers = {name: np.asarray(value, dtype=np.float32) for name, value in checkpoint["scalers"].items()}
    model = make_baseline_model(model_id, len(specs.semantic_fields), len(specs.dino_mean_fields), len(specs.dynamic_fields), checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    features = tuple(torch.as_tensor(value, device=device) for value in normalized_inputs(data, specs, scalers))
    return model, features, float(scalers["tmrt_mean"][0]), float(scalers["tmrt_scale"][0])


def infer_neural(model: torch.nn.Module, features: tuple[torch.Tensor, ...], manifest: pd.DataFrame, tmrt_mean: float, tmrt_scale: float, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    indices = [torch.as_tensor(rows(manifest, field), dtype=torch.long, device=features[0].device) for field in ("static_feature_row", "dino_mean_row", "dino_window_row", "dynamic_condition_row")]
    shade: list[np.ndarray] = []
    tmrt: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(manifest), batch_size):
            stop = start + batch_size
            with torch.amp.autocast("cuda", enabled=True):
                output = model(*(feature[index[start:stop]] for feature, index in zip(features, indices, strict=True)))
            shade.append(output["shade_prediction"].float().cpu().numpy())
            tmrt.append((output["tmrt_standardized"].float().cpu().numpy() * tmrt_scale + tmrt_mean))
    return np.concatenate(shade), np.concatenate(tmrt)


def save_model_predictions(model_id: str, view: str, predictions: dict[str, pd.DataFrame], root: Path, metrics: list[dict[str, Any]]) -> None:
    for seed, frame in predictions.items():
        atomic_prediction(root / model_id / view / f"seed_{seed}.csv.gz", frame)
        metrics.extend(metric_rows(model_id, seed, view, frame))
    if len(predictions) > 1:
        first = next(iter(predictions.values()))
        ensemble = first.iloc[:, :7].copy()
        for column in ("shade_pred", "tmrt_pred", "utci_pred"):
            matrix = np.column_stack([frame[column].to_numpy(float) for frame in predictions.values()])
            counts = np.isfinite(matrix).sum(axis=1)
            ensemble[column] = np.divide(np.nansum(matrix, axis=1), counts, out=np.full(len(matrix), np.nan), where=counts > 0)
        atomic_prediction(root / model_id / view / "ensemble.csv.gz", ensemble)
        metrics.extend(metric_rows(model_id, "ensemble", view, ensemble))


def evaluate_selected_trees(model_ids: list[str], views: dict[str, pd.DataFrame], data: Any, specs: Any, config: dict[str, Any], checkpoint_root: Path, prediction_root: Path, metrics_path: Path) -> dict[str, Any]:
    raw = {"weather": tree_features("weather_hgbr", data, specs)[0], "static": tree_features("static_hgbr", data, specs)[0]}
    payloads: dict[str, dict[int, Any]] = {}
    for model_id in model_ids:
        payloads[model_id] = {}
        for seed in config["training"]["seeds"]:
            with (checkpoint_root / model_id / f"seed_{seed}.pkl").open("rb") as handle:
                payload = pickle.load(handle)
            if model_id.startswith("static_"):
                semantic_count = len(specs.semantic_fields)
                source = raw["static"]
                payload["features"] = np.concatenate((source[:, :semantic_count], payload["pca"].transform(source[:, semantic_count:])), axis=1)
            payloads[model_id][int(seed)] = payload
    existing = pd.read_csv(metrics_path, dtype={"seed": str}) if metrics_path.is_file() else pd.DataFrame()
    metrics = [] if existing.empty else existing.loc[~existing["model_id"].isin(model_ids)].to_dict("records")
    for view, manifest in views.items():
        base, weather = base_frame(manifest, data, specs, view)
        for model_id in model_ids:
            key = "dynamic_condition_row" if model_id.startswith("weather_") else "static_feature_row"
            model_predictions = {}
            for seed, payload in payloads[model_id].items():
                matrix = payload.get("features", raw["weather"])
                x = matrix[rows(manifest, key)]
                models = payload["models"]
                model_predictions[str(seed)] = add_utci(base, models["shade"].predict(x), models["tmrt"].predict(x), weather, f"{view} {model_id} seed {seed} UTCI")
            save_model_predictions(model_id, view, model_predictions, prediction_root, metrics)
    atomic_csv(metrics_path, metrics)
    return {"step": 126, "status": "COMPLETE", "models": model_ids, "test_views": {name: len(frame) for name, frame in views.items()}, "metric_rows": len(metrics), "test_evaluation_runs": 1, "utci": implementation_metadata()}


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths)
    specs = load_specs(paths)
    views = test_views(data.manifest, config)
    report_root = resolve(config["outputs"]["report_root"])
    report_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = resolve(config["training"]["checkpoint_root"])
    if args.models:
        required = [checkpoint_root / model / f"seed_{seed}.pkl" for model in args.models for seed in config["training"]["seeds"]]
    else:
        required = [checkpoint_root / "mean_statistics.json"]
        required += [checkpoint_root / model / f"seed_{seed}{'.pkl' if model in TREE_MODELS else '.pt'}" for model in ("weather_hgbr", "static_hgbr", "weather_xgboost", "static_xgboost", "multimodal_mlp", "directional_mlp") for seed in config["training"]["seeds"]]
        required += [resolve(config["proposed"]["prediction_root"]) / f"{view}_predictions.csv" for view in views]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs: " + "; ".join(missing))
    if args.mode == "check":
        report = {"step": 126, "status": "PASS", "test_views": {name: len(frame) for name, frame in views.items()}, "frozen_input_count": len(required), "test_predictions_computed": False}
        atomic_json(report_root / "step126_check.json", report)
        print(json.dumps(report, indent=2))
        return 0
    if not args.approved_by_user:
        raise PermissionError("--approved-by-user is required")
    metrics_path = resolve(config["outputs"]["metrics_root"]) / "per_seed_test_metrics.csv"
    prediction_root = resolve(config["outputs"]["prediction_root"])
    if args.models:
        summary = evaluate_selected_trees(args.models, views, data, specs, config, checkpoint_root, prediction_root, metrics_path)
        atomic_json(report_root / ("step126_evaluation_summary_" + "_".join(args.models) + ".json"), summary)
        print(json.dumps(summary, indent=2))
        return 0
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError("Step126 outputs exist; use --overwrite")
    means = json.loads((checkpoint_root / "mean_statistics.json").read_text(encoding="utf-8"))
    static_features, _ = tree_features("static_hgbr", data, specs)
    weather_features, _ = tree_features("weather_hgbr", data, specs)
    tree_models: dict[str, dict[int, Any]] = {}
    for model_id in TREE_MODELS:
        tree_models[model_id] = {}
        for seed in config["training"]["seeds"]:
            with (checkpoint_root / model_id / f"seed_{seed}.pkl").open("rb") as handle:
                payload = pickle.load(handle)
                if model_id.startswith("static_"):
                    semantic_count = len(specs.semantic_fields)
                    payload["features"] = np.concatenate((static_features[:, :semantic_count], payload["pca"].transform(static_features[:, semantic_count:])), axis=1)
                tree_models[model_id][int(seed)] = payload
    device = torch.device("cuda")
    neural = {(model_id, int(seed)): load_neural(model_id, int(seed), data, specs, config, device) for model_id in ("multimodal_mlp", "directional_mlp") for seed in config["training"]["seeds"]}
    metrics: list[dict[str, Any]] = []
    for view, manifest in views.items():
        base, weather = base_frame(manifest, data, specs, view)
        global_pred = means["global"]
        save_model_predictions("mean_global", view, {"deterministic": add_utci(base, np.full(len(base), global_pred["shade"]), np.full(len(base), global_pred["tmrt"]), weather, f"{view} mean_global UTCI")}, prediction_root, metrics)
        hourly_shade = manifest["hour"].astype(str).map({key: value["shade"] for key, value in means["hourly"].items()}).to_numpy(float)
        hourly_tmrt = manifest["hour"].astype(str).map({key: value["tmrt"] for key, value in means["hourly"].items()}).to_numpy(float)
        save_model_predictions("mean_hourly", view, {"deterministic": add_utci(base, hourly_shade, hourly_tmrt, weather, f"{view} mean_hourly UTCI")}, prediction_root, metrics)
        for model_id, feature_matrix, key in (("weather_hgbr", weather_features, "dynamic_condition_row"), ("static_hgbr", static_features, "static_feature_row"), ("weather_xgboost", weather_features, "dynamic_condition_row"), ("static_xgboost", static_features, "static_feature_row")):
            model_predictions = {}
            for seed, payload in tree_models[model_id].items():
                models = payload["models"]
                matrix = payload.get("features", feature_matrix)
                x = matrix[rows(manifest, key)]
                model_predictions[str(seed)] = add_utci(base, models["shade"].predict(x), models["tmrt"].predict(x), weather, f"{view} {model_id} seed {seed} UTCI")
            save_model_predictions(model_id, view, model_predictions, prediction_root, metrics)
        for model_id in ("multimodal_mlp", "directional_mlp"):
            model_predictions = {}
            for seed in config["training"]["seeds"]:
                model, features, tmrt_mean, tmrt_scale = neural[(model_id, int(seed))]
                shade, tmrt = infer_neural(model, features, manifest, tmrt_mean, tmrt_scale, int(config["training"]["inference_batch_size"]))
                model_predictions[str(seed)] = add_utci(base, shade, tmrt, weather, f"{view} {model_id} seed {seed} UTCI")
            save_model_predictions(model_id, view, model_predictions, prediction_root, metrics)
        proposed_source = pd.read_csv(resolve(config["proposed"]["prediction_root"]) / f"{view}_predictions.csv", dtype={"sample_id": str, "point_id": str, "date": str})
        if not np.array_equal(base["sample_id"].to_numpy(), proposed_source["sample_id"].to_numpy()):
            raise RuntimeError(f"Proposed sample alignment failed: {view}")
        proposed_predictions = {str(seed): add_utci(base, proposed_source[f"shade_pred_seed_{seed}"].to_numpy(float), proposed_source[f"tmrt_pred_seed_{seed}"].to_numpy(float), weather, f"{view} proposed seed {seed} UTCI") for seed in config["proposed"]["seeds"]}
        save_model_predictions("proposed", view, proposed_predictions, prediction_root, metrics)
    atomic_csv(metrics_path, metrics)
    summary = {"step": 126, "status": "COMPLETE", "test_views": {name: len(frame) for name, frame in views.items()}, "metric_rows": len(metrics), "test_evaluation_runs": 1, "utci": implementation_metadata(), "prediction_root": str(prediction_root), "metrics_path": str(metrics_path)}
    atomic_json(report_root / "step126_evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
