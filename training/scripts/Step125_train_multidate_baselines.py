"""Step125: train strong baselines using Train and Validation only."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.src.baseline_models import make_baseline_model  # noqa: E402
from training.src.data_loading import TrainingPaths, load_formal_data, load_specs, load_yaml  # noqa: E402
from training.src.logging_utils import atomic_json  # noqa: E402
from training.src.reproducibility import seed_everything  # noqa: E402
from training.src.trainer import atomic_torch_save  # noqa: E402


NEURAL_MODELS = ("multimodal_mlp", "directional_mlp")
TREE_MODELS = ("weather_hgbr", "static_hgbr", "weather_xgboost", "static_xgboost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_baselines.yaml")
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--models", nargs="+", choices=TREE_MODELS + NEURAL_MODELS)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def partitions(manifest: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    field = "spatiotemporal_partition"
    train = manifest.loc[manifest[field].eq(config["partitions"]["train"])].copy()
    validation = manifest.loc[manifest[field].eq(config["partitions"]["validation"])].copy()
    if train.empty or validation.empty:
        raise RuntimeError("Train or validation partition is empty")
    if set(train["dataset_split"].str.lower()) != {"train"} or set(train["weather_split"].str.lower()) != {"train"}:
        raise RuntimeError("Train partition identity mismatch")
    if set(validation["dataset_split"].str.lower()) != {"validation"} or set(validation["weather_split"].str.lower()) != {"validation"}:
        raise RuntimeError("Validation partition identity mismatch")
    return train, validation


def rows(frame: pd.DataFrame, field: str) -> np.ndarray:
    return frame[field].to_numpy(dtype=np.int64, copy=True)


def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1.0
    return mean, scale


def fit_scalers(train: pd.DataFrame, data: Any, specs: Any) -> dict[str, np.ndarray]:
    static_rows = np.unique(rows(train, "static_feature_row"))
    dynamic_rows = rows(train, "dynamic_condition_row")
    label_rows = rows(train, "label_row")
    semantic = data.semantic.iloc[static_rows][specs.semantic_fields].to_numpy(np.float32)
    global_dino = data.dino_mean.iloc[static_rows][specs.dino_mean_fields].to_numpy(np.float32)
    directional = np.asarray(data.dino_window[static_rows], dtype=np.float32).reshape(-1, data.dino_window.shape[-1])
    dynamic = data.dynamic.iloc[dynamic_rows][specs.dynamic_fields].to_numpy(np.float32)
    tmrt = data.labels.iloc[label_rows][specs.tmrt_target].to_numpy(np.float32)
    output: dict[str, np.ndarray] = {}
    for name, values in (("semantic", semantic), ("global", global_dino), ("directional", directional), ("dynamic", dynamic)):
        output[f"{name}_mean"], output[f"{name}_scale"] = moments(values)
    output["tmrt_mean"] = np.asarray([tmrt.mean(dtype=np.float64)], dtype=np.float32)
    output["tmrt_scale"] = np.asarray([tmrt.std(dtype=np.float64) or 1.0], dtype=np.float32)
    return output


def normalized_inputs(data: Any, specs: Any, scalers: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    semantic = data.semantic[specs.semantic_fields].to_numpy(np.float32)
    global_dino = data.dino_mean[specs.dino_mean_fields].to_numpy(np.float32)
    directional = np.asarray(data.dino_window, dtype=np.float32)
    dynamic = data.dynamic[specs.dynamic_fields].to_numpy(np.float32)
    return tuple(
        (values - scalers[f"{name}_mean"]) / scalers[f"{name}_scale"]
        for name, values in (("semantic", semantic), ("global", global_dino), ("directional", directional), ("dynamic", dynamic))
    )


def tensor_split(frame: pd.DataFrame, data: Any, specs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    label_rows = rows(frame, "label_row")
    return {
        "static": torch.as_tensor(rows(frame, "static_feature_row"), dtype=torch.long, device=device),
        "global": torch.as_tensor(rows(frame, "dino_mean_row"), dtype=torch.long, device=device),
        "directional": torch.as_tensor(rows(frame, "dino_window_row"), dtype=torch.long, device=device),
        "dynamic": torch.as_tensor(rows(frame, "dynamic_condition_row"), dtype=torch.long, device=device),
        "shade": torch.as_tensor(data.labels.iloc[label_rows][specs.shade_target].to_numpy(np.float32), device=device),
        "tmrt": torch.as_tensor(data.labels.iloc[label_rows][specs.tmrt_target].to_numpy(np.float32), device=device),
    }


def predict_neural(model: torch.nn.Module, split: dict[str, torch.Tensor], features: tuple[torch.Tensor, ...], batch_size: int, tmrt_mean: torch.Tensor, tmrt_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()
    shade_parts: list[torch.Tensor] = []
    tmrt_parts: list[torch.Tensor] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(split["shade"]), batch_size):
            stop = start + batch_size
            output = model(features[0][split["static"][start:stop]], features[1][split["global"][start:stop]], features[2][split["directional"][start:stop]], features[3][split["dynamic"][start:stop]])
            truth_tmrt = (split["tmrt"][start:stop] - tmrt_mean) / tmrt_scale
            loss = torch.nn.functional.smooth_l1_loss(output["shade_prediction"], split["shade"][start:stop], reduction="sum")
            loss += torch.nn.functional.smooth_l1_loss(output["tmrt_standardized"], truth_tmrt, reduction="sum")
            loss_sum += float(loss)
            shade_parts.append(output["shade_prediction"])
            tmrt_parts.append(output["tmrt_standardized"])
    return torch.cat(shade_parts), torch.cat(tmrt_parts), loss_sum / len(split["shade"])


def train_neural(model_id: str, seed: int, train: pd.DataFrame, validation: pd.DataFrame, data: Any, specs: Any, scalers: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    settings = config["training"]
    seed_everything(seed)
    device = torch.device("cuda")
    arrays = normalized_inputs(data, specs, scalers)
    features = tuple(torch.as_tensor(value, device=device) for value in arrays)
    train_data = tensor_split(train, data, specs, device)
    validation_data = tensor_split(validation, data, specs, device)
    model_config = {**settings, "dropout": settings["dropout"]}
    model = make_baseline_model(model_id, len(specs.semantic_fields), len(specs.dino_mean_fields), len(specs.dynamic_fields), model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    batch_size = int(settings["batch_size"])
    tmrt_mean = torch.as_tensor(scalers["tmrt_mean"], device=device)
    tmrt_scale = torch.as_tensor(scalers["tmrt_scale"], device=device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_data["shade"]), generator=generator, device=device)
        train_loss = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                output = model(features[0][train_data["static"][index]], features[1][train_data["global"][index]], features[2][train_data["directional"][index]], features[3][train_data["dynamic"][index]])
                tmrt_truth = (train_data["tmrt"][index] - tmrt_mean) / tmrt_scale
                loss = torch.nn.functional.smooth_l1_loss(output["shade_prediction"], train_data["shade"][index]) + torch.nn.functional.smooth_l1_loss(output["tmrt_standardized"], tmrt_truth)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(index)
        _, _, validation_loss = predict_neural(model, validation_data, features, int(settings["inference_batch_size"]), tmrt_mean, tmrt_scale)
        history.append({"epoch": epoch, "train_loss": train_loss / len(order), "validation_loss": validation_loss})
        print(f"{model_id} seed={seed} epoch={epoch} train={history[-1]['train_loss']:.6f} validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, patience = validation_loss, epoch, 0
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Neural baseline produced no checkpoint")
    checkpoint = {
        "model_id": model_id, "seed": seed, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "model_state_dict": best_state,
        "scalers": {name: value.tolist() for name, value in scalers.items()},
        "model_config": model_config, "history": history,
        "train_records": len(train), "validation_records": len(validation), "test_records_used": 0,
    }
    path = resolve(settings["checkpoint_root"]) / model_id / f"seed_{seed}.pt"
    atomic_torch_save(path, checkpoint)
    del model, features, train_data, validation_data
    torch.cuda.empty_cache()
    return {"model_id": model_id, "seed": seed, "checkpoint": str(path), "best_epoch": best_epoch, "best_validation_loss": best_loss}


def aggregate_target(frame: pd.DataFrame, key: str, target: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = rows(frame, key)
    counts = np.bincount(keys, minlength=size).astype(np.float64)
    sums = np.bincount(keys, weights=target[rows(frame, "label_row")], minlength=size)
    keep = counts > 0
    return np.flatnonzero(keep), sums[keep] / counts[keep], counts[keep]


def tree_features(model_id: str, data: Any, specs: Any) -> tuple[np.ndarray, str]:
    if model_id.startswith("weather_"):
        return data.dynamic[specs.dynamic_fields].to_numpy(np.float32), "dynamic_condition_row"
    directional = np.asarray(data.dino_window, dtype=np.float32)
    pooled = np.concatenate((directional.mean(1), directional.std(1)), axis=1)
    return np.concatenate((data.semantic[specs.semantic_fields].to_numpy(np.float32), data.dino_mean[specs.dino_mean_fields].to_numpy(np.float32), pooled), axis=1), "static_feature_row"


def train_trees(model_id: str, seed: int, train: pd.DataFrame, validation: pd.DataFrame, data: Any, specs: Any, config: dict[str, Any]) -> dict[str, Any]:
    settings = config["training"]
    features, key = tree_features(model_id, data, specs)
    pca = None
    if model_id.startswith("static_"):
        train_points = np.unique(rows(train, key))
        semantic_count = len(specs.semantic_fields)
        pca = PCA(n_components=int(settings["static_pca_components"]), svd_solver="randomized", random_state=seed)
        pca.fit(features[train_points, semantic_count:])
        features = np.concatenate((features[:, :semantic_count], pca.transform(features[:, semantic_count:])), axis=1)
    targets = {"shade": data.labels[specs.shade_target].to_numpy(float), "tmrt": data.labels[specs.tmrt_target].to_numpy(float)}
    models: dict[str, HistGradientBoostingRegressor] = {}
    validation_rmse: dict[str, float] = {}
    for name, target in targets.items():
        indices, averages, weights = aggregate_target(train, key, target, len(features))
        if model_id.endswith("xgboost"):
            validation_indices, validation_averages, validation_weights = aggregate_target(validation, key, target, len(features))
            model = XGBRegressor(
                n_estimators=int(settings["xgboost_n_estimators"]), max_depth=int(settings["xgboost_max_depth"]),
                learning_rate=float(settings["tree_learning_rate"]), subsample=0.9, colsample_bytree=0.9,
                random_state=seed, tree_method="hist", early_stopping_rounds=int(settings["xgboost_early_stopping_rounds"]),
            )
            model.fit(features[indices], averages, sample_weight=weights, eval_set=[(features[validation_indices], validation_averages)], sample_weight_eval_set=[validation_weights], verbose=False)
        else:
            model = HistGradientBoostingRegressor(
                max_iter=int(settings["tree_max_iter"]), learning_rate=float(settings["tree_learning_rate"]),
                max_leaf_nodes=int(settings["tree_max_leaf_nodes"]), early_stopping=False, loss="squared_error",
            )
            model.fit(features[indices], averages, sample_weight=weights)
        prediction = model.predict(features[rows(validation, key)])
        truth = target[rows(validation, "label_row")]
        validation_rmse[name] = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        models[name] = model
    path = resolve(settings["checkpoint_root"]) / model_id / f"seed_{seed}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        algorithm = "xgboost.XGBRegressor" if model_id.endswith("xgboost") else settings["tree_algorithm"]
        pickle.dump({"model_id": model_id, "seed": seed, "models": models, "pca": pca, "validation_rmse": validation_rmse, "algorithm": algorithm, "test_records_used": 0}, handle)
    randomness = "randomized train-only PCA" if pca is not None else ("seeded row/column subsampling" if model_id.endswith("xgboost") else "deterministic HGBR; seed registry retained for fair three-run accounting")
    return {"model_id": model_id, "seed": seed, "checkpoint": str(path), "validation_rmse": validation_rmse, "algorithm_randomness": randomness, "static_pca_components": int(settings["static_pca_components"]) if pca is not None else None}


def train_means(train: pd.DataFrame, data: Any, specs: Any, config: dict[str, Any]) -> dict[str, Any]:
    label_rows = rows(train, "label_row")
    values = data.labels.iloc[label_rows][[specs.shade_target, specs.tmrt_target]].copy()
    values["hour"] = train["hour"].to_numpy()
    output = {
        "global": {"shade": float(values[specs.shade_target].mean()), "tmrt": float(values[specs.tmrt_target].mean())},
        "hourly": {str(int(hour)): {"shade": float(group[specs.shade_target].mean()), "tmrt": float(group[specs.tmrt_target].mean())} for hour, group in values.groupby("hour")},
        "fit_split": "train", "train_records": len(train), "deterministic": True,
    }
    atomic_json(resolve(config["training"]["checkpoint_root"]) / "mean_statistics.json", output)
    return output


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths)
    specs = load_specs(paths)
    train, validation = partitions(data.manifest, config)
    report_root = resolve(config["outputs"]["report_root"])
    report_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "check":
        report = {"step": 125, "status": "PASS", "train_records": len(train), "validation_records": len(validation), "test_records_used": 0, "seeds": config["training"]["seeds"], "models": args.models or list(TREE_MODELS + NEURAL_MODELS), "tree_algorithms": [config["training"]["tree_algorithm"], "xgboost.XGBRegressor 3.2.0"], "cuda": torch.cuda.is_available()}
        atomic_json(report_root / "step125_check.json", report)
        print(json.dumps(report, indent=2))
        return 0
    if not args.approved_by_user:
        raise PermissionError("--approved-by-user is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural baselines")
    selected = set(args.models or TREE_MODELS + NEURAL_MODELS)
    suffix = "_" + "_".join(args.models) if args.models else ""
    summary_path = report_root / f"step125_training_summary{suffix}.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError("Step125 outputs exist; use --overwrite")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if not args.models:
        train_means(train, data, specs, config)
    scalers = fit_scalers(train, data, specs)
    for model_id in (model for model in TREE_MODELS if model in selected):
        for seed in config["training"]["seeds"]:
            results.append(train_trees(model_id, int(seed), train, validation, data, specs, config))
    for model_id in (model for model in NEURAL_MODELS if model in selected):
        for seed in config["training"]["seeds"]:
            results.append(train_neural(model_id, int(seed), train, validation, data, specs, scalers, config))
    summary = {"step": 125, "status": "COMPLETE", "elapsed_seconds": time.perf_counter() - started, "train_records": len(train), "validation_records": len(validation), "test_records_used": 0, "results": results}
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
