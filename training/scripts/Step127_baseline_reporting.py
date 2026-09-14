"""Step127: paper tables, point-cluster bootstrap and baseline figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "training/cache/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.scripts.Step123_ablation_reporting import metric_arrays, point_statistics  # noqa: E402
from training.scripts.Step125_train_multidate_baselines import resolve  # noqa: E402
from training.src.data_loading import load_yaml  # noqa: E402
from training.src.logging_utils import atomic_csv, atomic_json  # noqa: E402
from training.src.metrics import regression_metrics  # noqa: E402


TARGETS = {
    "shade": ("shade_true", "shade_pred", "Shade"),
    "tmrt_celsius": ("tmrt_true", "tmrt_pred", "Tmrt"),
    "utci_celsius": ("utci_true", "utci_pred", "UTCI"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_baselines.yaml")
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prediction_path(root: Path, model_id: str, view: str) -> Path:
    ensemble = root / model_id / view / "ensemble.csv.gz"
    return ensemble if ensemble.is_file() else root / model_id / view / "seed_deterministic.csv.gz"


def metrics_for(frame: pd.DataFrame, grouping: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for target, (truth, prediction, _) in TARGETS.items():
        valid = np.isfinite(frame[truth]) & np.isfinite(frame[prediction])
        values = regression_metrics(frame.loc[valid, truth].to_numpy(), frame.loc[valid, prediction].to_numpy(), "metric")
        output.append({**grouping, "target": target, "record_count": len(frame), "valid_count": int(valid.sum()), "MAE": values["metric_mae"], "RMSE": values["metric_rmse"], "R2": values["metric_r2"], "bias": values["metric_bias"], "Pearson_correlation": values["metric_pearson"]})
    return output


def grouped_metrics(config: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hourly: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    for model in config["models"]:
        for view in config["partitions"]["test_views"]:
            frame = pd.read_csv(prediction_path(root, model["id"], view), dtype={"sample_id": str, "point_id": str, "date": str})
            for hour, subset in frame.groupby("hour", sort=True):
                hourly.extend(metrics_for(subset, {"model_id": model["id"], "test_view": view, "hour": int(hour)}))
            for date, subset in frame.groupby("date", sort=True):
                daily.extend(metrics_for(subset, {"model_id": model["id"], "test_view": view, "date": str(date)}))
    return hourly, daily


def seed_uncertainty(per_seed: pd.DataFrame) -> list[dict[str, Any]]:
    numeric = per_seed.loc[per_seed["seed"].astype(str).str.fullmatch(r"\d+")].copy()
    output = []
    for keys, group in numeric.groupby(["model_id", "test_view", "target"], sort=False):
        model_id, view, target = keys
        for metric in ("MAE", "RMSE", "R2", "bias", "Pearson_correlation"):
            values = group[metric].to_numpy(float)
            output.append({"model_id": model_id, "test_view": view, "target": target, "metric": metric, "seed_count": len(values), "mean": float(np.nanmean(values)), "standard_deviation": float(np.nanstd(values, ddof=1)), "minimum": float(np.nanmin(values)), "maximum": float(np.nanmax(values))})
    return output


def paired_bootstrap(config: dict[str, Any], prediction_root: Path) -> list[dict[str, Any]]:
    strict = config["reporting"]["strict_view"]
    proposed = pd.read_csv(prediction_path(prediction_root, "proposed", strict), dtype={"point_id": str})
    point_ids = np.sort(proposed["point_id"].unique())
    count = len(point_ids)
    settings = config["bootstrap"]
    rng = np.random.default_rng(int(settings["seed"]))
    weights = rng.multinomial(count, np.full(count, 1.0 / count), size=int(settings["replicates"])).astype(float)
    alpha = (1.0 - float(settings["confidence_level"])) / 2.0
    output = []
    for target, (truth, prediction, _) in TARGETS.items():
        for model in config["models"]:
            if model["id"] == "proposed":
                continue
            frame = pd.read_csv(prediction_path(prediction_root, model["id"], strict), dtype={"point_id": str})
            if not np.array_equal(frame["sample_id"].astype(str).to_numpy(), proposed["sample_id"].astype(str).to_numpy()):
                raise RuntimeError(f"Paired sample alignment failed: {model['id']}")
            common = np.isfinite(proposed[truth]) & np.isfinite(proposed[prediction]) & np.isfinite(frame[truth]) & np.isfinite(frame[prediction])
            proposed_matrix = point_statistics(proposed.loc[common], truth, prediction).reindex(point_ids)
            matrix = point_statistics(frame.loc[common], truth, prediction).reindex(point_ids)
            if proposed_matrix.isna().any().any() or matrix.isna().any().any():
                raise RuntimeError(f"Point alignment failed: {model['id']}")
            proposed_values = proposed_matrix.to_numpy(float)
            matrix_values = matrix.to_numpy(float)
            proposed_observed = metric_arrays(proposed_values.sum(0, keepdims=True))
            proposed_boot = metric_arrays(weights @ proposed_values)
            observed = metric_arrays(matrix_values.sum(0, keepdims=True))
            bootstrap = metric_arrays(weights @ matrix_values)
            for metric in ("mae", "rmse", "r2"):
                if metric == "r2":
                    delta = proposed_observed[metric][0] - observed[metric][0]
                    samples = proposed_boot[metric] - bootstrap[metric]
                else:
                    delta = observed[metric][0] - proposed_observed[metric][0]
                    samples = bootstrap[metric] - proposed_boot[metric]
                lower, upper = np.nanquantile(samples, [alpha, 1.0 - alpha])
                output.append({
                    "model_id": model["id"], "test_view": strict, "target": target, "metric": metric.upper(),
                    "proposed_value": float(proposed_observed[metric][0]), "baseline_value": float(observed[metric][0]),
                    "difference_positive_is_proposed_better": float(delta), "ci_lower": float(lower), "ci_upper": float(upper),
                    "ci_excludes_zero": bool(lower > 0 or upper < 0), "confidence_level": settings["confidence_level"],
                    "bootstrap_replicates": settings["replicates"], "bootstrap_unit": settings["unit"], "cluster_count": count,
                })
    return output


def save_figure(fig: plt.Figure, directory: Path, stem: str, config: dict[str, Any]) -> list[str]:
    outputs = []
    directory.mkdir(parents=True, exist_ok=True)
    for extension in config["reporting"]["formats"]:
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=int(config["reporting"]["dpi"]), bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def figures(overall: pd.DataFrame, hourly: pd.DataFrame, bootstrap: pd.DataFrame, config: dict[str, Any], directory: Path) -> list[str]:
    sns.set_theme(style="whitegrid", context="paper")
    labels = {item["id"]: item["label"] for item in config["models"]}
    order = [item["id"] for item in config["models"]]
    strict = config["reporting"]["strict_view"]
    strict_data = overall.loc[overall["test_view"].eq(strict)].copy()
    strict_data["model"] = strict_data["model_id"].map(labels)
    outputs: list[str] = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric in zip(axes, ("MAE", "RMSE", "R2"), strict=True):
        sns.barplot(data=strict_data, x="model", y=metric, hue="target", ax=axis)
        axis.tick_params(axis="x", rotation=55); axis.set_title(metric); axis.set_xlabel("")
    fig.suptitle("Strict test performance: unseen points × unseen weather")
    outputs += save_figure(fig, directory, "strict_test_mae_rmse_r2", config)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (target, (_, _, title)) in zip(axes, TARGETS.items(), strict=True):
        subset = strict_data.loc[strict_data["target"].eq(target)]
        sns.barplot(data=subset, x="model", y="MAE", order=[labels[key] for key in order], ax=axis, color="#4C78A8")
        axis.tick_params(axis="x", rotation=55); axis.set_title(title); axis.set_xlabel("")
    outputs += save_figure(fig, directory, "target_model_comparison", config)
    cross = overall.copy(); cross["model"] = cross["model_id"].map(labels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, (target, (_, _, title)) in zip(axes, TARGETS.items(), strict=True):
        sns.barplot(data=cross.loc[cross["target"].eq(target)], x="model", y="MAE", hue="test_view", ax=axis)
        axis.tick_params(axis="x", rotation=55); axis.set_title(title); axis.set_xlabel("")
    outputs += save_figure(fig, directory, "three_view_generalization", config)
    hour = hourly.loc[hourly["test_view"].eq(strict)].copy(); hour["model"] = hour["model_id"].map(labels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, (target, (_, _, title)) in zip(axes, TARGETS.items(), strict=True):
        sns.lineplot(data=hour.loc[hour["target"].eq(target)], x="hour", y="MAE", hue="model", marker="o", ax=axis)
        axis.set_title(title); axis.set_xticks(sorted(hour["hour"].unique()))
    outputs += save_figure(fig, directory, "hourly_error", config)
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for row, target in enumerate(TARGETS):
        for column, metric in enumerate(("MAE", "RMSE", "R2")):
            axis = axes[row, column]
            subset = bootstrap.loc[bootstrap["target"].eq(target) & bootstrap["metric"].eq(metric)].copy()
            subset["model"] = subset["model_id"].map(labels)
            y = np.arange(len(subset)); x = subset["difference_positive_is_proposed_better"].to_numpy(float)
            axis.errorbar(x, y, xerr=np.vstack((x - subset["ci_lower"], subset["ci_upper"] - x)), fmt="o", capsize=3)
            axis.axvline(0, color="black", linewidth=0.8); axis.set_yticks(y, subset["model"]); axis.set_title(f"{TARGETS[target][2]} {metric}")
    fig.suptitle("Baseline − Proposed degradation with paired point-cluster 95% CI")
    fig.subplots_adjust(wspace=0.5, hspace=0.45)
    outputs += save_figure(fig, directory, "difference_vs_proposed_95ci", config)
    return outputs


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    metrics_root = resolve(config["outputs"]["metrics_root"])
    prediction_root = resolve(config["outputs"]["prediction_root"])
    report_root = resolve(config["outputs"]["report_root"])
    per_seed_path = metrics_root / "per_seed_test_metrics.csv"
    required = [per_seed_path] + [prediction_path(prediction_root, model["id"], view) for model in config["models"] for view in config["partitions"]["test_views"]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step126 inputs: " + "; ".join(missing))
    report_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "check":
        report = {"step": 127, "status": "PASS", "input_count": len(required), "bootstrap_replicates": config["bootstrap"]["replicates"], "statistics_generated": False}
        atomic_json(report_root / "step127_check.json", report)
        print(json.dumps(report, indent=2))
        return 0
    summary_path = metrics_root / "baseline_comparison_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError("Step127 outputs exist; use --overwrite")
    per_seed = pd.read_csv(per_seed_path, dtype={"seed": str})
    overall = per_seed.loc[per_seed["seed"].isin(["ensemble", "deterministic"])].copy()
    hourly_rows, daily_rows = grouped_metrics(config, prediction_root)
    uncertainty_rows = seed_uncertainty(per_seed)
    bootstrap_rows = paired_bootstrap(config, prediction_root)
    hourly = pd.DataFrame(hourly_rows); daily = pd.DataFrame(daily_rows); bootstrap = pd.DataFrame(bootstrap_rows)
    atomic_csv(metrics_root / "overall_baseline_comparison.csv", overall.to_dict("records"))
    atomic_csv(metrics_root / "seed_uncertainty.csv", uncertainty_rows)
    atomic_csv(metrics_root / "hourly_baseline_comparison.csv", hourly_rows)
    atomic_csv(metrics_root / "daily_baseline_comparison.csv", daily_rows)
    atomic_csv(metrics_root / "paired_baseline_bootstrap_ci.csv", bootstrap_rows)
    figure_paths = figures(overall, hourly, bootstrap, config, resolve(config["outputs"]["figure_root"]))
    summary = {
        "step": 127, "status": "COMPLETE", "models": [item["id"] for item in config["models"]],
        "test_views": list(config["partitions"]["test_views"]), "targets": list(TARGETS),
        "seeds": config["training"]["seeds"], "proposed_seeds": config["proposed"]["seeds"],
        "bootstrap_replicates": config["bootstrap"]["replicates"], "bootstrap_unit": config["bootstrap"]["unit"],
        "tree_algorithm": config["training"]["tree_algorithm"], "tree_randomness": "deterministic",
        "static_tree_preprocessing": f"direction mean/std pooling; {config['training']['static_pca_components']}-component randomized PCA fit on Train only",
        "neural_preprocessing": "all normalization moments fit on Train only",
        "model_selection": "MLP early stopping on validation_unseen_points_unseen_weather only; Test never used for selection",
        "utci": "training.src.utci.calculate_utci for truth and every model prediction",
        "test_evaluation_runs": 1, "figure_files": figure_paths,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
