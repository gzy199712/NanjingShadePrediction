"""Step123: paired point-cluster bootstrap and formal ablation figures."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SCRIPT_VERSION = "1.0.0"
TARGETS = {
    "shade": ("shade_true", "shade_pred_mean", "Shade", "rate"),
    "tmrt_celsius": ("tmrt_true", "tmrt_pred_mean", "Tmrt", "°C"),
    "utci_celsius": ("utci_true", "utci_pred_mean", "UTCI", "°C"),
}
GROUP_COLORS = {
    "visual_semantic": "#3B6FB6",
    "weather": "#D95F3A",
    "fusion_multitask": "#2B9B65",
    "baseline": "#6B7280",
}
VIEW_LABELS = {
    "unseen_points_seen_weather": "Unseen points\nSeen weather",
    "seen_points_unseen_weather": "Seen points\nUnseen weather",
    "unseen_points_unseen_weather": "Unseen points\nUnseen weather",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_ablation_reporting.yaml")
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Config root must be a mapping")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def logger_for(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step123"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def load_prediction(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as archive:
        return pd.DataFrame({key: archive[key] for key in archive.files})


def point_statistics(frame: pd.DataFrame, truth: str, prediction: str) -> pd.DataFrame:
    work = frame[["point_id", truth, prediction]].copy()
    valid = np.isfinite(work[truth].astype(float)) & np.isfinite(work[prediction].astype(float))
    work = work.loc[valid].copy(); y = work[truth].astype(float); p = work[prediction].astype(float); error = p - y
    work["count"] = 1.0; work["sum_y"] = y; work["sum_y2"] = y * y
    work["sum_abs_error"] = np.abs(error); work["sum_sq_error"] = error * error; work["sum_error"] = error
    return work.groupby("point_id", sort=True)[["count", "sum_y", "sum_y2", "sum_abs_error", "sum_sq_error", "sum_error"]].sum()


def metric_arrays(totals: np.ndarray) -> dict[str, np.ndarray]:
    count, sum_y, sum_y2, sum_abs, sum_sq, sum_error = [totals[..., index] for index in range(totals.shape[-1])]
    sst = sum_y2 - sum_y * sum_y / count
    return {
        "mae": sum_abs / count,
        "rmse": np.sqrt(sum_sq / count),
        "r2": 1.0 - sum_sq / sst,
        "bias": sum_error / count,
    }


def paired_bootstrap(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    settings = config["bootstrap"]; strict = config["reporting"]["strict_view"]
    prediction_root = resolve(root, config["inputs"]["prediction_directory"])
    baseline_frame = load_prediction(prediction_root / "full_model_3seed" / f"{strict}.npz")
    point_ids = np.sort(baseline_frame["point_id"].astype(str).unique())
    n = len(point_ids); replicates = int(settings["replicates"]); rng = np.random.default_rng(int(settings["seed"]))
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=replicates).astype(np.float64)
    alpha = (1.0 - float(settings["confidence_level"])) / 2.0
    rows: list[dict[str, Any]] = []
    variants = [item for item in config["variants"] if item["id"] != "full_model_3seed"]
    for target, (truth, prediction, _, _) in tqdm(TARGETS.items(), desc="Paired bootstrap targets", unit="target", dynamic_ncols=True):
        baseline_stats = point_statistics(baseline_frame, truth, prediction).reindex(point_ids)
        if baseline_stats.isna().any().any(): raise RuntimeError("Baseline point alignment failed")
        baseline_matrix = baseline_stats.to_numpy(dtype=np.float64)
        base_observed = metric_arrays(baseline_matrix.sum(axis=0, keepdims=True))
        base_bootstrap = metric_arrays(weights @ baseline_matrix)
        for item in tqdm(variants, desc=f"{target} variants", unit="variant", dynamic_ncols=True, leave=False):
            frame = load_prediction(prediction_root / item["id"] / f"{strict}.npz")
            stats = point_statistics(frame, truth, prediction).reindex(point_ids)
            if stats.isna().any().any(): raise RuntimeError(f"Point alignment failed: {item['id']}")
            matrix = stats.to_numpy(dtype=np.float64)
            observed = metric_arrays(matrix.sum(axis=0, keepdims=True)); boot = metric_arrays(weights @ matrix)
            for metric in ("mae", "rmse", "r2"):
                if metric == "r2":
                    delta = float(base_observed[metric][0] - observed[metric][0]); samples = base_bootstrap[metric] - boot[metric]
                else:
                    delta = float(observed[metric][0] - base_observed[metric][0]); samples = boot[metric] - base_bootstrap[metric]
                lower, upper = np.nanquantile(samples, [alpha, 1.0 - alpha])
                rows.append({
                    "variant_id": item["id"], "label": item["label"], "group": item["group"], "test_view": strict,
                    "target": target, "metric": metric, "baseline_value": float(base_observed[metric][0]),
                    "ablation_value": float(observed[metric][0]), "degradation_positive_is_worse": delta,
                    "ci_lower": float(lower), "ci_upper": float(upper),
                    "ci_excludes_zero": bool(lower > 0 or upper < 0), "confidence_level": settings["confidence_level"],
                    "bootstrap_replicates": replicates, "bootstrap_unit": settings["unit"], "cluster_count": n,
                })
    return rows


def seed_summary(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    source = pd.read_csv(resolve(root, config["inputs"]["seed_metrics"]))
    labels = {item["id"]: item for item in config["variants"]}
    rows: list[dict[str, Any]] = []
    for keys, group in tqdm(source.groupby(["variant_id", "test_view", "target"], sort=False), desc="Seed uncertainty", unit="group", dynamic_ncols=True):
        variant, view, target = keys; info = labels[variant]
        for metric in ("mae", "rmse", "r2"):
            field = f"{target}_{metric}"; values = pd.to_numeric(group[field], errors="coerce").to_numpy(dtype=float)
            rows.append({
                "variant_id": variant, "label": info["label"], "group": info["group"], "test_view": view, "target": target, "metric": metric,
                "seed_count": len(values), "mean": float(np.nanmean(values)), "standard_deviation": float(np.nanstd(values, ddof=1)),
                "minimum": float(np.nanmin(values)), "maximum": float(np.nanmax(values)),
            })
    return rows


def save_figure(fig: plt.Figure, directory: Path, stem: str, config: dict[str, Any]) -> list[str]:
    outputs: list[str] = []
    for extension in tqdm(config["reporting"]["formats"], desc=f"Saving {stem}", unit="format", dynamic_ncols=True, leave=False):
        path = directory / f"{stem}.{extension}"; fig.savefig(path, dpi=int(config["reporting"]["dpi"]), bbox_inches="tight"); outputs.append(str(path))
    plt.close(fig); return outputs


def plot_degradation(bootstrap: pd.DataFrame, directory: Path, config: dict[str, Any]) -> list[str]:
    panels = [("shade", "r2", "Shade R² decrease", "ΔR²"), ("tmrt_celsius", "mae", "Tmrt MAE increase", "ΔMAE (°C)"), ("utci_celsius", "mae", "UTCI MAE increase", "ΔMAE (°C)")]
    labels = {item["id"]: item["label"] for item in config["variants"]}
    order = [item["id"] for item in config["variants"] if item["id"] != "full_model_3seed"]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 7.0), sharey=True)
    for axis, (target, metric, title, xlabel) in tqdm(list(zip(axes, panels)), desc="Plotting degradation", unit="panel", dynamic_ncols=True):
        subset = bootstrap[(bootstrap["target"] == target) & (bootstrap["metric"] == metric)].set_index("variant_id").loc[order]
        y = np.arange(len(order)); values = subset["degradation_positive_is_worse"].to_numpy(dtype=float)
        lower = values - subset["ci_lower"].to_numpy(dtype=float); upper = subset["ci_upper"].to_numpy(dtype=float) - values
        colors = [GROUP_COLORS[group] for group in subset["group"]]
        axis.barh(y, values, color=colors, alpha=0.82, height=0.62)
        axis.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#222222", capsize=3, linewidth=1.0)
        axis.axvline(0, color="#4B5563", linewidth=0.9); axis.set_title(title); axis.set_xlabel(xlabel); axis.grid(axis="x", alpha=0.22)
        axis.set_yticks(y, [labels[value] for value in order]); axis.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[group], alpha=0.82) for group in ("visual_semantic", "weather", "fusion_multitask")]
    fig.legend(handles, ["Visual / semantic", "Weather / solar", "Fusion / multitask"], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Ablation degradation on unseen points × unseen weather\nPaired point-cluster bootstrap 95% confidence intervals", fontsize=14)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.86, bottom=0.12, wspace=0.16)
    return save_figure(fig, directory, "01_strict_test_ablation_degradation", config)


def plot_seed_stability(seed: pd.DataFrame, directory: Path, config: dict[str, Any]) -> list[str]:
    strict = config["reporting"]["strict_view"]
    panels = [("shade", "r2", "Shade R²"), ("tmrt_celsius", "r2", "Tmrt R²"), ("utci_celsius", "r2", "UTCI R²")]
    order = [item["id"] for item in config["variants"]]; labels = {item["id"]: item["label"] for item in config["variants"]}
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 7.2), sharey=True)
    for axis, (target, metric, title) in tqdm(list(zip(axes, panels)), desc="Plotting seed stability", unit="panel", dynamic_ncols=True):
        subset = seed[(seed["test_view"] == strict) & (seed["target"] == target) & (seed["metric"] == metric)].set_index("variant_id").loc[order]
        y = np.arange(len(order)); values = subset["mean"].to_numpy(dtype=float); errors = subset["standard_deviation"].to_numpy(dtype=float)
        colors = [GROUP_COLORS[group] for group in subset["group"]]
        axis.errorbar(values, y, xerr=errors, fmt="none", ecolor="#4B5563", capsize=3, linewidth=1.0)
        axis.scatter(values, y, c=colors, s=38, zorder=3); axis.set_title(title); axis.set_xlabel("Mean ± SD across 3 seeds"); axis.grid(axis="x", alpha=0.22)
        axis.set_yticks(y, [labels[value] for value in order]); axis.invert_yaxis()
    fig.suptitle("Random-seed stability on the strict frozen Test view", fontsize=14)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.90, bottom=0.08, wspace=0.16)
    return save_figure(fig, directory, "02_three_seed_stability", config)


def plot_cross_view(degradation: pd.DataFrame, directory: Path, config: dict[str, Any]) -> list[str]:
    variants = [item for item in config["variants"] if item["id"] != "full_model_3seed"]
    order = [item["id"] for item in variants]; labels = [item["label"] for item in variants]; views = list(VIEW_LABELS)
    short_views = ["UP × SW", "SP × UW", "UP × UW"]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 7.2), sharey=True)
    matrices: list[np.ndarray] = []
    for target in TARGETS:
        subset = degradation[(degradation["target"] == target) & (degradation["metric"] == "r2")]
        pivot = subset.pivot(index="variant_id", columns="test_view", values="relative_degradation_percent").reindex(index=order, columns=views)
        matrices.append(pivot.to_numpy(dtype=float))
    bound = max(abs(float(np.nanmin(np.concatenate(matrices)))), abs(float(np.nanmax(np.concatenate(matrices)))))
    images = []
    for axis, target, matrix in tqdm(list(zip(axes, TARGETS, matrices)), desc="Plotting cross-view heatmaps", unit="panel", dynamic_ncols=True):
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound); images.append(image)
        axis.set_title(TARGETS[target][2]); axis.set_xticks(np.arange(len(views)), short_views)
        axis.set_yticks(np.arange(len(order)), labels)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                text_color = "white" if abs(matrix[row, column]) > bound * 0.55 else "black"
                axis.text(column, row, f"{matrix[row, column]:+.1f}%", ha="center", va="center", fontsize=7, color=text_color)
    colorbar_axis = fig.add_axes([0.915, 0.23, 0.014, 0.58])
    colorbar = fig.colorbar(images[-1], cax=colorbar_axis); colorbar.set_label("Relative R² degradation (positive = worse)")
    fig.suptitle("Cross-view robustness of ablation effects", fontsize=14)
    fig.text(0.56, 0.025, "UP: unseen points   SP: seen points   SW: seen weather   UW: unseen weather", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.18, right=0.88, top=0.90, bottom=0.10, wspace=0.16)
    return save_figure(fig, directory, "03_cross_view_ablation_robustness", config)


def main() -> int:
    args = parse_args(); config = load_yaml(args.config.resolve()); root = Path(config["project_root"])
    metrics_dir = resolve(root, config["outputs"]["metrics_directory"]); figure_dir = resolve(root, config["outputs"]["figure_directory"]); report_dir = resolve(root, config["outputs"]["report_directory"]); record_dir = resolve(root, config["outputs"]["record_directory"])
    for directory in (metrics_dir, figure_dir, report_dir, record_dir): directory.mkdir(parents=True, exist_ok=True)
    logger = logger_for(record_dir / "step123.log"); started = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    try:
        required = [resolve(root, config["inputs"][key]) for key in ("overall_metrics", "seed_metrics", "degradation_metrics")]
        strict = config["reporting"]["strict_view"]; prediction_root = resolve(root, config["inputs"]["prediction_directory"])
        required.extend(prediction_root / item["id"] / f"{strict}.npz" for item in config["variants"])
        missing = [str(path) for path in required if not path.exists()]
        if missing: raise FileNotFoundError("Missing Step123 inputs: " + "; ".join(missing))
        if not args.overwrite and (metrics_dir / "paired_bootstrap_degradation_ci.csv").exists(): raise FileExistsError("Step123 outputs exist; use --overwrite")
        if args.mode == "check":
            report = {"status":"PASS", "input_count":len(required), "variant_count":len(config["variants"]), "bootstrap_replicates":config["bootstrap"]["replicates"], "no_statistics_or_figures_generated":True}
            atomic_json(record_dir / "preflight.json", report); print(json.dumps(report, ensure_ascii=True, indent=2)); return 0
        bootstrap_rows = paired_bootstrap(config, root); seed_rows = seed_summary(config, root)
        atomic_csv(metrics_dir / "paired_bootstrap_degradation_ci.csv", bootstrap_rows); atomic_csv(metrics_dir / "three_seed_uncertainty.csv", seed_rows)
        bootstrap = pd.DataFrame(bootstrap_rows); seeds = pd.DataFrame(seed_rows); degradation = pd.read_csv(resolve(root, config["inputs"]["degradation_metrics"]))
        figures: list[str] = []
        figures.extend(plot_degradation(bootstrap, figure_dir, config)); figures.extend(plot_seed_stability(seeds, figure_dir, config)); figures.extend(plot_cross_view(degradation, figure_dir, config))
        ended = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
        summary = {"step":123, "status":"PASS", "started_at":started.isoformat(), "ended_at":ended.isoformat(), "elapsed_seconds":(ended-started).total_seconds(), "paired_bootstrap_rows":len(bootstrap_rows), "seed_uncertainty_rows":len(seed_rows), "bootstrap_replicates":config["bootstrap"]["replicates"], "cluster_count":int(bootstrap_rows[0]["cluster_count"]), "figures":figures, "model_inference_performed":False, "training_performed":False}
        atomic_json(metrics_dir / "summary.json", summary); atomic_csv(record_dir / "failed_files.csv", [])
        logger.info("Step123 completed | bootstrap=%d | figures=%d", len(bootstrap_rows), len(figures)); print(json.dumps(summary, ensure_ascii=True, indent=2)); return 0
    except Exception as error:
        logger.exception("Step123 failed"); atomic_json(record_dir / "failure.json", {"status":"FAIL", "error":str(error), "traceback":traceback.format_exc()}); return 1


if __name__ == "__main__": raise SystemExit(main())
