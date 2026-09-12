"""Step119: bootstrap uncertainty and publication figures for Step118."""

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
VIEW_LABELS = {
    "unseen_points_seen_weather": "Unseen points\nSeen weather",
    "seen_points_unseen_weather": "Seen points\nUnseen weather",
    "unseen_points_unseen_weather": "Unseen points\nUnseen weather",
}
TARGETS = {
    "shade": ("shade_true", "shade_pred_mean", "Shade rate"),
    "tmrt_celsius": ("tmrt_true", "tmrt_pred_mean", "Tmrt"),
    "utci_celsius": ("utci_true", "utci_pred_mean", "UTCI"),
}
COLORS = ["#3B6FB6", "#2B9B65", "#D95F3A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_reporting.yaml")
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle: value = yaml.safe_load(handle)
    if not isinstance(value, dict): raise ValueError("Config root must be a mapping")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value); return path if path.is_absolute() else root / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def configure_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True); logger = logging.getLogger("step119"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def sufficient_statistics(frame: pd.DataFrame, truth: str, prediction: str) -> pd.DataFrame:
    work = frame[["point_id", truth, prediction]].copy()
    work[truth] = pd.to_numeric(work[truth], errors="coerce"); work[prediction] = pd.to_numeric(work[prediction], errors="coerce")
    work = work[np.isfinite(work[truth]) & np.isfinite(work[prediction])].copy()
    work["sum_y"] = work[truth]; work["sum_p"] = work[prediction]
    work["sum_y2"] = np.square(work[truth]); work["sum_p2"] = np.square(work[prediction])
    work["sum_yp"] = work[truth] * work[prediction]
    error = work[prediction] - work[truth]
    work["sum_abs_error"] = np.abs(error); work["sum_sq_error"] = np.square(error); work["sum_error"] = error; work["count"] = 1
    fields = ["count", "sum_y", "sum_p", "sum_y2", "sum_p2", "sum_yp", "sum_abs_error", "sum_sq_error", "sum_error"]
    return work.groupby("point_id", sort=False)[fields].sum()


def metrics_from_totals(values: np.ndarray) -> dict[str, float]:
    count, sum_y, sum_p, sum_y2, sum_p2, sum_yp, sum_abs, sum_sq, sum_error = values
    sst = sum_y2 - sum_y * sum_y / count
    covariance = sum_yp - sum_y * sum_p / count
    variance_p = sum_p2 - sum_p * sum_p / count
    denominator = math.sqrt(max(sst * variance_p, 0.0))
    return {
        "mae": sum_abs / count, "rmse": math.sqrt(sum_sq / count),
        "r2": 1.0 - sum_sq / sst if sst > 0 else float("nan"),
        "bias": sum_error / count, "pearson": covariance / denominator if denominator > 0 else float("nan"),
    }


def bootstrap_metrics(stats: pd.DataFrame, replicates: int, confidence: float, rng: np.random.Generator) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    matrix = stats.to_numpy(dtype=np.float64); observed = metrics_from_totals(matrix.sum(axis=0)); n = len(matrix)
    samples = rng.integers(0, n, size=(replicates, n), endpoint=False)
    totals = matrix[samples].sum(axis=1)
    estimates = {name: np.empty(replicates, dtype=float) for name in observed}
    for index in tqdm(range(replicates), desc="Cluster bootstrap", unit="replicate", dynamic_ncols=True, leave=False):
        values = metrics_from_totals(totals[index])
        for name in estimates: estimates[name][index] = values[name]
    alpha = (1.0 - confidence) / 2.0
    intervals = {name: (float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1.0 - alpha))) for name, values in estimates.items()}
    return observed, intervals


def create_bootstrap_table(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    settings = config["bootstrap"]; rng = np.random.default_rng(int(settings["seed"])); rows: list[dict[str, Any]] = []
    for view, relative in tqdm(config["inputs"]["predictions"].items(), desc="Bootstrap Test views", unit="view", dynamic_ncols=True):
        usecols = ["point_id", *{field for truth, prediction, _ in TARGETS.values() for field in (truth, prediction)}]
        frame = pd.read_csv(resolve(root, relative), usecols=usecols, dtype={"point_id": str})
        for target, (truth, prediction, _) in tqdm(TARGETS.items(), desc=f"Targets {view}", unit="target", dynamic_ncols=True, leave=False):
            stats = sufficient_statistics(frame, truth, prediction)
            observed, intervals = bootstrap_metrics(stats, int(settings["replicates"]), float(settings["confidence_level"]), rng)
            for metric in ("mae", "rmse", "r2", "bias", "pearson"):
                rows.append({
                    "test_view": view, "target": target, "metric": metric, "estimate": observed[metric],
                    "ci_lower": intervals[metric][0], "ci_upper": intervals[metric][1],
                    "confidence_level": settings["confidence_level"], "bootstrap_replicates": settings["replicates"],
                    "bootstrap_unit": settings["unit"], "cluster_count": len(stats), "valid_record_count": int(stats["count"].sum()),
                })
    return rows


def save_figure(fig: plt.Figure, directory: Path, stem: str, formats: list[str], dpi: int) -> list[str]:
    outputs: list[str] = []
    for extension in tqdm(formats, desc=f"Saving {stem}", unit="format", dynamic_ncols=True, leave=False):
        path = directory / f"{stem}.{extension}"; fig.savefig(path, dpi=dpi, bbox_inches="tight"); outputs.append(str(path))
    plt.close(fig); return outputs


def figure_generalization(ci: pd.DataFrame, directory: Path, config: dict[str, Any]) -> list[str]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=False)
    views = list(VIEW_LABELS); x = np.arange(len(views))
    for axis, (target, (_, _, label)) in zip(axes, TARGETS.items()):
        subset = ci[(ci["target"] == target) & (ci["metric"] == "r2")].set_index("test_view").loc[views]
        values = subset["estimate"].to_numpy(); lower = values - subset["ci_lower"].to_numpy(); upper = subset["ci_upper"].to_numpy() - values
        all_low = subset["ci_lower"].to_numpy(); all_high = subset["ci_upper"].to_numpy()
        span = max(float(all_high.max() - all_low.min()), 0.02)
        y_min = max(-0.05, float(all_low.min()) - span * 0.22)
        y_max = min(1.025, float(all_high.max()) + span * 0.28)
        for index in tqdm(range(len(views)), desc=f"Plotting {target}", unit="view", dynamic_ncols=True, leave=False):
            axis.errorbar(x[index], values[index], yerr=[[lower[index]], [upper[index]]], fmt="o", color=COLORS[index], markersize=7, capsize=4, linewidth=1.5)
            axis.text(x[index], min(y_max - span * 0.04, values[index] + span * 0.08), f"{values[index]:.3f}", ha="center", va="bottom", fontsize=8)
        axis.set_title(label); axis.set_xticks(x, [VIEW_LABELS[view] for view in views]); axis.set_ylim(y_min, y_max)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("R² with point-cluster bootstrap 95% CI")
    fig.suptitle("Multi-date generalization under frozen spatial and weather splits", fontsize=14, y=0.98)
    fig.supxlabel("Frozen Test view", y=0.10)
    fig.text(0.5, 0.02, "Intervals are conditional on the frozen weather dates; Test dates were not used for model selection.", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.24, wspace=0.24)
    return save_figure(fig, directory, "01_test_view_generalization", config["reporting"]["formats"], int(config["reporting"]["dpi"]))


def figure_stability(daily: pd.DataFrame, hourly: pd.DataFrame, directory: Path, config: dict[str, Any]) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    panels = [("tmrt_celsius", "r2", "Daily Tmrt R²"), ("utci_celsius", "r2", "Daily UTCI R²"), ("tmrt_celsius", "mae", "Hourly Tmrt MAE (°C)"), ("utci_celsius", "mae", "Hourly UTCI MAE (°C)")]
    for axis, (target, metric, title) in tqdm(list(zip(axes.flat, panels)), desc="Plotting stability panels", unit="panel", dynamic_ncols=True):
        source = daily if title.startswith("Daily") else hourly
        view_field = "test_view" if title.startswith("Daily") else "dataset_split"
        for index, view in enumerate(VIEW_LABELS):
            subset = source[(source[view_field] == view) & (source["target"] == target)].copy()
            column = f"{target}_{metric}"
            if subset.empty or column not in subset: continue
            if title.startswith("Daily"):
                subset = subset.sort_values("date"); x = pd.to_datetime(subset["date"]); axis.plot(x, subset[column], marker="o", linewidth=1.4, markersize=4, color=COLORS[index], label=VIEW_LABELS[view].replace("\n", " / "))
                axis.tick_params(axis="x", rotation=45)
            else:
                subset = subset.sort_values("hour"); axis.plot(subset["hour"], subset[column], marker="o", linewidth=1.4, markersize=4, color=COLORS[index], label=VIEW_LABELS[view].replace("\n", " / "))
                axis.set_xticks(range(6, 19, 2)); axis.set_xlabel("Hour (local time)")
        axis.set_title(title); axis.grid(alpha=0.25); axis.legend(frameon=False, fontsize=7)
    fig.suptitle("Date- and hour-stratified stability of the frozen five-seed ensemble", fontsize=14)
    fig.tight_layout()
    return save_figure(fig, directory, "02_date_hour_stability", config["reporting"]["formats"], int(config["reporting"]["dpi"]))


def figure_comparison(current: pd.DataFrame, original: pd.DataFrame, directory: Path, config: dict[str, Any]) -> list[str]:
    strict = current[current["dataset_split"].eq(config["reporting"]["primary_view"])]
    old = original[(original["dataset_split"].astype(str).str.lower() == "test") & (original["model"] == "ensemble_mean")]
    targets = ["shade", "tmrt_celsius", "utci_celsius"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8))
    for axis, target in tqdm(list(zip(axes, targets)), desc="Plotting model comparison", unit="target", dynamic_ncols=True):
        new_row = strict[strict["target"].eq(target)].iloc[0]; old_row = old[old["target"].eq(target)].iloc[0]
        column = f"{target}_r2"; values = [float(old_row[column]), float(new_row[column])]
        bars = axis.bar([0, 1], values, color=["#8A8F98", "#3B6FB6"], width=0.62)
        for bar, value in zip(bars, values): axis.text(bar.get_x() + bar.get_width()/2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
        axis.set_xticks([0, 1], ["Original\nsingle-day Test", "Multi-date strict Test"]); axis.set_ylim(max(0, min(values)-0.08), min(1.02, max(values)+0.06)); axis.set_title(TARGETS[target][2]); axis.set_ylabel("R²"); axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Reference comparison: different frozen Test designs", fontsize=14)
    fig.text(0.5, -0.03, "Descriptive comparison only: the two models use different training scopes and Test samples.", ha="center", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, directory, "03_single_day_vs_multidate_reference", config["reporting"]["formats"], int(config["reporting"]["dpi"]))


def main() -> int:
    args = parse_args(); config = load_yaml(args.config.resolve()); root = Path(config["project_root"])
    metrics_dir = resolve(root, config["outputs"]["metrics_directory"]); figure_dir = resolve(root, config["outputs"]["figure_directory"]); report_dir = resolve(root, config["outputs"]["report_directory"]); record_dir = resolve(root, config["outputs"]["record_directory"])
    for directory in (metrics_dir, figure_dir, report_dir, record_dir): directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(record_dir / "step119.log"); started = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    try:
        required = [resolve(root, config["inputs"][key]) for key in ("metrics", "hourly_metrics", "daily_metrics", "original_metrics")] + [resolve(root, value) for value in config["inputs"]["predictions"].values()]
        missing = [str(path) for path in required if not path.exists()]
        if missing: raise FileNotFoundError("Missing inputs: " + "; ".join(missing))
        if not args.overwrite and (metrics_dir / "bootstrap_confidence_intervals.csv").exists(): raise FileExistsError("Step119 outputs exist; use --overwrite")
        if args.mode == "check":
            check = {"status": "PASS", "input_count": len(required), "prediction_files": len(config["inputs"]["predictions"]), "bootstrap_replicates": config["bootstrap"]["replicates"], "no_statistics_or_figures_generated": True}
            atomic_json(record_dir / "preflight.json", check); print(json.dumps(check, ensure_ascii=False, indent=2)); return 0
        ci_rows = create_bootstrap_table(config, root); atomic_csv(metrics_dir / "bootstrap_confidence_intervals.csv", ci_rows)
        ci = pd.DataFrame(ci_rows); current = pd.read_csv(resolve(root, config["inputs"]["metrics"])); daily = pd.read_csv(resolve(root, config["inputs"]["daily_metrics"])); hourly = pd.read_csv(resolve(root, config["inputs"]["hourly_metrics"])); original = pd.read_csv(resolve(root, config["inputs"]["original_metrics"]))
        figures = []
        figures.extend(figure_generalization(ci, figure_dir, config)); figures.extend(figure_stability(daily, hourly, figure_dir, config)); figures.extend(figure_comparison(current, original, figure_dir, config))
        ended = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
        summary = {"step": 119, "status": "PASS", "started_at": started.isoformat(), "ended_at": ended.isoformat(), "elapsed_seconds": (ended-started).total_seconds(), "bootstrap_rows": len(ci_rows), "bootstrap_replicates": config["bootstrap"]["replicates"], "bootstrap_unit": config["bootstrap"]["unit"], "figures": figures, "formal_training_performed": False, "checkpoint_changed": False, "next_step": "Frozen ablation experiment implementation and smoke test."}
        atomic_json(metrics_dir / "summary.json", summary); atomic_csv(record_dir / "failed_files.csv", [])
        logger.info("Step119 completed | figures=%d | bootstrap_rows=%d", len(figures), len(ci_rows)); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    except Exception as error:
        logger.exception("Step119 failed"); atomic_json(record_dir / "failure.json", {"status":"FAIL", "error":str(error), "traceback":traceback.format_exc()}); return 1


if __name__ == "__main__": raise SystemExit(main())
