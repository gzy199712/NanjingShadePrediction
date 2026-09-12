"""Publication-style Phase E figures generated from formal Test outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_MPL_CACHE = Path(__file__).resolve().parents[1] / "cache/matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FigureStyle:
    dpi: int
    font_family: str
    scatter_alpha: float
    scatter_point_size: float
    output_formats: tuple[str, ...]


COLORS = {
    "train": "#3B6FB6",
    "validation": "#D1783C",
    "true": "#3B6FB6",
    "prediction": "#D1783C",
    "mae": "#3B6FB6",
    "rmse": "#D1783C",
}


def configure_style(style: FigureStyle) -> None:
    plt.rcParams.update(
        {
            "font.family": style.font_family,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    figure: plt.Figure, output_base: Path, style: FigureStyle
) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in style.output_formats:
        path = output_base.with_suffix(f".{extension}")
        figure.savefig(
            path,
            dpi=style.dpi if extension.lower() == "png" else None,
            metadata={"Creator": "GPTthermalcomfort Step20"},
        )
        outputs.append(path)
    plt.close(figure)
    return outputs


def plot_training_curves(
    histories: dict[int, pd.DataFrame],
    output_base: Path,
    style: FigureStyle,
) -> list[Path]:
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.1), sharey=True)
    axes_flat = axes.ravel()
    for axis, (seed, history) in zip(axes_flat, histories.items(), strict=False):
        axis.plot(
            history["epoch"],
            history["train_total_loss"],
            color=COLORS["train"],
            label="Train",
            linewidth=1.5,
        )
        axis.plot(
            history["epoch"],
            history["validation_total_loss"],
            color=COLORS["validation"],
            label="Validation",
            linewidth=1.5,
        )
        best = history.loc[history["validation_total_loss"].idxmin()]
        axis.scatter(
            [best["epoch"]],
            [best["validation_total_loss"]],
            color=COLORS["validation"],
            marker="*",
            s=55,
            zorder=3,
        )
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Total SmoothL1 loss")
        axis.legend(frameon=False)
    axes_flat[-1].axis("off")
    figure.suptitle("Training and validation loss by random seed")
    figure.tight_layout()
    return save_figure(figure, output_base, style)


def plot_observed_predicted(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    label: str,
    unit: str,
    output_base: Path,
    style: FigureStyle,
) -> list[Path]:
    truth = frame[truth_column].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    figure, axis = plt.subplots(figsize=(5.2, 4.8))
    axis.scatter(
        truth,
        prediction,
        s=style.scatter_point_size,
        alpha=style.scatter_alpha,
        color="#3B6FB6",
        edgecolors="none",
        rasterized=True,
    )
    axis.plot([lower, upper], [lower, upper], color="#333333", linewidth=1)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    suffix = f" ({unit})" if unit else ""
    axis.set_xlabel(f"Observed {label}{suffix}")
    axis.set_ylabel(f"Predicted {label}{suffix}")
    axis.set_title(f"Observed vs predicted {label} on Test")
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    return save_figure(figure, output_base, style)


def plot_hourly_errors(
    hourly: pd.DataFrame,
    target: str,
    label: str,
    unit: str,
    output_base: Path,
    style: FigureStyle,
) -> list[Path]:
    subset = hourly.loc[
        (hourly["dataset_split"] == "test") & (hourly["target"] == target)
    ].sort_values("hour")
    mae_column = f"{target}_mae"
    rmse_column = f"{target}_rmse"
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.plot(
        subset["hour"],
        subset[mae_column],
        color=COLORS["mae"],
        marker="o",
        linewidth=1.5,
        label="MAE",
    )
    axis.plot(
        subset["hour"],
        subset[rmse_column],
        color=COLORS["rmse"],
        marker="s",
        linewidth=1.5,
        label="RMSE",
    )
    axis.set_xticks(range(6, 19))
    axis.set_xlabel("Hour (Asia/Shanghai)")
    axis.set_ylabel(f"Error ({unit})" if unit else "Error")
    axis.set_title(f"Hourly Test error: {label}")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    return save_figure(figure, output_base, style)


def plot_error_distributions(
    frame: pd.DataFrame,
    output_base: Path,
    style: FigureStyle,
) -> list[Path]:
    definitions = [
        ("shade_pred_mean", "shade_true", "Shade rate", "proportion"),
        ("tmrt_pred_mean", "tmrt_true", "Tmrt", "°C"),
        ("utci_pred_mean", "utci_true", "UTCI", "°C"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for axis, (prediction, truth, label, unit) in zip(
        axes, definitions, strict=True
    ):
        error = frame[prediction] - frame[truth]
        axis.hist(error, bins=60, color="#3B6FB6", alpha=0.85)
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel(f"Prediction error ({unit})")
        axis.set_ylabel("Records")
    figure.suptitle("Test prediction error distributions")
    figure.tight_layout()
    return save_figure(figure, output_base, style)


def plot_uncertainty_distributions(
    frame: pd.DataFrame,
    output_base: Path,
    style: FigureStyle,
) -> list[Path]:
    definitions = [
        ("shade_pred_std", "Shade rate", "proportion"),
        ("tmrt_pred_std", "Tmrt", "°C"),
        ("utci_pred_std", "UTCI", "°C"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for axis, (column, label, unit) in zip(axes, definitions, strict=True):
        axis.hist(frame[column], bins=60, color="#D1783C", alpha=0.85)
        axis.set_title(label)
        axis.set_xlabel(f"Five-seed prediction SD ({unit})")
        axis.set_ylabel("Records")
    figure.suptitle("Ensemble predictive uncertainty on Test")
    figure.tight_layout()
    return save_figure(figure, output_base, style)


def plot_attention_by_hour(
    attention_index: pd.DataFrame,
    azimuth_degrees: list[float],
    output_base: Path,
    style: FigureStyle,
) -> list[Path]:
    columns = [f"attention_azimuth_{int(value)}" for value in azimuth_degrees]
    matrix = (
        attention_index.groupby("hour", sort=True)[columns].mean().to_numpy()
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
    )
    axis.set_xticks(range(len(azimuth_degrees)))
    axis.set_xticklabels([f"{int(value)}°" for value in azimuth_degrees])
    axis.set_yticks(range(13))
    axis.set_yticklabels(range(6, 19))
    axis.set_xlabel("Street-view window azimuth")
    axis.set_ylabel("Hour (Asia/Shanghai)")
    axis.set_title("Mean ensemble attention weight by hour and direction")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Attention weight")
    figure.tight_layout()
    return save_figure(figure, output_base, style)


def plot_hourly_boxplots(
    frame: pd.DataFrame,
    output_base: Path,
    style: FigureStyle,
) -> list[Path]:
    definitions = [
        ("shade_true", "shade_pred_mean", "Shade rate", "proportion"),
        ("tmrt_true", "tmrt_pred_mean", "Tmrt", "°C"),
        ("utci_true", "utci_pred_mean", "UTCI", "°C"),
    ]
    hours = list(range(6, 19))
    figure, axes = plt.subplots(3, 1, figsize=(10.2, 8.2), sharex=True)
    for axis, (truth, prediction, label, unit) in zip(
        axes, definitions, strict=True
    ):
        true_values = [
            frame.loc[frame["hour"] == hour, truth].to_numpy() for hour in hours
        ]
        predicted_values = [
            frame.loc[frame["hour"] == hour, prediction].to_numpy()
            for hour in hours
        ]
        left = np.arange(len(hours)) * 2.0 - 0.35
        right = np.arange(len(hours)) * 2.0 + 0.35
        true_plot = axis.boxplot(
            true_values,
            positions=left,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        predicted_plot = axis.boxplot(
            predicted_values,
            positions=right,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in true_plot["boxes"]:
            patch.set_facecolor(COLORS["true"])
            patch.set_alpha(0.7)
        for patch in predicted_plot["boxes"]:
            patch.set_facecolor(COLORS["prediction"])
            patch.set_alpha(0.7)
        axis.set_ylabel(f"{label} ({unit})")
        axis.set_title(label)
    axes[-1].set_xticks(np.arange(len(hours)) * 2.0)
    axes[-1].set_xticklabels(hours)
    axes[-1].set_xlabel("Hour (Asia/Shanghai)")
    handles = [
        plt.Line2D([0], [0], color=COLORS["true"], lw=6, alpha=0.7),
        plt.Line2D([0], [0], color=COLORS["prediction"], lw=6, alpha=0.7),
    ]
    figure.legend(
        handles,
        ["Observed", "Predicted"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Hourly observed and predicted distributions on Test", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    return save_figure(figure, output_base, style)


def plot_stress_confusion(
    frame: pd.DataFrame,
    output_base: Path,
    style: FigureStyle,
) -> list[Path]:
    order = [
        "moderate heat stress",
        "strong heat stress",
        "very strong heat stress",
        "extreme heat stress",
    ]
    confusion = pd.crosstab(
        frame["utci_stress_true"], frame["utci_stress_pred"]
    ).reindex(index=order, columns=order, fill_value=0)
    matrix = confusion.to_numpy()
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sum != 0,
    )
    short = ["Moderate", "Strong", "Very strong", "Extreme"]
    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#222222",
                fontsize=8,
            )
    axis.set_xticks(range(4))
    axis.set_xticklabels(short, rotation=25, ha="right")
    axis.set_yticks(range(4))
    axis.set_yticklabels(short)
    axis.set_xlabel("Predicted UTCI stress category")
    axis.set_ylabel("SOLWEIG UTCI stress category")
    axis.set_title("Test UTCI heat-stress confusion matrix")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Row-normalized proportion")
    figure.tight_layout()
    return save_figure(figure, output_base, style)


FigureBuilder = Callable[[], list[Path]]
