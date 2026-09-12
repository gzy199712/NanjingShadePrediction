"""Plot the current formal routing network and approved topology repairs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
SEGMENT_GDB = ROOT / "routing/data/thermal_edges/step26_thermal_segments.gdb"
BOUNDARY_SHP = ROOT / "data/raw/Nanjing_center_boundary/Nanjing_center_UTM50N.shp"
FIGURE_DIR = ROOT / "routing/outputs/figures"
SUMMARY_PATH = ROOT / "routing/data/topology/step24e_current_network_figure_summary.json"


def geometry_lines(geometry: Any) -> list[np.ndarray]:
    if geometry is None or geometry.IsEmpty():
        return []
    name = geometry.GetGeometryName().upper()
    if name in {"LINESTRING", "LINEARRING"}:
        points = np.asarray([(geometry.GetX(i), geometry.GetY(i)) for i in range(geometry.GetPointCount())], dtype=float)
        return [points] if len(points) >= 2 else []
    result: list[np.ndarray] = []
    for index in range(geometry.GetGeometryCount()):
        result.extend(geometry_lines(geometry.GetGeometryRef(index)))
    return result


def add_scale_and_north(axis: Any, low: np.ndarray, high: np.ndarray) -> None:
    width = high[0] - low[0]
    height = high[1] - low[1]
    scale = 5000.0
    x0 = low[0] + width * 0.06
    y0 = low[1] + height * 0.055
    axis.plot([x0, x0 + scale], [y0, y0], color="#202020", linewidth=3.0, solid_capstyle="butt", zorder=10)
    axis.text(x0 + scale / 2, y0 + height * 0.015, "5 km", ha="center", va="bottom", fontsize=10)
    nx = high[0] - width * 0.07
    ny = high[1] - height * 0.16
    axis.annotate("", xy=(nx, ny + height * 0.09), xytext=(nx, ny), arrowprops={"facecolor": "#202020", "edgecolor": "#202020", "width": 3, "headwidth": 12})
    axis.text(nx, ny - height * 0.018, "N", ha="center", va="top", fontsize=12, fontweight="bold")


def main() -> int:
    os.environ.setdefault("GDAL_PAM_PROXY_DIR", str(ROOT / "routing/data/thermal_edges/.gdal_pam_proxy"))
    from osgeo import ogr
    ogr.UseExceptions()
    categories: dict[str, list[np.ndarray]] = {
        "formal_network": [], "initial_2m": [], "od025": [], "global": [],
    }
    repair_midpoints: dict[str, list[tuple[float, float]]] = {"initial_2m": [], "od025": [], "global": []}
    dataset = ogr.Open(str(SEGMENT_GDB), 0)
    layer = dataset.GetLayerByName("ThermalCostSegments")
    if layer is None:
        raise KeyError("ThermalCostSegments")
    for feature in tqdm(layer, total=layer.GetFeatureCount(), desc="Reading current formal route network", unit="segment", dynamic_ncols=True):
        original = str(feature.GetField("orig_edge") or "")
        if original == "EXT_GAP_0001":
            category = "od025"
        elif original.startswith("EXT_GLOBAL_"):
            category = "global"
        elif original.startswith("SNAP_"):
            category = "initial_2m"
        else:
            category = "formal_network"
        lines = geometry_lines(feature.GetGeometryRef())
        categories[category].extend(lines)
        if category != "formal_network":
            for line in lines:
                repair_midpoints[category].append(tuple(((line[0] + line[-1]) / 2.0).tolist()))
    dataset = None

    boundary_lines: list[np.ndarray] = []
    boundary_ds = ogr.Open(str(BOUNDARY_SHP), 0)
    boundary_layer = boundary_ds.GetLayer(0)
    for feature in tqdm(boundary_layer, total=boundary_layer.GetFeatureCount(), desc="Reading center boundary", unit="feature", dynamic_ncols=True):
        boundary = feature.GetGeometryRef().Boundary()
        boundary_lines.extend(geometry_lines(boundary))
    boundary_ds = None
    boundary_points = np.vstack(boundary_lines)
    low, high = boundary_points.min(axis=0), boundary_points.max(axis=0)
    padding = (high - low) * 0.025
    low -= padding; high += padding

    colors = {
        "formal_network": "#8ba6b5", "initial_2m": "#f2a93b", "od025": "#7b3294", "global": "#d73027",
    }
    figure, axis = plt.subplots(figsize=(13.5, 11.5))
    axis.add_collection(LineCollection(categories["formal_network"], colors=colors["formal_network"], linewidths=0.28, alpha=0.62, zorder=1))
    axis.add_collection(LineCollection(boundary_lines, colors="#252525", linewidths=1.0, alpha=0.9, zorder=2))
    for category, linewidth in (("initial_2m", 1.2), ("od025", 2.4), ("global", 2.2)):
        if categories[category]:
            axis.add_collection(LineCollection(categories[category], colors=colors[category], linewidths=linewidth, alpha=0.98, zorder=4))
        if repair_midpoints[category]:
            points = np.asarray(repair_midpoints[category])
            size = 8 if category == "initial_2m" else 30
            axis.scatter(points[:, 0], points[:, 1], s=size, c=colors[category], edgecolors="none", alpha=0.9, zorder=5)
    axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1]); axis.set_aspect("equal", adjustable="box")
    axis.set_title("Current formal thermal-comfort routing network — after Step24e", fontsize=17, pad=14)
    axis.set_xlabel("Easting (m), EPSG:32650"); axis.set_ylabel("Northing (m), EPSG:32650")
    axis.grid(color="#d8d8d8", linewidth=0.35, alpha=0.45)
    add_scale_and_north(axis, low, high)
    legend = [
        Line2D([0], [0], color=colors["formal_network"], linewidth=2.0, label="Formal routing network (206,324 segments)"),
        Line2D([0], [0], color=colors["initial_2m"], marker="o", markersize=5, linewidth=1.5, label="Initial ≤2 m repairs (57)"),
        Line2D([0], [0], color=colors["od025"], marker="o", markersize=6, linewidth=2.5, label="od_025 approved repair (1)"),
        Line2D([0], [0], color=colors["global"], marker="o", markersize=6, linewidth=2.3, label="Global strict repairs (4)"),
        Patch(facecolor="none", edgecolor="#252525", label="Nanjing center boundary"),
    ]
    axis.legend(handles=legend, loc="lower right", frameon=True, framealpha=0.94, fontsize=10)
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    overview_png = FIGURE_DIR / "current_formal_route_network_step24e.png"
    overview_pdf = FIGURE_DIR / "current_formal_route_network_step24e.pdf"
    figure.savefig(overview_png, dpi=300, bbox_inches="tight")
    figure.savefig(overview_pdf, bbox_inches="tight")
    plt.close(figure)

    graph = np.load(ROOT / "routing/data/graph/step27_route_graph.npz", allow_pickle=False)
    xy, edge_u, edge_v = graph["node_xy"], graph["edge_u"], graph["edge_v"]
    original_ids = graph["original_edge_ids"].astype(str)
    focus_ids = ["EXT_GAP_0001", "EXT_GLOBAL_00001", "EXT_GLOBAL_00002", "EXT_GLOBAL_00003", "EXT_GLOBAL_00004"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9.5), squeeze=False)
    for axis, edge_id in zip(axes.flat, focus_ids, strict=False):
        indexes = np.flatnonzero(original_ids == edge_id)
        if not len(indexes):
            axis.set_title(f"{edge_id} — missing"); axis.axis("off"); continue
        endpoints = np.vstack((xy[edge_u[indexes]], xy[edge_v[indexes]]))
        center = endpoints.mean(axis=0)
        repair_length = float(graph["lengths"][indexes].sum())
        radius = max(140.0, repair_length * 1.7)
        low_local, high_local = center - radius, center + radius
        mask = (
            (xy[edge_u, 0] >= low_local[0]) & (xy[edge_u, 0] <= high_local[0]) & (xy[edge_u, 1] >= low_local[1]) & (xy[edge_u, 1] <= high_local[1])
        ) | (
            (xy[edge_v, 0] >= low_local[0]) & (xy[edge_v, 0] <= high_local[0]) & (xy[edge_v, 1] >= low_local[1]) & (xy[edge_v, 1] <= high_local[1])
        )
        local = np.flatnonzero(mask & ~np.isin(np.arange(len(edge_u)), indexes))
        local_segments = np.stack((xy[edge_u[local]], xy[edge_v[local]]), axis=1)
        repair_segments = np.stack((xy[edge_u[indexes]], xy[edge_v[indexes]]), axis=1)
        axis.add_collection(LineCollection(local_segments, colors="#c7c7c7", linewidths=0.75, alpha=0.8))
        repair_color = colors["od025"] if edge_id == "EXT_GAP_0001" else colors["global"]
        axis.add_collection(LineCollection(repair_segments, colors=repair_color, linewidths=3.0, zorder=4))
        axis.scatter(endpoints[:, 0], endpoints[:, 1], c=repair_color, s=22, zorder=5)
        axis.set_xlim(low_local[0], high_local[0]); axis.set_ylim(low_local[1], high_local[1]); axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{edge_id} | {repair_length:.2f} m", fontsize=11)
        axis.grid(color="#dddddd", linewidth=0.35, alpha=0.5)
        axis.tick_params(labelsize=8)
    axes.flat[-1].axis("off")
    figure.suptitle("Approved aligned-gap repairs in the current route network", fontsize=16)
    figure.tight_layout()
    detail_png = FIGURE_DIR / "current_route_network_repair_details_step24e.png"
    detail_pdf = FIGURE_DIR / "current_route_network_repair_details_step24e.pdf"
    figure.savefig(detail_png, dpi=300, bbox_inches="tight")
    figure.savefig(detail_pdf, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "success": True,
        "formal_segment_count": sum(len(value) for value in categories.values()),
        "initial_2m_repair_segment_count": len(categories["initial_2m"]),
        "od025_repair_segment_count": len(categories["od025"]),
        "global_repair_segment_count": len(categories["global"]),
        "overview_png": str(overview_png), "overview_pdf": str(overview_pdf),
        "detail_png": str(detail_png), "detail_pdf": str(detail_pdf),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
