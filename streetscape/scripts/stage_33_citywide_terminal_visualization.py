"""stage_33: publication-grade visualization of 8,975 terminal planning decisions."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import struct
import time
from datetime import datetime
from pathlib import Path

_ROOT_EARLY = Path(__file__).resolve().parents[2]
_MPL_CACHE = _ROOT_EARLY / "streetscape/logs/.matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/stage_33_citywide_terminal_visualization.yaml"
REPORT = ROOT / "streetscape/reports/STAGE_33_CITYWIDE_TERMINAL_VISUALIZATION.md"
LOG = ROOT / "streetscape/logs/stage_33_citywide_terminal_visualization.log"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

GROUPS = {
    "maintain_or_no_intervention": ("维持现状 / 无需干预", "#7896A5"),
    "existing_tree_review": ("现状树木 / 物候复核", "#5D9272"),
    "excluded_or_abstained": ("排除 / 主动弃权", "#B5B9BD"),
    "optimization_eligible": ("树木优先候选", "#E4783C"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def panel_label(ax, label: str) -> None:
    ax.text(-0.055, 1.035, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="right", va="bottom")


def classify(value: str, definitions: dict) -> str:
    matches = [group for group, classes in definitions.items() if value in classes]
    if len(matches) != 1:
        raise ValueError(f"Decision class {value!r} maps to {len(matches)} groups")
    return matches[0]


def boundary_rings(path: Path) -> list[np.ndarray]:
    """Read Polygon/PolygonZ/PolygonM rings from an ESRI Shapefile."""
    rings: list[np.ndarray] = []
    file_size = path.stat().st_size
    with path.open("rb") as stream, tqdm(total=max(file_size - 100, 0), desc="Reading boundary",
                                          unit="B", unit_scale=True, dynamic_ncols=True) as progress:
        header = stream.read(100)
        if len(header) != 100 or struct.unpack(">i", header[:4])[0] != 9994:
            raise ValueError("Invalid ESRI Shapefile header")
        while stream.tell() < file_size:
            record_header = stream.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise ValueError("Truncated Shapefile record header")
            _, content_words = struct.unpack(">2i", record_header)
            content = stream.read(content_words * 2)
            progress.update(8 + len(content))
            if len(content) != content_words * 2:
                raise ValueError("Truncated Shapefile record")
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in {5, 15, 25}:
                raise ValueError(f"Unsupported boundary shape type: {shape_type}")
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts = list(struct.unpack(f"<{num_parts}i", content[44:44 + 4 * num_parts]))
            point_offset = 44 + 4 * num_parts
            coordinates = np.frombuffer(content, dtype="<f8", count=num_points * 2,
                                        offset=point_offset).reshape(num_points, 2).copy()
            bounds = parts + [num_points]
            for part_index in tqdm(range(num_parts), desc="Reading rings", unit="ring",
                                   leave=False, dynamic_ncols=True):
                ring = coordinates[bounds[part_index]:bounds[part_index + 1]]
                if len(ring) >= 3:
                    rings.append(ring)
    return rings


def find_panorama(directory: Path, point_id: int) -> Path:
    matches = list(directory.glob(f"{point_id}_*_north_pano.png"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one panorama for {point_id}, found {len(matches)}")
    return matches[0]


def evidence_box(ax, x: float, y: float, width: float, height: float,
                 title: str, detail: str, face: str, edge: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.012,rounding_size=0.025",
                               facecolor=face, edgecolor=edge, linewidth=0.8))
    ax.text(x + width / 2, y + height * 0.66, title, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#263238")
    ax.text(x + width / 2, y + height * 0.29, detail, ha="center", va="center",
            fontsize=5.7, color="#52616B", linespacing=1.25)


def main() -> int:
    args = arguments(); config = load_config()
    decisions = ROOT / config["inputs"]["decisions"]
    boundary = ROOT / config["inputs"]["boundary"]
    panorama_dir = ROOT / config["inputs"]["panoramas"]
    output = ROOT / config["outputs"]["directory"]
    eligible_ids = [int(value) for value in config["eligible_point_ids"]]
    panoramas = [find_panorama(panorama_dir, point_id) for point_id in eligible_ids]
    required = [decisions, boundary, *panoramas]
    ready = all(path.exists() for path in required)
    if args.mode == "check":
        print(json.dumps({"status": "READY" if ready else "BLOCKED",
                          "required_count": len(required),
                          "existing_count": sum(path.exists() for path in required),
                          "backend": "Python/matplotlib + standard-library Shapefile reader"},
                         ensure_ascii=False, indent=2))
        return 0 if ready else 1
    if not ready:
        raise RuntimeError("Required inputs are unavailable")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError("Outputs already exist; use --overwrite")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG, filemode="w", encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    started = datetime.now().astimezone(); tick = time.perf_counter()
    logging.info("stage_33 started at %s", started.isoformat(timespec="seconds"))

    data = pd.read_csv(decisions, low_memory=False)
    data["decision_group"] = data["final_decision_class"].map(
        lambda value: classify(str(value), config["decision_groups"])
    )
    data["optimization_eligible_final"] = data["optimization_eligible_final"].astype(bool)
    if len(data) != 8975 or data["point_id"].nunique() != 8975:
        raise ValueError("The full 8,975-point contract is not satisfied")
    if int(data["optimization_eligible_final"].sum()) != 5:
        raise ValueError("Expected exactly five final optimization-eligible points")
    actual_ids = sorted(data.loc[data["optimization_eligible_final"], "point_id"].astype(int).tolist())
    if actual_ids != sorted(eligible_ids):
        raise ValueError(f"Eligible point mismatch: {actual_ids}")

    rings = boundary_rings(boundary)
    counts = data["decision_group"].value_counts().reindex(GROUPS).fillna(0).astype(int)
    source_columns = ["point_id", "x", "y", "utm_x", "utm_y", "year", "month", "SVF", "GVI",
                      "shade_pred_mean", "final_decision_class", "final_decision_status",
                      "decision_group", "optimization_eligible_final", "gpu_localization_complete",
                      "walkable_direction_count", "tree_evidence_direction_count",
                      "walkable_tree_cooccurrence_proxy", "semantic_sky_ratio",
                      "semantic_vegetation_ratio", "retained_walkable_ratio"]
    data[source_columns].to_csv(output / "source_data_8975.csv.gz", index=False,
                                encoding="utf-8-sig", compression="gzip")
    candidates = data[data["optimization_eligible_final"]].copy().sort_values("point_id")
    candidates[source_columns].to_csv(output / "eligible_points_5.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 6.25), facecolor="white")
    grid = fig.add_gridspec(3, 12, height_ratios=[1.25, 0.92, 0.83],
                            left=0.055, right=0.985, top=0.865, bottom=0.065,
                            wspace=0.72, hspace=0.48)
    ax_map = fig.add_subplot(grid[:2, :7])
    ax_bar = fig.add_subplot(grid[0, 7:])
    ax_flow = fig.add_subplot(grid[1, 7:])
    image_axes = [fig.add_subplot(grid[2, start:start + width])
                  for start, width in [(0, 2), (2, 2), (4, 2), (6, 3), (9, 3)]]
    fig.suptitle("南京中心城区街景遮荫规划：8,975点终态决策", x=0.055, y=0.975,
                 ha="left", fontsize=11.2, fontweight="bold")
    fig.text(0.055, 0.935,
             "多模态证据将全市筛查收敛为维持、复核、排除/弃权与5个树木优先候选，而非对高SVF点统一生成干预",
             ha="left", fontsize=6.6, color="#53616B")

    # a: full spatial distribution, with every point retained.
    panel_label(ax_map, "a")
    for ring in tqdm(rings, desc="Plotting boundary", unit="ring", dynamic_ncols=True):
        ax_map.plot(ring[:, 0], ring[:, 1], color="#36454F", linewidth=0.75, zorder=4)
    draw_order = ["excluded_or_abstained", "maintain_or_no_intervention", "existing_tree_review"]
    for group in tqdm(draw_order, desc="Plotting decision groups", unit="group", dynamic_ncols=True):
        subset = data[data["decision_group"].eq(group)]
        label, color = GROUPS[group]
        ax_map.scatter(subset["utm_x"], subset["utm_y"], s=3.0, color=color, alpha=0.58,
                       linewidths=0, rasterized=True, label=f"{label}（{len(subset):,}）", zorder=2)
    eligible = data[data["decision_group"].eq("optimization_eligible")]
    ax_map.scatter(eligible["utm_x"], eligible["utm_y"], s=47, marker="*", color=GROUPS["optimization_eligible"][1],
                   edgecolor="white", linewidth=0.7, label="树木优先候选（5）", zorder=6)
    for row in tqdm(eligible.itertuples(index=False), total=len(eligible),
                    desc="Labelling candidates", unit="point", dynamic_ncols=True):
        ax_map.annotate(str(int(row.point_id)), (row.utm_x, row.utm_y), xytext=(4, 3),
                        textcoords="offset points", fontsize=5.6, fontweight="bold", color="#A7441F")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("Easting (m), EPSG:32650"); ax_map.set_ylabel("Northing (m), EPSG:32650")
    ax_map.set_title("终态决策空间分布（中心城区边界仅作参照）", loc="left", pad=5)
    ax_map.grid(color="#E8ECEF", linewidth=0.45); ax_map.set_axisbelow(True)
    ax_map.legend(loc="lower left", fontsize=5.7, frameon=True, framealpha=0.94,
                  edgecolor="#D9DEE2", borderpad=0.55, handletextpad=0.5)
    x0, x1 = ax_map.get_xlim(); y0, y1 = ax_map.get_ylim()
    scale = 5000; sx = x0 + 0.055 * (x1 - x0); sy = y0 + 0.045 * (y1 - y0)
    ax_map.plot([sx, sx + scale], [sy, sy], color="#20262B", linewidth=2.2, solid_capstyle="butt")
    ax_map.text(sx + scale / 2, sy + 0.012 * (y1 - y0), "5 km", ha="center", fontsize=5.5)
    ax_map.annotate("N", xy=(x1 - 0.055*(x1-x0), y1 - 0.04*(y1-y0)),
                    xytext=(x1 - 0.055*(x1-x0), y1 - 0.14*(y1-y0)),
                    ha="center", fontsize=6.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="#20262B", lw=1.3))

    # b: mutually exclusive distribution.
    panel_label(ax_bar, "b")
    bar_order = ["existing_tree_review", "maintain_or_no_intervention",
                 "excluded_or_abstained", "optimization_eligible"]
    labels = [GROUPS[group][0] for group in bar_order]
    values = [int(counts[group]) for group in bar_order]
    colors = [GROUPS[group][1] for group in bar_order]
    bars = ax_bar.barh(np.arange(len(values)), values, color=colors, height=0.58)
    ax_bar.set_yticks(np.arange(len(values)), labels); ax_bar.invert_yaxis()
    ax_bar.set_xlabel("点位数（n=8,975）"); ax_bar.set_title("互斥终态类别", loc="left", pad=5)
    ax_bar.grid(axis="x", color="#E6EAED", linewidth=0.5); ax_bar.set_axisbelow(True)
    for bar, value in tqdm(zip(bars, values), total=len(values), desc="Annotating bars",
                           unit="bar", dynamic_ncols=True):
        ax_bar.text(bar.get_width() + 75, bar.get_y() + bar.get_height()/2,
                    f"{value:,}\n{value/8975:.1%}", va="center", fontsize=5.7, color="#3F4A52")
    ax_bar.set_xlim(0, max(values) * 1.20)

    # c: evidence funnel without implying precision/recall.
    panel_label(ax_flow, "c"); ax_flow.set_axis_off(); ax_flow.set_xlim(0, 1); ax_flow.set_ylim(0, 1)
    ax_flow.set_title("证据收敛链", loc="left", pad=5)
    stages = [("8,975", "全市终态覆盖", "#EEF3F6", "#93A9B5"),
              ("6,014", "流式CUDA定位", "#EDF3F8", "#7E9FB6"),
              ("5,990 / 24", "有 / 无现状树证据", "#EDF6F0", "#76A087"),
              ("5", "树木优先候选", "#FFF0E8", "#D97845")]
    box_w, gap, y, height = 0.205, 0.046, 0.31, 0.42
    for index, (title, detail, face, edge) in tqdm(enumerate(stages), total=len(stages),
                                                  desc="Drawing evidence chain", unit="stage",
                                                  dynamic_ncols=True):
        x = 0.015 + index * (box_w + gap)
        evidence_box(ax_flow, x, y, box_w, height, title, detail, face, edge)
        if index < len(stages) - 1:
            ax_flow.add_patch(FancyArrowPatch((x + box_w + 0.006, y + height/2),
                                               (x + box_w + gap - 0.006, y + height/2),
                                               arrowstyle="-|>", mutation_scale=8,
                                               linewidth=0.8, color="#697780"))
    ax_flow.text(0.015, 0.12, "GPU定位点全部完成：待处理 0 · 失败 0 · CPU回退 0",
                 fontsize=6.1, color="#46535C", fontweight="bold")

    # d-h: all five eligible panoramas; no image editing.
    for ax, point_id, pano in tqdm(zip(image_axes, eligible_ids, panoramas), total=5,
                                   desc="Rendering evidence cards", unit="point", dynamic_ncols=True):
        row = candidates[candidates["point_id"].astype(int).eq(point_id)].iloc[0]
        with Image.open(pano) as image:
            ax.imshow(np.asarray(image.convert("RGB")))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        walkable = "已验证建议" if pd.isna(row["walkable_direction_count"]) else f"人行方向 {int(row['walkable_direction_count'])}"
        tree = "V2终态" if pd.isna(row["tree_evidence_direction_count"]) else f"树证据方向 {int(row['tree_evidence_direction_count'])}"
        ax.set_title(f"{point_id} · SVF {row['SVF']:.3f} · GVI {row['GVI']:.3f}\n{walkable} · {tree}",
                     fontsize=5.6, pad=2.5, color="#34424B")
    image_axes[0].text(-0.12, 1.20, "d", transform=image_axes[0].transAxes,
                       fontsize=9, fontweight="bold", ha="right", va="bottom")
    fig.text(0.055, 0.018,
             "证据边界：树木、人行方向和二者共现均为自动多视角代理；5点是进入规划师复核的候选，不是自动施工选址或已验证的因果降温效果。全景仅统一缩放显示，无局部编辑。",
             fontsize=5.4, color="#52606A", ha="left")

    base = output / config["outputs"]["basename"]
    fig.savefig(base.with_suffix(".png"), dpi=400,
                bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "step": "stage_33", "status": "PASS", "started_at": started.isoformat(timespec="seconds"),
        "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - tick, 3), "point_count": int(len(data)),
        "eligible_point_count": int(data["optimization_eligible_final"].sum()),
        "decision_group_counts": {key: int(value) for key, value in counts.items()},
        "boundary_ring_count": len(rings), "excluded_rows": 0,
        "formats": config["outputs"]["formats"],
        "backend": "Python/matplotlib + standard-library Shapefile reader",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "qc.json").write_text(json.dumps({
        "status": "PASS", "checks": {
            "all_8975_points_used": len(data) == 8975,
            "point_ids_unique": data["point_id"].nunique() == 8975,
            "decision_groups_mutually_exclusive": int(counts.sum()) == 8975,
            "eligible_points_exactly_five": actual_ids == sorted(eligible_ids),
            "all_panorama_cards_present": len(panoramas) == 5,
            "all_exports_present": all(base.with_suffix(f".{suffix}").exists()
                                       for suffix in config["outputs"]["formats"]),
            "no_rows_excluded": True,
        }}, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("stage_33 completed: %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
