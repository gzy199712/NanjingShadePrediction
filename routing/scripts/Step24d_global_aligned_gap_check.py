"""Step24d: global check-only scan for high-confidence aligned road gaps."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "routing/configs/global_aligned_gap_check.yaml"
FCLASS_RANK = {
    "footway": 1, "path": 1, "steps": 1, "pedestrian": 1, "cycleway": 1,
    "living_street": 2, "service": 2, "track": 2, "residential": 3,
    "unclassified": 3, "tertiary": 4, "secondary": 5, "primary": 6,
}


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve(value: str) -> Path:
    return (ROOT / value).resolve()


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "t", "yes", "y"}


def layer(value: Any) -> float:
    try:
        return float(value) if not pd.isna(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(math.degrees(math.acos(np.clip(np.dot(first, second), -1.0, 1.0))))


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def strict_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    return ((o1 > 1e-9 and o2 < -1e-9) or (o1 < -1e-9 and o2 > 1e-9)) and ((o3 > 1e-9 and o4 < -1e-9) or (o3 < -1e-9 and o4 > 1e-9))


def node_edge_ids(node: int, data: dict[str, np.ndarray]) -> list[str]:
    start, end = int(data["offsets"][node]), int(data["offsets"][node + 1])
    indexes = data["adjacency_edges"][start:end]
    return sorted(set(str(value) for value in data["original_edge_ids"][indexes].tolist()))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in tqdm(rows, total=len(rows), desc=f"Writing {path.name}", unit="row", dynamic_ncols=True):
            writer.writerow(row)


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step24d_global_gap_check")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="w" if overwrite else "a", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Global aligned degree-one road-gap check")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = resolve(config["outputs"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(resolve(config["outputs"]["log"]), args.overwrite)
    started_at = now_text()
    failures: list[dict[str, Any]] = []
    try:
        graph_npz = np.load(resolve(config["input"]["formal_graph"]), allow_pickle=False)
        data = {name: graph_npz[name] for name in graph_npz.files}
        xy, edge_u, edge_v = data["node_xy"], data["edge_u"], data["edge_v"]
        degree = np.diff(data["offsets"])
        endpoints = np.flatnonzero(degree == 1)
        logger.info("Loaded graph: %s nodes, %s edges, %s degree-one endpoints", len(xy), len(edge_u), len(endpoints))

        frame = pd.read_csv(resolve(config["input"]["road_classification"]), dtype={"edge_id": str}, low_memory=False)
        metadata = frame.set_index("edge_id").to_dict("index")
        maximum_gap = float(config["geometry"]["maximum_gap_m"])
        minimum_gap = float(config["geometry"]["minimum_gap_m"])
        review_angle = float(config["geometry"]["review_alignment_deg"])
        formal_angle = float(config["geometry"]["formal_alignment_deg"])
        max_rank_difference = int(config["network"]["maximum_fclass_rank_difference"])

        endpoint_tree = cKDTree(xy[endpoints])
        pairs = sorted(endpoint_tree.query_pairs(maximum_gap))
        logger.info("Spatial endpoint pairs within %.1f m: %s", maximum_gap, len(pairs))
        preliminary: list[dict[str, Any]] = []
        for first_local, second_local in tqdm(pairs, total=len(pairs), desc="Filtering aligned endpoint pairs", unit="pair", dynamic_ncols=True):
            first, second = int(endpoints[first_local]), int(endpoints[second_local])
            vector = xy[second] - xy[first]
            gap = float(np.linalg.norm(vector))
            if gap <= minimum_gap:
                continue
            first_neighbor = int(data["adjacency_nodes"][int(data["offsets"][first])])
            second_neighbor = int(data["adjacency_nodes"][int(data["offsets"][second])])
            first_out = xy[first] - xy[first_neighbor]; first_out /= np.linalg.norm(first_out)
            second_out = xy[second] - xy[second_neighbor]; second_out /= np.linalg.norm(second_out)
            first_angle = angle_deg(first_out, vector / gap)
            second_angle = angle_deg(second_out, -vector / gap)
            if first_angle > review_angle or second_angle > review_angle:
                continue
            first_ids, second_ids = node_edge_ids(first, data), node_edge_ids(second, data)
            first_id, second_id = first_ids[0], second_ids[0]
            if first_id.startswith(("SNAP_", "EXT_GAP_")) or second_id.startswith(("SNAP_", "EXT_GAP_")):
                continue
            first_meta, second_meta = metadata.get(first_id, {}), metadata.get(second_id, {})
            first_class, second_class = clean(first_meta.get("fclass")), clean(second_meta.get("fclass"))
            rank_difference = abs(FCLASS_RANK.get(first_class, 99) - FCLASS_RANK.get(second_class, 99))
            structure_compatible = (
                clean(first_meta.get("bridge")) == clean(second_meta.get("bridge"))
                and clean(first_meta.get("tunnel")) == clean(second_meta.get("tunnel"))
                and math.isclose(layer(first_meta.get("layer")), layer(second_meta.get("layer")), abs_tol=1e-9)
                and truthy(first_meta.get("grade_separated")) == truthy(second_meta.get("grade_separated"))
            )
            first_name, second_name = clean(first_meta.get("name")), clean(second_meta.get("name"))
            first_ref, second_ref = clean(first_meta.get("ref")), clean(second_meta.get("ref"))
            identity_compatible = not first_name or not second_name or first_name == second_name or (first_ref and first_ref == second_ref)
            preliminary.append({
                "source_node": first, "target_node": second,
                "source_x": float(xy[first, 0]), "source_y": float(xy[first, 1]),
                "target_x": float(xy[second, 0]), "target_y": float(xy[second, 1]),
                "gap_m": gap, "source_angle_deg": first_angle, "target_angle_deg": second_angle,
                "source_edge_id": first_id, "target_edge_id": second_id,
                "source_fclass": first_class, "target_fclass": second_class,
                "source_name": first_name, "target_name": second_name,
                "source_ref": first_ref, "target_ref": second_ref,
                "fclass_rank_difference": rank_difference,
                "structure_compatible": structure_compatible, "identity_compatible": identity_compatible,
                "source_component": int(data["component_all"][first]), "target_component": int(data["component_all"][second]),
                "source_incident_edge_index": int(data["adjacency_edges"][int(data["offsets"][first])]),
                "target_incident_edge_index": int(data["adjacency_edges"][int(data["offsets"][second])]),
            })
        logger.info("Direction-review candidates: %s", len(preliminary))

        cell = float(config["geometry"]["spatial_grid_m"])
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for edge in tqdm(range(len(edge_u)), total=len(edge_u), desc="Indexing graph segments for crossing checks", unit="edge", dynamic_ncols=True):
            a, b = xy[edge_u[edge]], xy[edge_v[edge]]
            min_cell = np.floor(np.minimum(a, b) / cell).astype(int); max_cell = np.floor(np.maximum(a, b) / cell).astype(int)
            for gx in range(min_cell[0], max_cell[0] + 1):
                for gy in range(min_cell[1], max_cell[1] + 1):
                    grid[(gx, gy)].append(edge)

        strict_rows = [row for row in preliminary if row["source_angle_deg"] <= formal_angle and row["target_angle_deg"] <= formal_angle and row["structure_compatible"] and row["identity_compatible"] and row["fclass_rank_difference"] <= max_rank_difference]
        edge_count = len(edge_u)
        rows_index = np.repeat(np.arange(len(xy)), degree)
        weights = data["lengths"][data["adjacency_edges"]]
        matrix = csr_matrix((weights, (rows_index, data["adjacency_nodes"])), shape=(len(xy), len(xy)))
        for row in tqdm(preliminary, total=len(preliminary), desc="Evaluating crossings and network detours", unit="candidate", dynamic_ncols=True):
            a = np.array([row["source_x"], row["source_y"]]); b = np.array([row["target_x"], row["target_y"]])
            min_cell = np.floor(np.minimum(a, b) / cell).astype(int); max_cell = np.floor(np.maximum(a, b) / cell).astype(int)
            nearby: set[int] = set()
            for gx in range(min_cell[0], max_cell[0] + 1):
                for gy in range(min_cell[1], max_cell[1] + 1):
                    nearby.update(grid.get((gx, gy), []))
            excluded = {row["source_incident_edge_index"], row["target_incident_edge_index"]}
            crossings = 0
            for edge in nearby - excluded:
                if strict_intersection(a, b, xy[edge_u[edge]], xy[edge_v[edge]]):
                    crossings += 1
            row["interior_crossing_count"] = crossings
            same_component = row["source_component"] == row["target_component"]
            row["same_component_before"] = same_component
            if row in strict_rows and same_component:
                distances = dijkstra(matrix, directed=False, indices=int(row["source_node"]), limit=max(float(config["network"]["minimum_detour_saving_m"]) + row["gap_m"], row["gap_m"] * float(config["network"]["minimum_detour_ratio"])))
                network_distance = float(distances[int(row["target_node"])])
            else:
                network_distance = math.inf if not same_component else math.nan
            row["existing_network_distance_m"] = network_distance
            row["network_detour_ratio"] = network_distance / row["gap_m"] if np.isfinite(network_distance) else math.inf
            row["estimated_distance_saving_m"] = network_distance - row["gap_m"] if np.isfinite(network_distance) else math.inf
            network_material = (not same_component) or (
                np.isfinite(network_distance)
                and row["network_detour_ratio"] >= float(config["network"]["minimum_detour_ratio"])
                and row["estimated_distance_saving_m"] >= float(config["network"]["minimum_detour_saving_m"])
            )
            row["high_confidence"] = bool(
                row["source_angle_deg"] <= formal_angle and row["target_angle_deg"] <= formal_angle
                and row["structure_compatible"] and row["identity_compatible"]
                and row["fclass_rank_difference"] <= max_rank_difference
                and crossings == 0 and network_material
            )
            row["decision"] = "eligible_for_versioned_formal_repair" if row["high_confidence"] else "retain_for_review_or_reject"

        from osgeo import ogr
        ogr.UseExceptions()
        boundary_ds = ogr.Open(str(resolve(config["input"]["center_boundary"])), 0)
        boundary_layer = boundary_ds.GetLayer(0)
        boundary_geometry = None
        for feature in boundary_layer:
            geometry = feature.GetGeometryRef()
            if geometry is None:
                continue
            boundary_geometry = geometry.Clone() if boundary_geometry is None else boundary_geometry.Union(geometry)
        if boundary_geometry is None:
            raise RuntimeError("Center boundary has no valid geometry")
        motor_ds = ogr.Open(str(resolve(config["input"]["formal_network_gdb"])), 0)
        motor_layer = motor_ds.GetLayerByName(config["input"]["motor_only_layer"])
        if motor_layer is None:
            raise KeyError(config["input"]["motor_only_layer"])
        clearance = float(config["geometry"]["motor_only_clearance_m"])
        for row in tqdm(preliminary, total=len(preliminary), desc="Checking boundary and motor-only conflicts", unit="candidate", dynamic_ncols=True):
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint_2D(row["source_x"], row["source_y"]); line.AddPoint_2D(row["target_x"], row["target_y"])
            within = line.Intersection(boundary_geometry)
            row["center_boundary_coverage_ratio"] = float(within.Length() / row["gap_m"]) if within is not None and not within.IsEmpty() else 0.0
            envelope = line.Buffer(clearance).GetEnvelope()
            motor_layer.SetSpatialFilterRect(envelope[0], envelope[2], envelope[1], envelope[3])
            conflict_ids: list[str] = []
            buffer = line.Buffer(clearance)
            for feature in motor_layer:
                geometry = feature.GetGeometryRef()
                if geometry is not None and buffer.Intersects(geometry):
                    conflict_ids.append(clean(feature.GetField("edge_id")))
            motor_layer.ResetReading(); motor_layer.SetSpatialFilter(None)
            row["motor_only_conflict_count"] = len(set(conflict_ids))
            row["motor_only_conflict_edge_ids"] = ";".join(sorted(set(conflict_ids)))
            row["high_confidence"] = bool(
                row["high_confidence"]
                and row["center_boundary_coverage_ratio"] >= 0.999
                and row["motor_only_conflict_count"] == 0
            )
            strong_identity = bool(
                (row["source_name"] and row["source_name"] == row["target_name"])
                or (row["source_ref"] and row["source_ref"] == row["target_ref"])
            )
            excluded_classes = set(config["network"].get("automatic_excluded_fclasses", []))
            automatic_scope = (
                row["source_fclass"] not in excluded_classes
                and row["target_fclass"] not in excluded_classes
                and (
                    row["gap_m"] <= float(config["network"]["maximum_unnamed_automatic_gap_m"])
                    or strong_identity
                )
            )
            row["strong_identity_evidence"] = strong_identity
            row["automatic_scope_eligible"] = automatic_scope
            row["high_confidence"] = bool(row["high_confidence"] and automatic_scope)
            row["decision"] = "eligible_for_component_pair_dedup" if row["high_confidence"] else "retain_for_review_or_reject"
        boundary_ds = None; motor_ds = None

        eligible = [row for row in preliminary if row["high_confidence"]]
        eligible.sort(key=lambda row: (row["gap_m"], row["source_angle_deg"] + row["target_angle_deg"]))
        selected_pairs: set[tuple[int, int]] = set()
        for row in eligible:
            component_pair = tuple(sorted((int(row["source_component"]), int(row["target_component"]))))
            row["component_pair"] = f"{component_pair[0]}-{component_pair[1]}"
            if bool(config["network"].get("one_connector_per_component_pair_per_round", True)) and component_pair in selected_pairs:
                row["high_confidence"] = False
                row["decision"] = "deferred_duplicate_component_pair"
            else:
                selected_pairs.add(component_pair)
                row["decision"] = "eligible_for_versioned_formal_repair"

        preliminary.sort(key=lambda row: (not row["high_confidence"], row["gap_m"], row["source_node"], row["target_node"]))
        for index, row in enumerate(preliminary, start=1):
            row["candidate_id"] = f"global_gap_{index:05d}"
        approved = [row for row in preliminary if row["high_confidence"]]
        fields = [
            "candidate_id", "source_node", "target_node", "source_x", "source_y", "target_x", "target_y", "gap_m",
            "source_angle_deg", "target_angle_deg", "source_edge_id", "target_edge_id", "source_fclass", "target_fclass",
            "source_name", "target_name", "source_ref", "target_ref", "fclass_rank_difference", "structure_compatible",
            "identity_compatible", "source_component", "target_component", "component_pair", "same_component_before", "interior_crossing_count",
            "center_boundary_coverage_ratio", "motor_only_conflict_count", "motor_only_conflict_edge_ids",
            "strong_identity_evidence", "automatic_scope_eligible",
            "existing_network_distance_m", "network_detour_ratio", "estimated_distance_saving_m", "high_confidence", "decision",
        ]
        write_csv(resolve(config["outputs"]["all_candidates"]), preliminary, fields)
        write_csv(resolve(config["outputs"]["high_confidence"]), approved, fields)
        write_csv(resolve(config["outputs"]["failed"]), failures, ["record_id", "stage", "error_message"])

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        figure, axis = plt.subplots(figsize=(12, 10))
        sample_step = max(1, len(edge_u) // 100000)
        sampled = np.arange(0, len(edge_u), sample_step)
        segments = np.stack((xy[edge_u[sampled]], xy[edge_v[sampled]]), axis=1)
        axis.add_collection(LineCollection(segments, colors="#d9d9d9", linewidths=0.25, alpha=0.5))
        for row in tqdm(approved, total=len(approved), desc="Plotting global gap shortlist", unit="candidate", dynamic_ncols=True):
            axis.plot([row["source_x"], row["target_x"]], [row["source_y"], row["target_y"]], color="#d73027", linewidth=1.6)
        axis.autoscale(); axis.set_aspect("equal", adjustable="box"); axis.set_title(f"Global high-confidence aligned gaps (n={len(approved)})")
        axis.set_xlabel("Easting (m), EPSG:32650"); axis.set_ylabel("Northing (m), EPSG:32650"); axis.grid(alpha=0.1)
        figure.savefig(resolve(config["outputs"]["overview_figure"]), dpi=220, bbox_inches="tight"); plt.close(figure)

        columns = 3
        rows = max(1, math.ceil(len(approved) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(15, 4.8 * rows), squeeze=False)
        for axis, row in zip(axes.flat, approved, strict=False):
            center = np.array([(row["source_x"] + row["target_x"]) / 2.0, (row["source_y"] + row["target_y"]) / 2.0])
            radius = max(120.0, row["gap_m"] * 2.0)
            low, high = center - radius, center + radius
            mask = (
                (xy[edge_u, 0] >= low[0]) & (xy[edge_u, 0] <= high[0]) & (xy[edge_u, 1] >= low[1]) & (xy[edge_u, 1] <= high[1])
            ) | (
                (xy[edge_v, 0] >= low[0]) & (xy[edge_v, 0] <= high[0]) & (xy[edge_v, 1] >= low[1]) & (xy[edge_v, 1] <= high[1])
            )
            local = np.flatnonzero(mask)
            local_segments = np.stack((xy[edge_u[local]], xy[edge_v[local]]), axis=1)
            axis.add_collection(LineCollection(local_segments, colors="#bdbdbd", linewidths=0.7, alpha=0.75))
            axis.plot([row["source_x"], row["target_x"]], [row["source_y"], row["target_y"]], color="#d73027", linewidth=2.5, linestyle="--")
            axis.scatter([row["source_x"], row["target_x"]], [row["source_y"], row["target_y"]], c=["#2166ac", "#fdae61"], s=28, zorder=4)
            axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1]); axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{row['candidate_id']} | {row['gap_m']:.1f} m | {row['source_fclass']}")
            axis.grid(alpha=0.12)
        for axis in axes.flat[len(approved):]:
            axis.axis("off")
        figure.suptitle("Step24d high-confidence local topology review", fontsize=16)
        figure.tight_layout()
        figure.savefig(resolve(config["outputs"]["contact_sheet"]), dpi=200, bbox_inches="tight"); plt.close(figure)

        summary = {
            "step": "24d", "mode": "check", "started_at": started_at, "finished_at": now_text(),
            "graph_node_count": len(xy), "graph_edge_count": edge_count, "degree_one_endpoint_count": len(endpoints),
            "endpoint_pair_count_within_100m": len(pairs), "direction_review_candidate_count": len(preliminary),
            "high_confidence_candidate_count": len(approved),
            "same_component_high_confidence_count": sum(row["same_component_before"] for row in approved),
            "cross_component_high_confidence_count": sum(not row["same_component_before"] for row in approved),
            "total_candidate_length_m": sum(row["gap_m"] for row in approved),
            "formal_network_modified": False, "failed_record_count": len(failures),
            "status": "CHECK_COMPLETE_HIGH_CONFIDENCE_SHORTLIST_READY",
        }
        resolve(config["outputs"]["summary"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        report = [
            "# Step24d 全路网共线断点检查报告", "", f"完成时间：{summary['finished_at']}", "",
            f"- degree-1端点：{len(endpoints):,}", f"- 100米内端点对：{len(pairs):,}",
            f"- 双端方向偏差不超过15°的复核候选：{len(preliminary):,}",
            f"- 通过5°方向、结构、道路类别、名称/编号、无内部交叉及网络绕行门槛的高置信候选：{len(approved):,}",
            f"- 候选总长度：{summary['total_candidate_length_m']:.2f}米", "",
            "本步只生成候选，没有修改正式路网。长距离断点仍采用显式连接、版本化备份和下游全量重建。",
            "", "`READY_FOR_GLOBAL_VERSIONED_REPAIR_REVIEW = TRUE`",
        ]
        report_path = resolve(config["outputs"]["report"]); report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        logger.info("Step24d complete: %s", json.dumps(summary, ensure_ascii=False))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        logger.error("Step24d failed: %s", error); logger.error(traceback.format_exc())
        write_csv(resolve(config["outputs"]["failed"]), [{"record_id": "step24d", "stage": "fatal", "error_message": str(error)}], ["record_id", "stage", "error_message"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
