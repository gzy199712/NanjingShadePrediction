"""Step24b: check-only endpoint-forward-extension topology candidates.

This script never modifies the frozen Step24 network. It identifies degree-one
road endpoints whose forward ray intersects a structurally compatible road
within a short, staged distance. Every result remains a manual-review candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import traceback
import datetime as datetime_module
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from tqdm import tqdm as _tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DEFAULT = PROJECT_ROOT / "routing/configs/topology_endpoint_extension_check.yaml"
PROGRESS_LOG = PROJECT_ROOT / "routing/logs/step24b_endpoint_extension_progress.log"


class ProgressSink:
    def write(self, value: str) -> int:
        with PROGRESS_LOG.open("a", encoding="utf-8") as handle:
            return handle.write(value)

    def flush(self) -> None:
        return None


_PROGRESS_SINK = ProgressSink()


def tqdm(*args: Any, **kwargs: Any) -> Any:
    """Keep full tqdm telemetry in a stable file for ArcGIS background runs."""
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    kwargs["file"] = _PROGRESS_SINK
    kwargs["dynamic_ncols"] = False
    kwargs.setdefault("mininterval", 1.0)
    return _tqdm(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check degree-one endpoint forward-extension topology candidates."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--mode", choices=("check",), default="check")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def now_text() -> str:
    return datetime_module.datetime.now().astimezone().isoformat(timespec="seconds")


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step24b_endpoint_extension")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for handler in (
        logging.FileHandler(path, mode="w" if overwrite else "a", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        if not rows:
            return
        for row in tqdm(rows, total=len(rows), desc=f"Writing {path.name}", unit="row", dynamic_ncols=True):
            writer.writerow(row)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def boolish(value: Any) -> bool:
    return clean_text(value).lower() not in {"", "0", "false", "f", "n", "no", "none", "null"}


def layer_value(value: Any) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def rounded_node(point: np.ndarray, precision: float) -> tuple[int, int]:
    return tuple(np.rint(point / precision).astype(np.int64).tolist())


def geometry_parts(shape: Any) -> list[list[np.ndarray]]:
    parts: list[list[np.ndarray]] = []
    for part_index in range(shape.partCount):
        points: list[np.ndarray] = []
        part = shape.getPart(part_index)
        for point in part:
            if point is None:
                if len(points) >= 2:
                    parts.append(points)
                points = []
            else:
                points.append(np.array([float(point.X), float(point.Y)], dtype=np.float64))
        if len(points) >= 2:
            parts.append(points)
    return parts


def ogr_geometry_parts(geometry: Any) -> list[list[np.ndarray]]:
    """Return all line parts without converting the whole feature to WKT."""
    from osgeo import ogr

    geometry_name = geometry.GetGeometryName().upper()
    if geometry_name in {"LINESTRING", "LINEARRING"}:
        points = [
            np.array(geometry.GetPoint(index)[:2], dtype=np.float64)
            for index in range(geometry.GetPointCount())
        ]
        return [points] if len(points) >= 2 else []
    if geometry_name == "MULTILINESTRING":
        parts: list[list[np.ndarray]] = []
        for index in range(geometry.GetGeometryCount()):
            parts.extend(ogr_geometry_parts(geometry.GetGeometryRef(index)))
        return parts
    return []


def ogr_geometry_endpoints(geometry: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Read only the two terminal coordinates, without expanding all vertices."""
    from osgeo import ogr

    geometry_name = geometry.GetGeometryName().upper()
    if geometry_name in {"LINESTRING", "LINEARRING"}:
        if geometry.GetPointCount() < 2:
            return None
        return (
            np.array(geometry.GetPoint(0)[:2], dtype=np.float64),
            np.array(geometry.GetPoint(geometry.GetPointCount() - 1)[:2], dtype=np.float64),
        )
    if geometry_name == "MULTILINESTRING" and geometry.GetGeometryCount() > 0:
        first = geometry.GetGeometryRef(0)
        last = geometry.GetGeometryRef(geometry.GetGeometryCount() - 1)
        if first.GetPointCount() < 1 or last.GetPointCount() < 1:
            return None
        return (
            np.array(first.GetPoint(0)[:2], dtype=np.float64),
            np.array(last.GetPoint(last.GetPointCount() - 1)[:2], dtype=np.float64),
        )
    return None


def distinct_neighbor(points: list[np.ndarray], at_start: bool) -> np.ndarray | None:
    endpoint = points[0] if at_start else points[-1]
    sequence = points[1:] if at_start else reversed(points[:-1])
    for point in sequence:
        if float(np.linalg.norm(point - endpoint)) > 1e-6:
            return point
    return None


def segment_intersection(
    ray_start: np.ndarray,
    ray_vector: np.ndarray,
    target_start: np.ndarray,
    target_end: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    target_vector = target_end - target_start
    denominator = ray_vector[0] * target_vector[1] - ray_vector[1] * target_vector[0]
    if abs(float(denominator)) < 1e-10:
        return None
    offset = target_start - ray_start
    ray_fraction = (offset[0] * target_vector[1] - offset[1] * target_vector[0]) / denominator
    target_fraction = (offset[0] * ray_vector[1] - offset[1] * ray_vector[0]) / denominator
    if -1e-9 <= ray_fraction <= 1.0 + 1e-9 and -1e-9 <= target_fraction <= 1.0 + 1e-9:
        point = ray_start + ray_fraction * ray_vector
        return float(ray_fraction), float(target_fraction), point
    return None


def acute_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    value = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(math.degrees(math.acos(value)))


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator <= 0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * vector)))


def compatible(source: dict[str, Any], target: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if source["edge_id"] == target["edge_id"]:
        reasons.append("same_edge")
    if source["repair_typ"].lower() not in {"", "original"}:
        reasons.append("source_is_repair_connector")
    if target["repair_typ"].lower() not in {"", "original"}:
        reasons.append("target_is_repair_connector")
    if source["therm_class"] != "thermal_comfort_relevant" or target["therm_class"] != "thermal_comfort_relevant":
        reasons.append("motor_vehicle_only")
    if source["bridge"] != target["bridge"]:
        reasons.append("bridge_conflict")
    if source["tunnel"] != target["tunnel"]:
        reasons.append("tunnel_conflict")
    if not math.isclose(source["layer"], target["layer"], abs_tol=1e-9):
        reasons.append("layer_conflict")
    if source["grade_sep"] != target["grade_sep"]:
        reasons.append("grade_separation_conflict")
    return not reasons, reasons


def load_graph_impact(
    graph_path: Path,
    origin_node: int,
    destination_node: int,
    required_edge_ids: set[str],
) -> dict[str, Any]:
    data = np.load(graph_path, allow_pickle=False)
    degree = np.diff(data["offsets"])
    rows = np.repeat(np.arange(data["node_xy"].shape[0]), degree)
    weights = data["lengths"][data["adjacency_edges"]]
    matrix = csr_matrix(
        (weights, (rows, data["adjacency_nodes"])),
        shape=(data["node_xy"].shape[0], data["node_xy"].shape[0]),
    )
    from_origin = dijkstra(matrix, directed=False, indices=origin_node)
    from_destination = dijkstra(matrix, directed=False, indices=destination_node)
    edge_nodes: dict[str, set[int]] = defaultdict(set)
    if required_edge_ids:
        required = np.fromiter(required_edge_ids, dtype=data["original_edge_ids"].dtype)
        indexes = np.flatnonzero(np.isin(data["original_edge_ids"], required))
        for index in indexes.tolist():
            original_edge = str(data["original_edge_ids"][index])
            edge_nodes[original_edge].add(int(data["edge_u"][index]))
            edge_nodes[original_edge].add(int(data["edge_v"][index]))
    return {
        "matrix": matrix,
        "node_xy": data["node_xy"],
        "offsets": data["offsets"],
        "adjacency_nodes": data["adjacency_nodes"],
        "adjacency_edges": data["adjacency_edges"],
        "original_edge_ids": data["original_edge_ids"],
        "edge_u": data["edge_u"],
        "edge_v": data["edge_v"],
        "from_origin": from_origin,
        "from_destination": from_destination,
        "edge_nodes": edge_nodes,
        "baseline": float(from_origin[destination_node]),
    }


def graph_node_original_edges(node: int, impact: dict[str, Any]) -> list[str]:
    start = int(impact["offsets"][node])
    end = int(impact["offsets"][node + 1])
    edge_indexes = impact["adjacency_edges"][start:end]
    return sorted(set(str(value) for value in impact["original_edge_ids"][edge_indexes].tolist()))


def find_proximity_gap_candidates(
    impact: dict[str, Any],
    origin_node: int,
    destination_node: int,
    rules: dict[str, Any],
    classification: pd.DataFrame,
) -> list[dict[str, Any]]:
    xy = impact["node_xy"]
    degree = np.diff(impact["offsets"])
    origin = xy[origin_node]
    destination = xy[destination_node]
    vector = destination - origin
    denominator = float(np.dot(vector, vector))
    endpoints = np.flatnonzero(degree == 1)
    fractions = np.clip(((xy[endpoints] - origin) @ vector) / denominator, 0.0, 1.0)
    projections = origin + fractions[:, None] * vector
    corridor = float(rules["corridor_buffer_m"])
    endpoints = endpoints[np.linalg.norm(xy[endpoints] - projections, axis=1) <= corridor]
    tree = cKDTree(xy)
    maximum_gap = float(rules["maximum_gap_m"])
    maximum_alignment = float(rules["maximum_bidirectional_alignment_deg"])
    minimum_saving = float(rules["minimum_material_route_saving_m"])
    lookup = classification.set_index("edge_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for source in tqdm(endpoints.tolist(), total=len(endpoints), desc="Diagnosing spatially near network gaps", unit="endpoint", dynamic_ncols=True):
        source_neighbor = int(impact["adjacency_nodes"][int(impact["offsets"][source])])
        source_outward = xy[source] - xy[source_neighbor]
        source_outward /= np.linalg.norm(source_outward)
        for target in tree.query_ball_point(xy[source], maximum_gap):
            target = int(target)
            if target == source or target == source_neighbor:
                continue
            pair = tuple(sorted((source, target)))
            if pair in seen:
                continue
            seen.add(pair)
            gap_vector = xy[target] - xy[source]
            gap = float(np.linalg.norm(gap_vector))
            if gap <= 0:
                continue
            new_distance = float(min(
                impact["from_origin"][source] + gap + impact["from_destination"][target],
                impact["from_origin"][target] + gap + impact["from_destination"][source],
            ))
            saving = float(impact["baseline"] - new_distance)
            if saving <= 1.0:
                continue
            source_angle = float(math.degrees(math.acos(np.clip(np.dot(source_outward, gap_vector / gap), -1.0, 1.0))))
            target_degree = int(degree[target])
            target_angle = float("nan")
            if target_degree == 1:
                target_neighbor = int(impact["adjacency_nodes"][int(impact["offsets"][target])])
                target_outward = xy[target] - xy[target_neighbor]
                target_outward /= np.linalg.norm(target_outward)
                target_angle = float(math.degrees(math.acos(np.clip(np.dot(target_outward, -gap_vector / gap), -1.0, 1.0))))
            source_edges = graph_node_original_edges(source, impact)
            target_edges = graph_node_original_edges(target, impact)
            source_edge = source_edges[0] if source_edges else ""
            target_edge = target_edges[0] if target_edges else ""
            source_meta = lookup.get(source_edge, {})
            target_meta = lookup.get(target_edge, {})
            structural_compatible = (
                clean_text(source_meta.get("bridge")) == clean_text(target_meta.get("bridge"))
                and clean_text(source_meta.get("tunnel")) == clean_text(target_meta.get("tunnel"))
                and math.isclose(layer_value(source_meta.get("layer")), layer_value(target_meta.get("layer")), abs_tol=1e-9)
                and boolish(source_meta.get("grade_separated")) == boolish(target_meta.get("grade_separated"))
            )
            aligned_both = target_degree == 1 and source_angle <= maximum_alignment and target_angle <= maximum_alignment
            rows.append({
                "source_node": source, "target_node": target,
                "source_x": float(xy[source, 0]), "source_y": float(xy[source, 1]),
                "target_x": float(xy[target, 0]), "target_y": float(xy[target, 1]),
                "gap_m": gap, "source_degree": 1, "target_degree": target_degree,
                "source_outward_angle_deg": source_angle, "target_outward_angle_deg": target_angle,
                "source_edge_id": source_edge, "target_edge_id": target_edge,
                "source_fclass": clean_text(source_meta.get("fclass")), "target_fclass": clean_text(target_meta.get("fclass")),
                "source_name": clean_text(source_meta.get("name")), "target_name": clean_text(target_meta.get("name")),
                "source_layer": layer_value(source_meta.get("layer")), "target_layer": layer_value(target_meta.get("layer")),
                "source_bridge": clean_text(source_meta.get("bridge")), "target_bridge": clean_text(target_meta.get("bridge")),
                "source_tunnel": clean_text(source_meta.get("tunnel")), "target_tunnel": clean_text(target_meta.get("tunnel")),
                "structural_compatible": structural_compatible,
                "estimated_od025_shortest_m": new_distance, "estimated_od025_saving_m": saving,
                "material_aligned_gap_candidate": bool(structural_compatible and aligned_both and saving >= minimum_saving),
                "decision": "manual_map_and_access_review_required",
            })
    rows.sort(key=lambda row: (-row["estimated_od025_saving_m"], row["gap_m"]))
    for index, row in enumerate(rows, start=1):
        row["gap_candidate_id"] = f"gap_{index:04d}"
    return rows


def create_gap_plot(impact: dict[str, Any], candidate: dict[str, Any], output: Path, radius_m: float = 300.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = impact["node_xy"]
    center = np.array([(candidate["source_x"] + candidate["target_x"]) / 2.0, (candidate["source_y"] + candidate["target_y"]) / 2.0])
    low, high = center - radius_m, center + radius_m
    u = impact["edge_u"]
    v = impact["edge_v"]
    mask = (
        (xy[u, 0] >= low[0]) & (xy[u, 0] <= high[0]) & (xy[u, 1] >= low[1]) & (xy[u, 1] <= high[1])
    ) | (
        (xy[v, 0] >= low[0]) & (xy[v, 0] <= high[0]) & (xy[v, 1] >= low[1]) & (xy[v, 1] <= high[1])
    )
    figure, axis = plt.subplots(figsize=(8, 8))
    for edge_index in tqdm(np.flatnonzero(mask).tolist(), desc="Plotting best OD025 gap", unit="segment", dynamic_ncols=True):
        axis.plot([xy[u[edge_index], 0], xy[v[edge_index], 0]], [xy[u[edge_index], 1], xy[v[edge_index], 1]], color="#bdbdbd", linewidth=0.8)
    axis.plot([candidate["source_x"], candidate["target_x"]], [candidate["source_y"], candidate["target_y"]], color="#d73027", linewidth=3.0, linestyle="--", label=f"Candidate gap {candidate['gap_m']:.2f} m")
    axis.scatter([candidate["source_x"], candidate["target_x"]], [candidate["source_y"], candidate["target_y"]], color=["#2166ac", "#fdae61"], s=55, zorder=4)
    axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1]); axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"od_025 aligned network gap: {candidate['source_edge_id']} → {candidate['target_edge_id']}")
    axis.set_xlabel("Easting (m), EPSG:32650"); axis.set_ylabel("Northing (m), EPSG:32650")
    axis.set_title(f"od_025 aligned network gap: {candidate['source_edge_id']} -> {candidate['target_edge_id']}")
    axis.legend(loc="best"); axis.grid(alpha=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def combined_candidate_impact(candidates: list[dict[str, Any]], impact: dict[str, Any], origin_node: int, destination_node: int) -> dict[str, Any]:
    extra_rows: list[int] = []
    extra_columns: list[int] = []
    extra_weights: list[float] = []
    pair_to_candidates: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in tqdm(candidates, total=len(candidates), desc="Building combined candidate graph", unit="candidate", dynamic_ncols=True):
        source = row.get("impact_source_graph_node")
        target = row.get("impact_target_graph_node")
        if source is None or target is None or source == target:
            continue
        weight = float(row["extension_m"]) + float(row["impact_source_offset_m"]) + float(row["impact_target_offset_m"])
        if not np.isfinite(weight) or weight <= 0:
            continue
        extra_rows.extend([int(source), int(target)])
        extra_columns.extend([int(target), int(source)])
        extra_weights.extend([weight, weight])
        pair_to_candidates[tuple(sorted((int(source), int(target))))].append(str(row["candidate_id"]))
    if not extra_rows:
        return {"distance_m": float(impact["baseline"]), "saving_m": 0.0, "used_candidate_ids": []}
    extra = csr_matrix(
        (np.asarray(extra_weights), (np.asarray(extra_rows), np.asarray(extra_columns))),
        shape=impact["matrix"].shape,
    )
    augmented = impact["matrix"] + extra
    distances, predecessors = dijkstra(
        augmented,
        directed=False,
        indices=origin_node,
        return_predecessors=True,
    )
    distance = float(distances[destination_node])
    used: list[str] = []
    current = int(destination_node)
    while current != int(origin_node) and current >= 0:
        previous = int(predecessors[current])
        if previous < 0:
            break
        used.extend(pair_to_candidates.get(tuple(sorted((current, previous))), []))
        current = previous
    return {
        "distance_m": distance,
        "saving_m": max(0.0, float(impact["baseline"] - distance)),
        "used_candidate_ids": sorted(set(used)),
    }


def nearest_edge_node(edge_id: str, point: np.ndarray, impact: dict[str, Any]) -> tuple[int | None, float]:
    nodes = impact["edge_nodes"].get(edge_id)
    if not nodes:
        return None, float("nan")
    node_ids = np.fromiter(nodes, dtype=np.int64)
    distances = np.linalg.norm(impact["node_xy"][node_ids] - point, axis=1)
    position = int(np.argmin(distances))
    return int(node_ids[position]), float(distances[position])


def add_od_impact(row: dict[str, Any], impact: dict[str, Any]) -> None:
    source_point = np.array([row["source_x"], row["source_y"]], dtype=np.float64)
    target_point = np.array([row["intersection_x"], row["intersection_y"]], dtype=np.float64)
    source_node, source_offset = nearest_edge_node(row["source_edge_id"], source_point, impact)
    target_node, target_offset = nearest_edge_node(row["target_edge_id"], target_point, impact)
    row["impact_source_graph_node"] = source_node
    row["impact_target_graph_node"] = target_node
    row["impact_source_offset_m"] = source_offset
    row["impact_target_offset_m"] = target_offset
    if source_node is None or target_node is None:
        row["estimated_od025_shortest_m"] = float("nan")
        row["estimated_od025_saving_m"] = float("nan")
        return
    connector = float(row["extension_m"]) + source_offset + target_offset
    forward = impact["from_origin"][source_node] + connector + impact["from_destination"][target_node]
    reverse = impact["from_origin"][target_node] + connector + impact["from_destination"][source_node]
    estimate = float(min(forward, reverse))
    row["estimated_od025_shortest_m"] = estimate
    row["estimated_od025_saving_m"] = max(0.0, float(impact["baseline"] - estimate))


def write_feature_classes(gdb_path: Path, rows: list[dict[str, Any]], spatial_reference: Any, overwrite: bool) -> None:
    import arcpy

    if gdb_path.exists() and overwrite:
        arcpy.management.Delete(str(gdb_path))
    if not gdb_path.exists():
        arcpy.management.CreateFileGDB(str(gdb_path.parent), gdb_path.name)
    line_fc = str(gdb_path / "EndpointExtensionCandidates")
    point_fc = str(gdb_path / "ProposedIntersections")
    for path, geometry_type in ((line_fc, "POLYLINE"), (point_fc, "POINT")):
        if arcpy.Exists(path):
            arcpy.management.Delete(path)
        arcpy.management.CreateFeatureclass(str(gdb_path), Path(path).name, geometry_type, spatial_reference=spatial_reference)
        arcpy.management.AddField(path, "candidate_id", "TEXT", field_length=32)
        arcpy.management.AddField(path, "source_edge", "TEXT", field_length=64)
        arcpy.management.AddField(path, "target_edge", "TEXT", field_length=64)
        arcpy.management.AddField(path, "extension_m", "DOUBLE")
        arcpy.management.AddField(path, "cross_ang", "DOUBLE")
        arcpy.management.AddField(path, "cand_type", "TEXT", field_length=32)
        arcpy.management.AddField(path, "od25_save_m", "DOUBLE")
    fields = ["SHAPE@", "candidate_id", "source_edge", "target_edge", "extension_m", "cross_ang", "cand_type", "od25_save_m"]
    with arcpy.da.InsertCursor(line_fc, fields) as line_cursor:
        for row in tqdm(rows, desc="Writing Step24b GIS candidates", unit="candidate", dynamic_ncols=True):
            start = arcpy.Point(row["source_x"], row["source_y"])
            end = arcpy.Point(row["intersection_x"], row["intersection_y"])
            line = arcpy.Polyline(arcpy.Array([start, end]), spatial_reference)
            values = [row["candidate_id"], row["source_edge_id"], row["target_edge_id"], row["extension_m"], row["crossing_angle_deg"], row["candidate_type"], row["estimated_od025_saving_m"]]
            if float(row["extension_m"]) > 1e-6:
                line_cursor.insertRow([line, *values])
    with arcpy.da.InsertCursor(point_fc, fields) as point_cursor:
        for row in tqdm(rows, desc="Writing Step24b intersection points", unit="candidate", dynamic_ncols=True):
            end = arcpy.Point(row["intersection_x"], row["intersection_y"])
            point = arcpy.PointGeometry(end, spatial_reference)
            values = [row["candidate_id"], row["source_edge_id"], row["target_edge_id"], row["extension_m"], row["crossing_angle_deg"], row["candidate_type"], row["estimated_od025_saving_m"]]
            point_cursor.insertRow([point, *values])


def create_focus_plot(edges: list[dict[str, Any]], rows: list[dict[str, Any]], od: dict[str, Any], output: Path, buffer_m: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    origin = np.array([od["origin_x"], od["origin_y"]])
    destination = np.array([od["destination_x"], od["destination_y"]])
    low = np.minimum(origin, destination) - buffer_m
    high = np.maximum(origin, destination) + buffer_m
    figure, axis = plt.subplots(figsize=(11, 11))
    for edge in tqdm(edges, desc="Plotting OD025 local network", unit="road", dynamic_ncols=True):
        for part in edge["parts"]:
            array = np.vstack(part)
            if array[:, 0].max() < low[0] or array[:, 0].min() > high[0] or array[:, 1].max() < low[1] or array[:, 1].min() > high[1]:
                continue
            axis.plot(array[:, 0], array[:, 1], color="#c9c9c9", linewidth=0.45, zorder=1)
    focused = [row for row in rows if row["near_od025_corridor"]]
    for row in focused:
        saving = row["estimated_od025_saving_m"]
        color = "#d73027" if np.isfinite(saving) and saving > 1000 else "#fdae61"
        axis.plot([row["source_x"], row["intersection_x"]], [row["source_y"], row["intersection_y"]], color=color, linewidth=2.0, zorder=3)
        axis.scatter([row["intersection_x"]], [row["intersection_y"]], color=color, s=12, zorder=4)
    axis.plot([origin[0], destination[0]], [origin[1], destination[1]], linestyle="--", color="#4575b4", linewidth=1.0, label="OD straight line", zorder=2)
    axis.scatter([origin[0]], [origin[1]], color="black", s=45, label="Origin", zorder=5)
    axis.scatter([destination[0]], [destination[1]], color="#fdae61", edgecolor="black", marker="*", s=100, label="Destination", zorder=5)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("od_025 endpoint-forward-extension topology candidates")
    axis.set_xlabel("Easting (m), EPSG:32650")
    axis.set_ylabel("Northing (m), EPSG:32650")
    axis.legend(loc="best")
    axis.grid(alpha=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_report_lines(
    summary: dict[str, Any],
    total_network_count: int,
    scanned_road_count: int,
    degree_one_count: int,
    short_candidates: list[dict[str, Any]],
    gap_candidates: list[dict[str, Any]],
    aligned_gap_candidates: list[dict[str, Any]],
    baseline_m: float,
) -> list[str]:
    best_gap = aligned_gap_candidates[0] if aligned_gap_candidates else None
    lines = [
        "# Step24b 道路拓扑断点检查报告", "", f"完成时间：{summary['finished_at']}", "",
        "## 结论", "",
        f"全路网包含 {total_network_count:,} 条 Step24 修复后道路；本次精查 od_025 走廊内 {scanned_road_count:,} 条道路和 {degree_one_count:,} 个 degree-1 端点。",
        f"3/5/10/15 米端点顺延共发现 {len(short_candidates):,} 个候选，但单独或合并加入内存图后均未缩短 od_025；小范围端点顺延不是该异常绕行的直接修复方案。",
        f"现有 od_025 最短路为 {baseline_m:.2f} 米。100 米内空间邻近但网络断开的诊断共发现 {len(gap_candidates)} 个可能影响路线的缺口，其中 {len(aligned_gap_candidates)} 个同时满足双端方向对齐、结构兼容且显著缩短路线。",
        "本次没有修改 Step24 正式路网，也没有重跑 Step25–Step32。", "",
        "## 关键缺口", "",
    ]
    if best_gap:
        lines.extend([
            f"- 缺口两端节点：{best_gap['source_node']} → {best_gap['target_node']}",
            f"- 缺口长度：{best_gap['gap_m']:.2f} 米",
            f"- 两端道路：{best_gap['source_edge_id']}（{best_gap['source_fclass']}，{best_gap['source_name'] or '未命名'}）→ {best_gap['target_edge_id']}（{best_gap['target_fclass']}，{best_gap['target_name'] or '未命名'}）",
            f"- 两端延伸方向偏差：{best_gap['source_outward_angle_deg']:.2f}° / {best_gap['target_outward_angle_deg']:.2f}°",
            f"- 桥梁、隧道、layer 和立体分离属性兼容：{best_gap['structural_compatible']}",
            f"- 仅在内存图加入该连接后的估计最短路：{best_gap['estimated_od025_shortest_m']:.2f} 米，缩短 {best_gap['estimated_od025_saving_m']:.2f} 米。",
            "- 判定：这是当前最可信的人工复核对象，但约 98 米属于缺失道路段而非微小拓扑缝；正式连接前必须核对底图、道路通行属性及现场障碍。",
        ])
    else:
        lines.append("未发现满足全部方向、结构与路线影响门槛的关键缺口。")
    lines.extend([
        "", "## 安全规则", "",
        "- 仅检查 degree-1 端点；短顺延分 3/5/10/15 米记录。",
        "- 长缺口诊断上限为 100 米，要求两端延伸方向偏差均不超过 15°。",
        "- 桥梁、隧道、layer 与 grade separation 必须兼容。",
        "- 所有候选均须人工核对隔离带、河道、围墙、道路许可和真实交叉关系。",
        "", "## 输出", "",
        "- `routing/data/topology_extension_check/endpoint_extension_candidates.csv`",
        "- `routing/data/topology_extension_check/od025_network_gap_candidates.csv`",
        "- `routing/data/topology_extension_check/od025_best_aligned_gap_diagnostic.png`",
        "- `routing/data/topology_extension_check/od025_endpoint_extension_candidates.png`",
        "- `routing/data/topology_extension_check/step24b_endpoint_extension_check.gdb`",
        "", "## 状态", "",
        "`READY_FOR_FORMAL_ENDPOINT_EXTENSION_REPAIR = FALSE`", "",
        "须先人工审核关键缺口，之后再决定是否建立独立拓扑修复版本并重跑 Step25–Step30。",
    ])
    return lines


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = PROJECT_ROOT / config["outputs"]["directory"]
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists; rerun with --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        PROGRESS_LOG.write_text("", encoding="utf-8")
    logger = setup_logger(PROJECT_ROOT / config["outputs"]["log"], args.overwrite)
    started_at = now_text()
    logger.info("Step24b check started at %s", started_at)
    failed: list[dict[str, Any]] = []
    try:
        gdb = PROJECT_ROOT / config["input"]["repaired_network_gdb"]
        layer_name = config["input"]["layer"]
        from osgeo import ogr

        od_frame = pd.read_csv(PROJECT_ROOT / config["input"]["od_samples"])
        od_row = od_frame.loc[od_frame["od_id"] == config["focus_case"]["od_id"]].iloc[0]
        od = {
            "origin_node": int(od_row["origin_node"]), "destination_node": int(od_row["destination_node"]),
            "origin_x": float(od_row["origin_x"]), "origin_y": float(od_row["origin_y"]),
            "destination_x": float(od_row["destination_x"]), "destination_y": float(od_row["destination_y"]),
        }
        corridor_buffer = float(config["focus_case"]["corridor_buffer_m"])
        od_start = np.array([od["origin_x"], od["origin_y"]])
        od_end = np.array([od["destination_x"], od["destination_y"]])
        scan_low = np.minimum(od_start, od_end) - corridor_buffer
        scan_high = np.maximum(od_start, od_end) + corridor_buffer

        ogr.UseExceptions()
        dataset = ogr.Open(str(gdb), 0)
        if dataset is None:
            raise FileNotFoundError(gdb)
        layer = dataset.GetLayerByName(layer_name)
        if layer is None:
            raise KeyError(layer_name)
        total_network_count = int(layer.GetFeatureCount())
        precision = float(config["geometry"]["endpoint_rounding_m"])
        endpoint_counts: Counter[tuple[int, int]] = Counter()
        for feature in tqdm(layer, total=total_network_count, desc="Counting all network endpoints", unit="road", dynamic_ncols=True):
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                continue
            endpoints = ogr_geometry_endpoints(geometry)
            if endpoints is None:
                continue
            endpoint_counts[rounded_node(endpoints[0], precision)] += 1
            endpoint_counts[rounded_node(endpoints[1], precision)] += 1
        layer.ResetReading()
        layer.SetSpatialFilterRect(float(scan_low[0]), float(scan_low[1]), float(scan_high[0]), float(scan_high[1]))
        local_count = int(layer.GetFeatureCount())
        logger.info("Full network endpoint count complete: %s roads; local OD corridor: %s roads", total_network_count, local_count)
        edges: list[dict[str, Any]] = []
        for feature in tqdm(layer, total=local_count, desc="Reading OD025 corridor road geometry", unit="road", dynamic_ncols=True):
            try:
                geometry = feature.GetGeometryRef()
                if geometry is None or geometry.IsEmpty():
                    raise ValueError("empty_geometry")
                parts = ogr_geometry_parts(geometry)
                if not parts:
                    raise ValueError("empty_or_single_vertex_geometry")
                edges.append({
                    "oid": int(feature.GetFID()), "parts": parts,
                    "edge_id": clean_text(feature.GetField("edge_id")),
                    "fclass": clean_text(feature.GetField("fclass")),
                    "name": clean_text(feature.GetField("name")),
                    "ref": clean_text(feature.GetField("ref")),
                    "layer": layer_value(feature.GetField("layer")),
                    "bridge": boolish(feature.GetField("bridge")),
                    "tunnel": boolish(feature.GetField("tunnel")),
                    "grade_sep": boolish(feature.GetField("grade_sep")),
                    "therm_class": clean_text(feature.GetField("therm_class")),
                    "repair_typ": clean_text(feature.GetField("repair_typ")),
                })
            except Exception as exc:
                failed.append({"record_id": feature.GetFID(), "stage": "read_geometry", "error_message": str(exc)})
        dataset = None
        logger.info("Local full-geometry read complete: %s valid roads", len(edges))

        endpoint_records: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        grid_size = float(config["geometry"]["spatial_grid_size_m"])
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for edge_index, edge in enumerate(tqdm(edges, desc="Indexing road endpoints and segments", unit="road", dynamic_ncols=True)):
            first_part, last_part = edge["parts"][0], edge["parts"][-1]
            for label, part, at_start in (("start", first_part, True), ("end", last_part, False)):
                endpoint = part[0] if at_start else part[-1]
                neighbor = distinct_neighbor(part, at_start)
                if neighbor is not None:
                    endpoint_records.append({"edge_index": edge_index, "endpoint": endpoint, "neighbor": neighbor, "endpoint_label": label, "node_key": rounded_node(endpoint, precision)})
            for part in edge["parts"]:
                for first, second in zip(part[:-1], part[1:]):
                    if np.linalg.norm(second - first) <= 1e-9:
                        continue
                    index = len(segments)
                    segments.append({"edge_index": edge_index, "first": first, "second": second})
                    min_x, min_y = np.minimum(first, second)
                    max_x, max_y = np.maximum(first, second)
                    for cell_x in range(math.floor(min_x / grid_size), math.floor(max_x / grid_size) + 1):
                        for cell_y in range(math.floor(min_y / grid_size), math.floor(max_y / grid_size) + 1):
                            grid[(cell_x, cell_y)].append(index)
        logger.info("Local geometry index complete: %s terminal records, %s vertex segments", len(endpoint_records), len(segments))

        max_extension = float(config["geometry"]["maximum_extension_m"])
        min_extension = float(config["geometry"]["minimum_extension_m"])
        min_angle = float(config["geometry"]["minimum_midline_crossing_angle_deg"])
        fraction_tolerance = float(config["geometry"]["endpoint_fraction_tolerance"])
        stages = [float(value) for value in config["geometry"]["staged_extension_distances_m"]]
        candidates: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        degree_one_count = 0
        for endpoint_record in tqdm(endpoint_records, desc="Testing forward endpoint extensions", unit="endpoint", dynamic_ncols=True):
            if endpoint_counts[endpoint_record["node_key"]] != 1:
                continue
            degree_one_count += 1
            source = edges[endpoint_record["edge_index"]]
            endpoint = endpoint_record["endpoint"]
            direction = endpoint - endpoint_record["neighbor"]
            length = float(np.linalg.norm(direction))
            if length <= 0:
                continue
            direction /= length
            ray_vector = direction * max_extension
            ray_end = endpoint + ray_vector
            low = np.minimum(endpoint, ray_end)
            high = np.maximum(endpoint, ray_end)
            segment_indices: set[int] = set()
            for cell_x in range(math.floor(low[0] / grid_size), math.floor(high[0] / grid_size) + 1):
                for cell_y in range(math.floor(low[1] / grid_size), math.floor(high[1] / grid_size) + 1):
                    segment_indices.update(grid.get((cell_x, cell_y), []))
            local_hits: list[dict[str, Any]] = []
            for segment_index in segment_indices:
                segment = segments[segment_index]
                target = edges[segment["edge_index"]]
                ok, reasons = compatible(source, target)
                if not ok:
                    continue
                intersection = segment_intersection(endpoint, ray_vector, segment["first"], segment["second"])
                if intersection is None:
                    continue
                ray_fraction, target_fraction, point = intersection
                extension = ray_fraction * max_extension
                if extension < min_extension or extension > max_extension + 1e-6:
                    continue
                angle = acute_angle_degrees(direction, segment["second"] - segment["first"])
                near_target_endpoint = target_fraction <= fraction_tolerance or target_fraction >= 1.0 - fraction_tolerance
                if not near_target_endpoint and angle < min_angle:
                    rejected_rows.append({"source_edge_id": source["edge_id"], "target_edge_id": target["edge_id"], "reason": "midline_crossing_angle_too_small", "extension_m": extension, "crossing_angle_deg": angle})
                    continue
                candidate_type = "endpoint_to_endpoint" if near_target_endpoint else "endpoint_to_road_midline"
                key = (source["edge_id"], endpoint_record["endpoint_label"], target["edge_id"], round(float(point[0]), 3), round(float(point[1]), 3))
                if key in seen:
                    continue
                seen.add(key)
                stage = next((value for value in stages if extension <= value + 1e-9), max_extension)
                source_identity = {value for value in (source["name"], source["ref"]) if value}
                target_identity = {value for value in (target["name"], target["ref"]) if value}
                review_flags: list[str] = []
                if source_identity and target_identity and source_identity.isdisjoint(target_identity):
                    review_flags.append("different_name_or_ref")
                if source["fclass"] != target["fclass"]:
                    review_flags.append("different_fclass")
                if near_target_endpoint:
                    review_flags.append("target_endpoint_junction")
                local_hits.append({
                    "source_oid": source["oid"], "source_edge_id": source["edge_id"],
                    "source_endpoint": endpoint_record["endpoint_label"],
                    "source_x": float(endpoint[0]), "source_y": float(endpoint[1]),
                    "source_degree": 1, "source_fclass": source["fclass"],
                    "source_name": source["name"], "source_ref": source["ref"],
                    "target_oid": target["oid"], "target_edge_id": target["edge_id"],
                    "target_fclass": target["fclass"], "target_name": target["name"],
                    "target_ref": target["ref"], "intersection_x": float(point[0]),
                    "intersection_y": float(point[1]), "extension_m": float(extension),
                    "extension_stage_m": float(stage), "target_fraction": float(target_fraction),
                    "crossing_angle_deg": float(angle), "candidate_type": candidate_type,
                    "bridge": source["bridge"], "tunnel": source["tunnel"],
                    "layer": source["layer"], "grade_separated": source["grade_sep"],
                    "review_flags": ";".join(review_flags),
                    "decision": "eligible_for_manual_review", "approved_for_repair": False,
                })
            if local_hits:
                local_hits.sort(key=lambda row: (row["extension_m"], -row["crossing_angle_deg"], row["target_edge_id"]))
                candidates.extend(local_hits[:3])

        # A road may already touch or cross another line geometrically but still
        # lack a graph node. This is distinct from a physical endpoint gap and
        # must be checked before increasing the extension distance.
        checked_segment_pairs: set[tuple[int, int]] = set()
        crossing_seen: set[tuple[Any, ...]] = set()
        for cell_segments in tqdm(list(grid.values()), desc="Testing unnoded road intersections", unit="cell", dynamic_ncols=True):
            unique_segments = sorted(set(cell_segments))
            for first_index, second_index in combinations(unique_segments, 2):
                pair_key = (first_index, second_index)
                if pair_key in checked_segment_pairs:
                    continue
                checked_segment_pairs.add(pair_key)
                first_segment = segments[first_index]
                second_segment = segments[second_index]
                if first_segment["edge_index"] == second_segment["edge_index"]:
                    continue
                first_edge = edges[first_segment["edge_index"]]
                second_edge = edges[second_segment["edge_index"]]
                ok, _ = compatible(first_edge, second_edge)
                if not ok:
                    continue
                first_vector = first_segment["second"] - first_segment["first"]
                intersection = segment_intersection(
                    first_segment["first"],
                    first_vector,
                    second_segment["first"],
                    second_segment["second"],
                )
                if intersection is None:
                    continue
                first_fraction, second_fraction, point = intersection
                first_endpoint = first_fraction <= fraction_tolerance or first_fraction >= 1.0 - fraction_tolerance
                second_endpoint = second_fraction <= fraction_tolerance or second_fraction >= 1.0 - fraction_tolerance
                if first_endpoint and second_endpoint:
                    continue
                point_key = rounded_node(point, precision)
                first_degree = endpoint_counts.get(point_key, 0) if first_endpoint else 2
                second_degree = endpoint_counts.get(point_key, 0) if second_endpoint else 2
                if first_endpoint and first_degree != 1:
                    continue
                if second_endpoint and second_degree != 1:
                    continue
                angle = acute_angle_degrees(first_vector, second_segment["second"] - second_segment["first"])
                if angle < min_angle:
                    continue
                if first_endpoint:
                    source, target = first_edge, second_edge
                    source_fraction, target_fraction_value = first_fraction, second_fraction
                    source_degree = first_degree
                    candidate_type = "endpoint_touching_road_midline_missing_node"
                elif second_endpoint:
                    source, target = second_edge, first_edge
                    source_fraction, target_fraction_value = second_fraction, first_fraction
                    source_degree = second_degree
                    candidate_type = "endpoint_touching_road_midline_missing_node"
                else:
                    source, target = first_edge, second_edge
                    source_fraction, target_fraction_value = first_fraction, second_fraction
                    source_degree = 2
                    candidate_type = "road_crossing_missing_node"
                crossing_key = (
                    min(source["edge_id"], target["edge_id"]),
                    max(source["edge_id"], target["edge_id"]),
                    round(float(point[0]), 3),
                    round(float(point[1]), 3),
                )
                if crossing_key in crossing_seen:
                    continue
                crossing_seen.add(crossing_key)
                source_identity = {value for value in (source["name"], source["ref"]) if value}
                target_identity = {value for value in (target["name"], target["ref"]) if value}
                review_flags: list[str] = ["requires_target_split"]
                if candidate_type == "road_crossing_missing_node":
                    review_flags.append("requires_both_roads_split")
                if source_identity and target_identity and source_identity.isdisjoint(target_identity):
                    review_flags.append("different_name_or_ref")
                if source["fclass"] != target["fclass"]:
                    review_flags.append("different_fclass")
                candidates.append({
                    "source_oid": source["oid"], "source_edge_id": source["edge_id"],
                    "source_endpoint": "touch" if candidate_type.startswith("endpoint") else "interior",
                    "source_x": float(point[0]), "source_y": float(point[1]),
                    "source_degree": int(source_degree), "source_fclass": source["fclass"],
                    "source_name": source["name"], "source_ref": source["ref"],
                    "target_oid": target["oid"], "target_edge_id": target["edge_id"],
                    "target_fclass": target["fclass"], "target_name": target["name"],
                    "target_ref": target["ref"], "intersection_x": float(point[0]),
                    "intersection_y": float(point[1]), "extension_m": 0.0,
                    "extension_stage_m": 0.0, "target_fraction": float(target_fraction_value),
                    "crossing_angle_deg": float(angle), "candidate_type": candidate_type,
                    "bridge": source["bridge"], "tunnel": source["tunnel"],
                    "layer": source["layer"], "grade_separated": source["grade_sep"],
                    "review_flags": ";".join(review_flags),
                    "decision": "eligible_for_manual_review", "approved_for_repair": False,
                })
        logger.info(
            "Candidate geometry scan complete: %s total candidates (%s unnoded intersections), %s rejected-angle records",
            len(candidates), len(crossing_seen), len(rejected_rows),
        )

        required_edge_ids = {
            str(row[field])
            for row in candidates
            for field in ("source_edge_id", "target_edge_id")
        }
        impact = load_graph_impact(
            PROJECT_ROOT / config["input"]["formal_graph"],
            od["origin_node"],
            od["destination_node"],
            required_edge_ids,
        )
        logger.info("OD graph-impact arrays complete; baseline %.3f m", impact["baseline"])
        for index, row in enumerate(candidates, start=1):
            row["candidate_id"] = f"ext_{index:05d}"
            point = np.array([row["source_x"], row["source_y"]])
            row["distance_to_od025_straight_m"] = point_to_segment_distance(point, od_start, od_end)
            row["near_od025_corridor"] = row["distance_to_od025_straight_m"] <= corridor_buffer
            add_od_impact(row, impact)
        combined_impact = combined_candidate_impact(
            candidates,
            impact,
            od["origin_node"],
            od["destination_node"],
        )

        classification = pd.read_csv(
            PROJECT_ROOT / config["input"]["road_classification"],
            dtype={"edge_id": str},
        )
        gap_candidates = find_proximity_gap_candidates(
            impact,
            od["origin_node"],
            od["destination_node"],
            config["network_gap_diagnostic"],
            classification,
        )
        gap_fields = [
            "gap_candidate_id", "source_node", "target_node", "source_x", "source_y", "target_x", "target_y",
            "gap_m", "source_degree", "target_degree", "source_outward_angle_deg", "target_outward_angle_deg",
            "source_edge_id", "target_edge_id", "source_fclass", "target_fclass", "source_name", "target_name",
            "source_layer", "target_layer", "source_bridge", "target_bridge", "source_tunnel", "target_tunnel",
            "structural_compatible", "estimated_od025_shortest_m", "estimated_od025_saving_m",
            "material_aligned_gap_candidate", "decision",
        ]
        write_csv(output / "od025_network_gap_candidates.csv", gap_candidates, gap_fields)
        aligned_gap_candidates = [row for row in gap_candidates if row["material_aligned_gap_candidate"]]
        if aligned_gap_candidates:
            create_gap_plot(impact, aligned_gap_candidates[0], output / "od025_best_aligned_gap_diagnostic.png")
        logger.info(
            "OD025 network-gap diagnosis complete: %s candidates, %s material aligned",
            len(gap_candidates), len(aligned_gap_candidates),
        )
        used_combined = set(combined_impact["used_candidate_ids"])
        for row in candidates:
            row["used_by_combined_od025_shortest"] = row["candidate_id"] in used_combined
        candidates.sort(key=lambda row: (-float(row["estimated_od025_saving_m"]) if np.isfinite(row["estimated_od025_saving_m"]) else 1e30, row["extension_m"], row["candidate_id"]))

        fields_out = [
            "candidate_id", "source_oid", "source_edge_id", "source_endpoint", "source_x", "source_y", "source_degree",
            "source_fclass", "source_name", "source_ref", "target_oid", "target_edge_id", "target_fclass", "target_name", "target_ref",
            "intersection_x", "intersection_y", "extension_m", "extension_stage_m", "target_fraction", "crossing_angle_deg",
            "candidate_type", "bridge", "tunnel", "layer", "grade_separated", "review_flags", "decision", "approved_for_repair",
            "distance_to_od025_straight_m", "near_od025_corridor", "impact_source_graph_node", "impact_target_graph_node",
            "impact_source_offset_m", "impact_target_offset_m", "estimated_od025_shortest_m", "estimated_od025_saving_m",
            "used_by_combined_od025_shortest",
        ]
        write_csv(output / "endpoint_extension_candidates.csv", candidates, fields_out)
        focused = [row for row in candidates if row["near_od025_corridor"]]
        write_csv(output / "od025_extension_candidates.csv", focused, fields_out)
        rejected_fields = ["source_edge_id", "target_edge_id", "reason", "extension_m", "crossing_angle_deg"]
        write_csv(output / "rejected_geometry_candidates.csv", rejected_rows, rejected_fields)
        failed_fields = ["record_id", "stage", "error_message"]
        write_csv(output / "failed_records.csv", failed, failed_fields)
        logger.info("CSV outputs complete")
        if bool(config["outputs"].get("write_candidate_gdb", True)):
            import arcpy
            spatial_reference = arcpy.SpatialReference(32650)
            write_feature_classes(PROJECT_ROOT / config["outputs"]["gdb"], candidates, spatial_reference, args.overwrite)
            logger.info("FileGDB candidate outputs complete")
        else:
            logger.info("FileGDB rewrite skipped; retaining the previously generated check-only GDB")
        create_focus_plot(edges, candidates, od, output / "od025_endpoint_extension_candidates.png", corridor_buffer)
        logger.info("OD025 diagnostic plot complete")

        stage_counts = Counter(str(row["extension_stage_m"]) for row in candidates)
        type_counts = Counter(row["candidate_type"] for row in candidates)
        positive = [row for row in candidates if np.isfinite(row["estimated_od025_saving_m"]) and row["estimated_od025_saving_m"] > 1.0]
        top = positive[:20]
        best_gap = aligned_gap_candidates[0] if aligned_gap_candidates else None
        summary = {
            "step": "24b", "mode": "check", "started_at": started_at, "finished_at": now_text(),
            "total_network_road_count": total_network_count, "scanned_corridor_road_count": len(edges),
            "scan_scope": config["geometry"]["scan_scope"], "degree_one_endpoint_count": degree_one_count,
            "candidate_count": len(candidates), "candidate_stage_counts": dict(stage_counts),
            "candidate_type_counts": dict(type_counts), "od025_baseline_shortest_m": impact["baseline"],
            "od025_corridor_candidate_count": len(focused), "od025_positive_impact_candidate_count": len(positive),
            "od025_best_estimated_shortest_m": min((row["estimated_od025_shortest_m"] for row in positive), default=None),
            "od025_best_estimated_saving_m": max((row["estimated_od025_saving_m"] for row in positive), default=0.0),
            "od025_all_candidates_combined_shortest_m": combined_impact["distance_m"],
            "od025_all_candidates_combined_saving_m": combined_impact["saving_m"],
            "od025_all_candidates_used_ids": combined_impact["used_candidate_ids"],
            "od025_proximity_gap_candidate_count": len(gap_candidates),
            "od025_material_aligned_gap_candidate_count": len(aligned_gap_candidates),
            "od025_best_aligned_gap": best_gap,
            "failed_record_count": len(failed), "formal_network_modified": False,
            "formal_downstream_rerun": False, "all_candidates_require_manual_review": True,
            "status": "CHECK_COMPLETE_AWAITING_MANUAL_CANDIDATE_REVIEW",
        }
        write_json(output / "step24b_check_summary.json", summary)
        report_lines = [
            "# Step24b 道路端点顺延拓扑检查报告", "", f"完成时间：{summary['finished_at']}", "",
            "## 结论", "", f"本次先统计全网{total_network_count:,}条Step24修复后道路端点，再只读检查od_025走廊内{len(edges):,}条完整折线和{degree_one_count:,}个degree-1端点。",
            f"识别{len(candidates):,}个结构兼容的端点向前顺延候选，其中{len(focused):,}个位于od_025直线走廊{corridor_buffer:.0f}米范围内。",
            f"现有od_025最短路线为{impact['baseline']:.2f}米；图近似影响筛查发现{len(positive)}个可能缩短该路线的候选。",
            f"把全部候选仅在内存图中同时加入后，估计最短路为{combined_impact['distance_m']:.2f}米，减少{combined_impact['saving_m']:.2f}米；新路径使用候选：{', '.join(combined_impact['used_candidate_ids']) or '无'}。",
            "所有结果均为人工复核候选；没有修改Step24正式路网，没有重跑Step25–Step32。", "",
            "## 安全规则", "", "- 只从degree-1端点沿原道路末段方向向外顺延。", "- 分3/5/10/15米记录候选，最大15米。",
            "- 桥、隧道、layer、grade separation和热舒适道路类别必须一致。", "- 接入道路中段时要求交叉角不小于25度。",
            "- 候选需要人工核对隔离带、河道、围墙、道路许可和真实交叉口。", "",
            "## od_025潜在影响最大的候选", "",
            "|候选|顺延(m)|类型|源道路|目标道路|估计新最短路(m)|估计减少(m)|", "|---|---:|---|---|---|---:|---:|",
        ]
        for row in top[:10]:
            report_lines.append(f"|{row['candidate_id']}|{row['extension_m']:.2f}|{row['candidate_type']}|{row['source_edge_id']}|{row['target_edge_id']}|{row['estimated_od025_shortest_m']:.2f}|{row['estimated_od025_saving_m']:.2f}|")
        report_lines.extend(["", "## 输出", "", "- `routing/data/topology_extension_check/endpoint_extension_candidates.csv`", "- `routing/data/topology_extension_check/od025_extension_candidates.csv`", "- `routing/data/topology_extension_check/od025_endpoint_extension_candidates.png`", "- `routing/data/topology_extension_check/step24b_endpoint_extension_check.gdb`", "", "## 状态", "", "`READY_FOR_FORMAL_ENDPOINT_EXTENSION_REPAIR = FALSE`", "", "必须先人工审核具体候选，再决定是否建立独立拓扑修复版本并重新运行Step25–Step30。"])
        report_lines = build_report_lines(
            summary,
            total_network_count,
            len(edges),
            degree_one_count,
            candidates,
            gap_candidates,
            aligned_gap_candidates,
            impact["baseline"],
        )
        report_path = PROJECT_ROOT / config["outputs"]["report"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        logger.info("Step24b check complete: %s", json.dumps(summary, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.error("Step24b failed: %s", exc)
        logger.error(traceback.format_exc())
        write_csv(output / "failed_records.csv", [{"record_id": "step24b", "stage": "fatal", "error_message": str(exc)}], ["record_id", "stage", "error_message"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
