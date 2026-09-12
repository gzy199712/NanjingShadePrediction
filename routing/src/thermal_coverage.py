"""Candidate-only thermal point-to-road coverage audits."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from tqdm import tqdm

from routing.src.mode_classification import mode_is_candidate
from routing.src.road_schema import RoadFeature
from routing.src.topology_audit import graph_metrics


def _grid_key(x: float, y: float, size: float) -> tuple[int, int]:
    return math.floor(x / size), math.floor(y / size)


def _cells_for_envelope(
    envelope: tuple[float, float, float, float],
    radius: float,
    size: float,
):
    min_x, max_x, min_y, max_y = envelope
    first = _grid_key(min_x - radius, min_y - radius, size)
    last = _grid_key(max_x + radius, max_y + radius, size)
    for gx in range(first[0], last[0] + 1):
        for gy in range(first[1], last[1] + 1):
            yield gx, gy


def thermal_coverage(
    roads: list[RoadFeature],
    points: list[dict[str, Any]],
    distances: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[float, set[str]]]:
    maximum = max(distances)
    point_grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        point_grid[_grid_key(point["x"], point["y"], maximum)].append(index)
    road_rows: list[dict[str, Any]] = []
    covered: dict[float, set[str]] = {distance: set() for distance in distances}
    for road in tqdm(
        roads,
        desc="Auditing thermal point-road distances",
        unit="road",
        dynamic_ncols=True,
    ):
        candidates: set[int] = set()
        for cell in _cells_for_envelope(road.envelope, maximum, maximum):
            candidates.update(point_grid.get(cell, []))
        values: list[tuple[str, float]] = []
        for index in candidates:
            distance = float(road.geometry.Distance(points[index]["geometry"]))
            if distance <= maximum:
                values.append((points[index]["point_id"], distance))
        values.sort(key=lambda item: item[1])
        row: dict[str, Any] = {
            "edge_id": road.edge_id,
            "fclass": road.attrs.get("fclass"),
            "fclass_cn": road.attrs.get("fclass_cn"),
            "length_m": road.length_m,
            "inside_center": road.inside_center,
            "inside_center_500m": road.inside_center_500m,
            "inside_center_1000m": road.inside_center_1000m,
            "walk_candidate": mode_is_candidate(road, "walk"),
            "bike_candidate": mode_is_candidate(road, "bike"),
            "shared_candidate": mode_is_candidate(road, "shared"),
            "nearest_point_id": values[0][0] if values else None,
            "nearest_point_distance_m": values[0][1] if values else None,
            "formal_match_performed": False,
        }
        for distance in distances:
            support = sum(value <= distance for _, value in values)
            row[f"support_points_within_{int(distance)}m"] = support
            row[f"covered_within_{int(distance)}m"] = support > 0
            if support:
                covered[distance].add(road.edge_id)
        road_rows.append(row)
    summaries: list[dict[str, Any]] = []
    group_definitions: list[tuple[str, str, list[RoadFeature]]] = [
        ("overall", "all", roads),
        ("scope", "center", [road for road in roads if road.inside_center]),
        (
            "scope",
            "outside_center",
            [road for road in roads if not road.inside_center],
        ),
        (
            "mode",
            "walk_candidate",
            [road for road in roads if mode_is_candidate(road, "walk")],
        ),
        (
            "mode",
            "bike_candidate",
            [road for road in roads if mode_is_candidate(road, "bike")],
        ),
        (
            "mode",
            "shared_candidate",
            [road for road in roads if mode_is_candidate(road, "shared")],
        ),
    ]
    classes = sorted(
        {str(road.attrs.get("fclass") or "NULL") for road in roads}
    )
    for fclass in classes:
        group_definitions.append(
            (
                "fclass",
                fclass,
                [
                    road
                    for road in roads
                    if str(road.attrs.get("fclass") or "NULL") == fclass
                ],
            )
        )
    for distance in tqdm(
        distances,
        desc="Summarizing thermal coverage radii",
        unit="radius",
        dynamic_ncols=True,
    ):
        supported = covered[distance]
        for group_type, group_name, members in group_definitions:
            count = len(members)
            length = sum(road.length_m for road in members)
            matched = [road for road in members if road.edge_id in supported]
            matched_length = sum(road.length_m for road in matched)
            summaries.append(
                {
                    "match_distance_m": distance,
                    "group_type": group_type,
                    "group_name": group_name,
                    "road_count": count,
                    "road_length_km": length / 1000,
                    "matched_road_count": len(matched),
                    "matched_road_length_km": matched_length / 1000,
                    "uncovered_road_count": count - len(matched),
                    "uncovered_road_length_km": (length - matched_length) / 1000,
                    "feature_coverage_ratio": len(matched) / count if count else None,
                    "length_coverage_ratio": matched_length / length if length else None,
                    "formal_matching_performed": False,
                }
            )
    return road_rows, summaries, covered


def center_scope_comparison(
    roads: list[RoadFeature],
    points: list[dict[str, Any]],
    center_geometry: Any,
    covered_75m: set[str],
    small_component_edges: int,
) -> list[dict[str, Any]]:
    from osgeo import ogr

    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    for point in tqdm(
        points,
        desc="Building thermal point convex hull",
        unit="point",
        dynamic_ncols=True,
    ):
        multipoint.AddGeometry(point["geometry"])
    hull = multipoint.ConvexHull()
    scopes: list[tuple[str, Any | None, list[RoadFeature]]] = [
        ("all_nanjing", None, roads),
        (
            "center",
            center_geometry,
            [road for road in roads if road.inside_center],
        ),
        (
            "center_plus_500m",
            center_geometry.Buffer(500),
            [road for road in roads if road.inside_center_500m],
        ),
        (
            "center_plus_1000m",
            center_geometry.Buffer(1000),
            [road for road in roads if road.inside_center_1000m],
        ),
        (
            "thermal_point_convex_hull",
            hull,
            [road for road in roads if road.geometry.Intersects(hull)],
        ),
        (
            "thermal_point_hull_plus_100m",
            hull.Buffer(100),
            [road for road in roads if road.geometry.Intersects(hull.Buffer(100))],
        ),
        (
            "actual_thermal_coverage_candidate_network",
            None,
            [road for road in roads if road.edge_id in covered_75m],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, polygon, members in tqdm(
        scopes,
        desc="Comparing candidate work scopes",
        unit="scope",
        dynamic_ncols=True,
    ):
        metrics = graph_metrics(name, members, small_component_edges)
        total_length = sum(road.length_m for road in members)
        covered_members = [
            road for road in members if road.edge_id in covered_75m
        ]
        rows.append(
            {
                "scope": name,
                "polygon_area_km2": (
                    float(polygon.Area()) / 1_000_000 if polygon is not None else None
                ),
                "road_count": len(members),
                "road_length_km": total_length / 1000,
                "covered_road_count_75m": len(covered_members),
                "covered_road_length_km_75m": sum(
                    road.length_m for road in covered_members
                )
                / 1000,
                "covered_length_ratio_75m": (
                    sum(road.length_m for road in covered_members) / total_length
                    if total_length
                    else None
                ),
                "node_count": metrics["node_count"],
                "edge_count": metrics["edge_count"],
                "connected_component_count": metrics[
                    "connected_component_count"
                ],
                "largest_component_node_ratio": metrics[
                    "largest_component_node_ratio"
                ],
                "degree_1_nodes": metrics["degree_1_nodes"],
                "final_scope_selected": False,
            }
        )
    return rows
