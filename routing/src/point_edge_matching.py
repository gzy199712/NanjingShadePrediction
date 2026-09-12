"""Candidate point-to-edge matching utilities for Phase G4."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


@dataclass
class EdgeCandidate:
    edge_id: str
    fclass: str
    bridge: str
    tunnel: str
    layer: float | None
    grade_separated: bool
    repair_type: str
    geometry: Any
    envelope: tuple[float, float, float, float]


def parse_move_directions(
    image_dir: Path,
    required_point_ids: set[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values: dict[str, set[float]] = defaultdict(set)
    failures: list[dict[str, Any]] = []
    files = list(image_dir.glob("*.png"))
    for path in tqdm(
        files,
        desc="Recovering point move_dir from source filenames",
        unit="image",
        dynamic_ncols=True,
    ):
        parts = path.stem.split("_")
        if not parts or parts[0] not in required_point_ids:
            continue
        try:
            if len(parts) != 8:
                raise ValueError(f"Expected 8 filename fields, found {len(parts)}")
            values[parts[0]].add(float(parts[6]) % 360.0)
        except Exception as error:
            failures.append(
                {
                    "category": "move_dir_parse",
                    "object_id": parts[0] if parts else "",
                    "error_type": type(error).__name__,
                    "error_message": f"{path.name}: {error}",
                }
            )
    result: dict[str, float] = {}
    for point_id in tqdm(
        sorted(required_point_ids),
        desc="Validating point move_dir consistency",
        unit="point",
        dynamic_ncols=True,
    ):
        point_values = values.get(point_id, set())
        if len(point_values) == 1:
            result[point_id] = next(iter(point_values))
        elif not point_values:
            failures.append(
                {
                    "category": "move_dir_missing",
                    "object_id": point_id,
                    "error_type": "MissingMoveDirection",
                    "error_message": "No source filename found for point.",
                }
            )
        else:
            failures.append(
                {
                    "category": "move_dir_conflict",
                    "object_id": point_id,
                    "error_type": "ConflictingMoveDirection",
                    "error_message": ",".join(map(str, sorted(point_values))),
                }
            )
    return result, failures


def grid_key(x: float, y: float, size: float) -> tuple[int, int]:
    return math.floor(x / size), math.floor(y / size)


def nearby_cells(
    x: float, y: float, radius: float, size: float
) -> Iterable[tuple[int, int]]:
    min_key = grid_key(x - radius, y - radius, size)
    max_key = grid_key(x + radius, y + radius, size)
    for gx in range(min_key[0], max_key[0] + 1):
        for gy in range(min_key[1], max_key[1] + 1):
            yield gx, gy


def build_edge_grid(
    edges: list[EdgeCandidate], grid_size: float
) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, edge in enumerate(
        tqdm(
            edges,
            desc="Indexing repaired thermal edges",
            unit="edge",
            dynamic_ncols=True,
        )
    ):
        min_x, max_x, min_y, max_y = edge.envelope
        min_key = grid_key(min_x, min_y, grid_size)
        max_key = grid_key(max_x, max_y, grid_size)
        for gx in range(min_key[0], max_key[0] + 1):
            for gy in range(min_key[1], max_key[1] + 1):
                grid[(gx, gy)].append(index)
    return grid


def _segments(geometry: Any):
    from osgeo import ogr

    flat_type = ogr.GT_Flatten(geometry.GetGeometryType())
    if flat_type == ogr.wkbMultiLineString:
        lines = [
            geometry.GetGeometryRef(index)
            for index in range(geometry.GetGeometryCount())
        ]
    else:
        lines = [geometry]
    for line in lines:
        for index in range(line.GetPointCount() - 1):
            first = line.GetPoint(index)
            second = line.GetPoint(index + 1)
            yield first[0], first[1], second[0], second[1]


def nearest_segment_bearing(
    geometry: Any, point_x: float, point_y: float
) -> tuple[float, float]:
    best_distance = math.inf
    best_bearing = 0.0
    for x1, y1, x2, y2 in _segments(geometry):
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            continue
        ratio = max(
            0.0,
            min(
                1.0,
                ((point_x - x1) * dx + (point_y - y1) * dy) / length_sq,
            ),
        )
        nearest_x = x1 + ratio * dx
        nearest_y = y1 + ratio * dy
        distance = math.hypot(point_x - nearest_x, point_y - nearest_y)
        if distance < best_distance:
            best_distance = distance
            best_bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    return best_distance, best_bearing


def undirected_angle_difference(first: float, second: float) -> float:
    value = abs(first - second) % 180.0
    return min(value, 180.0 - value)


def score_candidate(
    *,
    distance_m: float,
    direction_difference_deg: float | None,
    fclass: str,
    grade_separated: bool,
    rules: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    weights = rules["score_weights"]
    components = rules["score_components"]
    distance_cost = min(
        distance_m / float(components["distance_normalization_m"]), 1.0
    )
    direction_cost = (
        min(
            direction_difference_deg
            / float(components["direction_normalization_deg"]),
            1.0,
        )
        if direction_difference_deg is not None
        else 0.5
    )
    class_cost = (
        float(components["topology_connector_class_cost"])
        if fclass == "topology_connector"
        else float(components["ordinary_thermal_road_class_cost"])
    )
    level_cost = (
        float(components["grade_separated_without_point_level_cost"])
        if grade_separated
        else float(components["ground_level_cost"])
    )
    total = (
        float(weights["distance"]) * distance_cost
        + float(weights["direction"]) * direction_cost
        + float(weights["road_class"]) * class_cost
        + float(weights["level_uncertainty"]) * level_cost
    )
    return total, {
        "distance_cost": distance_cost,
        "direction_cost": direction_cost,
        "class_cost": class_cost,
        "level_cost": level_cost,
    }
