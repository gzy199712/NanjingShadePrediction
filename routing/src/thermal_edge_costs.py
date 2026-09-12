"""Direct thermal-edge aggregation and conservative interpolation audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class ThermalEdge:
    edge_id: str
    fclass: str
    bridge: str
    tunnel: str
    layer: float
    repair_type: str
    length_m: float
    start: tuple[float, float]
    end: tuple[float, float]
    geometry: Any


def node_key(point: tuple[float, float]) -> tuple[float, float]:
    return round(point[0], 3), round(point[1], 3)


def iter_segments(geometry: Any):
    from osgeo import ogr

    if ogr.GT_Flatten(geometry.GetGeometryType()) == ogr.wkbMultiLineString:
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


def project_along_edge(
    geometry: Any, point_x: float, point_y: float
) -> tuple[float, float]:
    cumulative = 0.0
    best_distance = math.inf
    best_along = 0.0
    for x1, y1, x2, y2 in iter_segments(geometry):
        dx = x2 - x1
        dy = y2 - y1
        segment_length = math.hypot(dx, dy)
        if segment_length == 0:
            continue
        ratio = max(
            0.0,
            min(
                1.0,
                ((point_x - x1) * dx + (point_y - y1) * dy)
                / (segment_length * segment_length),
            ),
        )
        nearest_x = x1 + ratio * dx
        nearest_y = y1 + ratio * dy
        distance = math.hypot(point_x - nearest_x, point_y - nearest_y)
        if distance < best_distance:
            best_distance = distance
            best_along = cumulative + ratio * segment_length
        cumulative += segment_length
    return best_along, best_distance


def assign_along_edge_weights(
    mapping: pd.DataFrame,
    edges: dict[str, ThermalEdge],
    minimum_weight_m: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for edge_id, members in tqdm(
        mapping.groupby("edge_id", sort=True),
        total=mapping["edge_id"].nunique(),
        desc="Computing along-edge point weights",
        unit="edge",
        dynamic_ncols=True,
    ):
        edge = edges[str(edge_id)]
        projected: list[dict[str, Any]] = []
        for row in members.to_dict("records"):
            along, perpendicular = project_along_edge(
                edge.geometry, float(row["utm_x"]), float(row["utm_y"])
            )
            projected.append(
                {
                    **row,
                    "edge_position_m": along,
                    "projection_distance_m": perpendicular,
                }
            )
        projected.sort(key=lambda value: value["edge_position_m"])
        positions = [float(value["edge_position_m"]) for value in projected]
        if len(projected) == 1:
            raw_weights = [max(edge.length_m, minimum_weight_m)]
        else:
            boundaries = [0.0]
            boundaries.extend(
                (positions[index] + positions[index + 1]) / 2
                for index in range(len(positions) - 1)
            )
            boundaries.append(edge.length_m)
            raw_weights = [
                max(boundaries[index + 1] - boundaries[index], minimum_weight_m)
                for index in range(len(projected))
            ]
        total = sum(raw_weights)
        for value, raw_weight in zip(projected, raw_weights, strict=True):
            value["along_edge_weight_m"] = raw_weight
            value["along_edge_weight"] = raw_weight / total
            rows.append(value)
    return pd.DataFrame(rows)


def mixture_mean_std(
    means: np.ndarray, stds: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    mean = float(np.sum(weights * means))
    variance = float(
        np.sum(weights * (np.square(stds) + np.square(means - mean)))
    )
    return mean, math.sqrt(max(variance, 0.0))


def aggregate_direct_edge_hours(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = predictions.groupby(["edge_id", "hour"], sort=True)
    for (edge_id, hour), members in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Aggregating direct edge-hour thermal costs",
        unit="edge-hour",
        dynamic_ncols=True,
    ):
        weights = members["along_edge_weight"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        shade_mean, shade_std = mixture_mean_std(
            members["shade_pred_mean"].to_numpy(dtype=float),
            members["shade_pred_std"].to_numpy(dtype=float),
            weights,
        )
        tmrt_mean, tmrt_std = mixture_mean_std(
            members["tmrt_pred_mean"].to_numpy(dtype=float),
            members["tmrt_pred_std"].to_numpy(dtype=float),
            weights,
        )
        utci_mean, utci_std = mixture_mean_std(
            members["utci_pred_mean"].to_numpy(dtype=float),
            members["utci_pred_std"].to_numpy(dtype=float),
            weights,
        )
        records.append(
            {
                "edge_id": str(edge_id),
                "hour": int(hour),
                "shade_mean": shade_mean,
                "shade_std": shade_std,
                "tmrt_mean": tmrt_mean,
                "tmrt_std": tmrt_std,
                "utci_mean": utci_mean,
                "utci_std": utci_std,
                "support_point_count": int(members["point_id"].nunique()),
                "nearest_point_distance_m": float(
                    members["projection_distance_m"].min()
                ),
                "maximum_point_distance_m": float(
                    members["projection_distance_m"].max()
                ),
                "coverage_type": "direct",
                "interpolation_distance_m": 0.0,
                "coverage_confidence": (
                    "high"
                    if members["point_id"].nunique() >= 2
                    else "medium"
                ),
                "formal_edge_cost": False,
            }
        )
    return pd.DataFrame(records)
