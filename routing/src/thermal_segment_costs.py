"""Approved Step26 thermal-segment construction and hourly cost assignment."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from routing.src.reporting import atomic_csv, atomic_json, atomic_text
from routing.src.road_schema import load_union_geometry


MEAN_COLUMNS = ("shade_pred_mean", "tmrt_pred_mean", "utci_pred_mean")
STD_COLUMNS = ("shade_pred_std", "tmrt_pred_std", "utci_pred_std")


@dataclass
class ThermalSegment:
    segment_id: str
    original_edge_id: str
    segment_index: int
    length_m: float
    midpoint_x: float
    midpoint_y: float
    fclass: str
    bridge: str
    tunnel: str
    layer: float
    repair_type: str
    source_coverage: str
    source_edge_id: str | None
    geometry: Any


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(value for value in path.rglob("*") if value.is_file()) if path.is_dir() else [path]
    for value in paths:
        if path.is_dir():
            digest.update(str(value.relative_to(path)).encode("utf-8"))
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: Any) -> bool:
    return str(value or "").strip().upper() not in {
        "",
        "0",
        "F",
        "FALSE",
        "N",
        "NO",
        "NONE",
    }


def _is_surface(bridge: str, tunnel: str, layer: float) -> bool:
    return not _truthy(bridge) and not _truthy(tunnel) and abs(layer) < 0.5


def _structure_key(bridge: str, tunnel: str, layer: float) -> tuple[bool, bool, float]:
    return _truthy(bridge), _truthy(tunnel), round(float(layer), 3)


def _iter_lines(geometry: Any) -> Iterable[Any]:
    from osgeo import ogr

    flat = ogr.GT_Flatten(geometry.GetGeometryType())
    if flat == ogr.wkbLineString:
        yield geometry
        return
    for index in range(geometry.GetGeometryCount()):
        child = geometry.GetGeometryRef(index)
        if child is not None:
            yield from _iter_lines(child)


def _subdivide_line(line: Any, target_length_m: float) -> Iterable[Any]:
    from osgeo import ogr

    for index in range(line.GetPointCount() - 1):
        first = line.GetPoint(index)
        second = line.GetPoint(index + 1)
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        part_count = max(1, int(math.ceil(length / target_length_m)))
        for part in range(part_count):
            start_ratio = part / part_count
            end_ratio = (part + 1) / part_count
            segment = ogr.Geometry(ogr.wkbLineString)
            segment.AddPoint(
                first[0] + start_ratio * dx,
                first[1] + start_ratio * dy,
            )
            segment.AddPoint(
                first[0] + end_ratio * dx,
                first[1] + end_ratio * dy,
            )
            yield segment


def build_center_segments(
    *,
    gdb_path: Path,
    boundary_path: Path,
    coverage: pd.DataFrame,
    interpolation: pd.DataFrame,
    target_epsg: int,
    target_length_m: float,
    routing_scope: str,
    failures: list[dict[str, Any]],
) -> list[ThermalSegment]:
    from osgeo import ogr

    boundary = load_union_geometry(boundary_path, target_epsg)
    if routing_scope == "center_boundary_bbox":
        envelope = boundary.GetEnvelope()
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in (
            (envelope[0], envelope[2]),
            (envelope[1], envelope[2]),
            (envelope[1], envelope[3]),
            (envelope[0], envelope[3]),
            (envelope[0], envelope[2]),
        ):
            ring.AddPoint_2D(x, y)
        scope_geometry = ogr.Geometry(ogr.wkbPolygon)
        scope_geometry.AddGeometry(ring)
    elif routing_scope == "center_boundary":
        scope_geometry = boundary
    else:
        raise ValueError(f"Unsupported routing_scope: {routing_scope}")
    coverage_index = coverage.set_index("edge_id", drop=False)
    interpolation_sources = (
        dict(zip(interpolation["edge_id"], interpolation["source_edge_id"]))
        if len(interpolation)
        else {}
    )
    dataset = ogr.Open(str(gdb_path), 0)
    if dataset is None:
        raise FileNotFoundError(gdb_path)
    layer = dataset.GetLayerByName("ThermalComfortNetworkRepaired")
    if layer is None:
        raise KeyError("ThermalComfortNetworkRepaired")
    segments: list[ThermalSegment] = []
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Building approved routing-scope thermal segments",
        unit="edge",
        dynamic_ncols=True,
    ):
        edge_id = str(feature.GetField("edge_id")).strip()
        try:
            geometry = feature.GetGeometryRef()
            if (
                geometry is None
                or geometry.IsEmpty()
                or not geometry.Intersects(scope_geometry)
            ):
                continue
            clipped = geometry.Intersection(scope_geometry)
            if clipped is None or clipped.IsEmpty():
                continue
            if edge_id not in coverage_index.index:
                raise KeyError("Missing Step26 coverage preview")
            row = coverage_index.loc[edge_id]
            source_coverage = str(row["coverage_type_preview"])
            source_edge_id = (
                edge_id
                if source_coverage == "direct"
                else interpolation_sources.get(edge_id)
            )
            segment_index = 0
            for line in _iter_lines(clipped):
                for segment_geometry in _subdivide_line(
                    line, target_length_m
                ):
                    segment_index += 1
                    first = segment_geometry.GetPoint(0)
                    second = segment_geometry.GetPoint(1)
                    segments.append(
                        ThermalSegment(
                            segment_id=f"{edge_id}_{segment_index:06d}",
                            original_edge_id=edge_id,
                            segment_index=segment_index,
                            length_m=float(segment_geometry.Length()),
                            midpoint_x=(first[0] + second[0]) / 2,
                            midpoint_y=(first[1] + second[1]) / 2,
                            fclass=str(row["fclass"]),
                            bridge=str(row["bridge"]),
                            tunnel=str(row["tunnel"]),
                            layer=float(row["layer"]),
                            repair_type=str(row["repair_type"]),
                            source_coverage=source_coverage,
                            source_edge_id=source_edge_id,
                            geometry=segment_geometry,
                        )
                    )
        except Exception as error:
            failures.append(
                {
                    "category": "segment_build",
                    "object_id": edge_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    return segments


def prediction_cube(
    predictions: pd.DataFrame, hours: list[int]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    points = (
        predictions[["point_id", "utm_x", "utm_y"]]
        .drop_duplicates("point_id")
        .sort_values("point_id")
        .reset_index(drop=True)
    )
    point_ids = points["point_id"].tolist()
    means = np.empty((len(points), len(hours), 3), dtype=np.float64)
    stds = np.empty_like(means)
    for hour_index, hour in enumerate(hours):
        current = (
            predictions[predictions["hour"] == hour]
            .set_index("point_id")
            .reindex(point_ids)
        )
        means[:, hour_index, :] = current.loc[:, MEAN_COLUMNS].to_numpy(float)
        stds[:, hour_index, :] = current.loc[:, STD_COLUMNS].to_numpy(float)
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("Prediction cube contains non-finite values")
    return points, means, stds


def mixture(
    means: np.ndarray, stds: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / weights.sum()
    output_mean = np.sum(normalized[:, None, None] * means, axis=0)
    variance = np.sum(
        normalized[:, None, None]
        * (np.square(stds) + np.square(means - output_mean[None, :, :])),
        axis=0,
    )
    return output_mean, np.sqrt(np.maximum(variance, 0.0))


def build_priors(
    *,
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    hours: list[int],
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int]]:
    point_classes = (
        mapping[mapping["match_status"] == "matched"][
            ["point_id", "fclass"]
        ]
        .drop_duplicates("point_id")
        .copy()
    )
    merged = predictions.merge(
        point_classes, on="point_id", how="inner", validate="many_to_one"
    )
    priors: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int]] = {}
    for (fclass, hour), members in tqdm(
        merged.groupby(["fclass", "hour"], sort=True),
        desc="Building road-class hourly priors",
        unit="class-hour",
        dynamic_ncols=True,
    ):
        weights = np.ones(len(members), dtype=float)
        mean, std = mixture(
            members.loc[:, MEAN_COLUMNS].to_numpy(float)[:, None, :],
            members.loc[:, STD_COLUMNS].to_numpy(float)[:, None, :],
            weights,
        )
        priors[(str(fclass), int(hour))] = (
            mean[0],
            std[0],
            int(members["point_id"].nunique()),
        )
    for hour, members in merged.groupby("hour", sort=True):
        weights = np.ones(len(members), dtype=float)
        mean, std = mixture(
            members.loc[:, MEAN_COLUMNS].to_numpy(float)[:, None, :],
            members.loc[:, STD_COLUMNS].to_numpy(float)[:, None, :],
            weights,
        )
        priors[("__global__", int(hour))] = (
            mean[0],
            std[0],
            int(members["point_id"].nunique()),
        )
    if any(("__global__", hour) not in priors for hour in hours):
        raise ValueError("Missing global hourly prior")
    return priors


def _distance_level(
    distance_m: float, levels: list[dict[str, Any]]
) -> tuple[str, str, str, float] | None:
    for level in levels:
        lower = float(level.get("minimum_distance_exclusive_m", -1))
        upper = float(level["maximum_distance_m"])
        if distance_m > lower and distance_m <= upper:
            return (
                str(level["coverage_type"]),
                str(level["confidence"]),
                str(level["routing_policy"]),
                upper,
            )
    return None


def _distance_multiplier(
    coverage_type: str,
    distance_m: float,
    upper_m: float,
    uncertainty: dict[str, Any],
) -> float:
    key = {
        "spatial_interpolated": "spatial_interpolated_max_multiplier",
        "spatial_interpolated_low": "spatial_interpolated_low_max_multiplier",
        "spatial_fallback": "spatial_fallback_max_multiplier",
        "structure_interpolated": "spatial_fallback_max_multiplier",
    }.get(coverage_type)
    if key is None:
        return 1.0
    maximum = float(uncertainty[key])
    return 1.0 + (maximum - 1.0) * min(distance_m / upper_m, 1.0)


def write_segment_gdb(
    path: Path,
    segments: list[ThermalSegment],
    assignments: list[dict[str, Any]],
    *,
    target_epsg: int,
    overwrite: bool,
) -> int:
    from osgeo import ogr, osr

    if path.exists():
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = ogr.GetDriverByName("OpenFileGDB")
    dataset = driver.CreateDataSource(str(path))
    if dataset is None:
        raise RuntimeError(f"Cannot create {path}")
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(target_epsg)
    layer = dataset.CreateLayer(
        "ThermalCostSegments", spatial_reference, ogr.wkbLineString
    )
    fields = [
        ("segment_id", ogr.OFTString, 80),
        ("orig_edge", ogr.OFTString, 80),
        ("seg_index", ogr.OFTInteger, None),
        ("length_m", ogr.OFTReal, None),
        ("fclass", ogr.OFTString, 40),
        ("bridge", ogr.OFTString, 20),
        ("tunnel", ogr.OFTString, 20),
        ("layer", ogr.OFTReal, None),
        ("repair_typ", ogr.OFTString, 30),
        ("coverage", ogr.OFTString, 40),
        ("confidence", ogr.OFTString, 20),
        ("near_dist", ogr.OFTReal, None),
        ("source_id", ogr.OFTString, 80),
        ("routing", ogr.OFTString, 50),
    ]
    for name, field_type, width in fields:
        definition = ogr.FieldDefn(name, field_type)
        if width is not None:
            definition.SetWidth(width)
        if field_type == ogr.OFTReal:
            definition.SetWidth(18)
            definition.SetPrecision(6)
        if layer.CreateField(definition) != 0:
            raise RuntimeError(f"Cannot create {name}")
    definition = layer.GetLayerDefn()
    layer.StartTransaction()
    try:
        for segment, assignment in tqdm(
            zip(segments, assignments, strict=True),
            total=len(segments),
            desc="Writing formal thermal segment GDB",
            unit="segment",
            dynamic_ncols=True,
        ):
            feature = ogr.Feature(definition)
            feature.SetGeometry(segment.geometry)
            values = {
                "segment_id": segment.segment_id,
                "orig_edge": segment.original_edge_id,
                "seg_index": segment.segment_index,
                "length_m": segment.length_m,
                "fclass": segment.fclass,
                "bridge": segment.bridge,
                "tunnel": segment.tunnel,
                "layer": segment.layer,
                "repair_typ": segment.repair_type,
                "coverage": assignment["coverage_type"],
                "confidence": assignment["coverage_confidence"],
                "near_dist": assignment["nearest_point_distance_m"],
                "source_id": assignment.get("source_edge_id"),
                "routing": assignment["routing_policy"],
            }
            for key, value in values.items():
                if value is not None and not (
                    isinstance(value, float) and not np.isfinite(value)
                ):
                    feature.SetField(key, value)
            if layer.CreateFeature(feature) != 0:
                raise RuntimeError(f"Cannot write {segment.segment_id}")
        layer.CommitTransaction()
    except Exception:
        layer.RollbackTransaction()
        raise
    dataset = None
    return len(segments)


def run_formal_segment_costs(
    *,
    root: Path,
    network_config: dict[str, Any],
    base_rules: dict[str, Any],
    expansion_rules: dict[str, Any],
    overwrite: bool,
    logger: Any,
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    output_dir = root / "routing/data/thermal_edges"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "costs": output_dir / "road_hour_thermal_cost.csv",
        "segments_csv": output_dir / "thermal_segment_coverage.csv",
        "segments_gdb": output_dir / "step26_thermal_segments.gdb",
        "failed": output_dir / "step26_formal_failed_records.csv",
        "summary": output_dir / "step26_formal_summary.json",
        "qc": output_dir / "step26_formal_qc.json",
        "report": root / "routing/reports/STEP26_FORMAL_RUN_REPORT.md",
    }
    for key, path in outputs.items():
        if key != "segments_gdb" and path.exists() and not overwrite:
            raise FileExistsError(f"Formal Step26 output exists: {path}")
    predictions_path = (
        root
        / "training/predictions/full_sample/routing_point_costs_20240729.csv"
    )
    mapping_path = (
        root / "routing/data/point_edge_mapping/point_edge_mapping.csv"
    )
    network_path = (
        root / "routing/data/topology/step24_repaired_network.gdb"
    )
    source_hashes_before = {
        "predictions": sha256_path(predictions_path),
        "mapping": sha256_path(mapping_path),
        "network": sha256_path(network_path),
    }
    failures: list[dict[str, Any]] = []
    coverage = pd.read_csv(
        output_dir / "edge_thermal_coverage_preview.csv",
        dtype={"edge_id": str, "candidate_source_edge_id": str},
        low_memory=False,
    )
    coverage["edge_id"] = coverage["edge_id"].str.strip()
    interpolation = pd.read_csv(
        output_dir / "edge_interpolation_candidates.csv",
        dtype={"edge_id": str, "source_edge_id": str},
        low_memory=False,
    )
    interpolation["edge_id"] = interpolation["edge_id"].str.strip()
    interpolation["source_edge_id"] = interpolation["source_edge_id"].str.strip()
    direct = pd.read_csv(
        output_dir / "road_hour_direct_cost_preview.csv",
        dtype={"edge_id": str},
        low_memory=False,
    )
    direct["edge_id"] = direct["edge_id"].str.strip()
    direct_lookup = {
        (str(row.edge_id), int(row.hour)): row
        for row in direct.itertuples(index=False)
    }
    mapping = pd.read_csv(
        mapping_path,
        dtype={"point_id": str, "edge_id": str},
        low_memory=False,
    )
    predictions = pd.read_csv(
        predictions_path, dtype={"point_id": str}, low_memory=False
    )
    hours = list(map(int, base_rules["hours"]))
    points, means, stds = prediction_cube(predictions, hours)
    point_tree = cKDTree(points[["utm_x", "utm_y"]].to_numpy(float))
    point_index = {value: index for index, value in enumerate(points["point_id"])}
    matched_points = (
        mapping[mapping["match_status"] == "matched"][
            ["point_id", "bridge", "tunnel", "layer"]
        ]
        .drop_duplicates("point_id")
        .copy()
    )
    structure_trees: dict[
        tuple[bool, bool, float], tuple[cKDTree, np.ndarray]
    ] = {}
    for key, members in matched_points.groupby(
        matched_points.apply(
            lambda row: _structure_key(
                row["bridge"], row["tunnel"], float(row["layer"])
            ),
            axis=1,
        )
    ):
        if key == (False, False, 0.0):
            continue
        indices = np.array(
            [point_index[point_id] for point_id in members["point_id"]],
            dtype=int,
        )
        coordinates = points.iloc[indices][["utm_x", "utm_y"]].to_numpy(float)
        if len(indices):
            structure_trees[key] = (cKDTree(coordinates), indices)
    priors = build_priors(
        predictions=predictions, mapping=mapping, hours=hours
    )
    target_epsg = int(network_config["spatial"]["target_crs"].split(":")[-1])
    boundary_path = root / network_config["inputs"]["center_boundary_shp"]
    segments = build_center_segments(
        gdb_path=network_path,
        boundary_path=boundary_path,
        coverage=coverage,
        interpolation=interpolation,
        target_epsg=target_epsg,
        target_length_m=float(
            expansion_rules["thermal_segmentation"]["target_segment_length_m"]
        ),
        routing_scope=str(
            network_config["spatial"].get("routing_scope", "center_boundary")
        ),
        failures=failures,
    )
    coordinates = np.array(
        [(segment.midpoint_x, segment.midpoint_y) for segment in segments],
        dtype=float,
    )
    all_distances, all_indices = point_tree.query(
        coordinates,
        k=int(expansion_rules["spatial_interpolation"]["nearest_point_count"]),
    )
    if all_distances.ndim == 1:
        all_distances = all_distances[:, None]
        all_indices = all_indices[:, None]
    interpolation_sources = dict(
        zip(interpolation["edge_id"], interpolation["source_edge_id"])
    )
    levels = expansion_rules["spatial_interpolation"]["levels"]
    uncertainty_rules = expansion_rules["uncertainty"]
    assignments: list[dict[str, Any]] = []
    spatial_payloads: list[
        tuple[np.ndarray, np.ndarray, float] | None
    ] = []
    for segment_index, segment in enumerate(
        tqdm(
            segments,
            desc="Classifying approved thermal segments",
            unit="segment",
            dynamic_ncols=True,
        )
    ):
        if segment.source_coverage == "direct":
            assignment = {
                "coverage_type": "direct",
                "coverage_confidence": "high",
                "routing_policy": "normal",
                "nearest_point_distance_m": 0.0,
                "source_edge_id": segment.original_edge_id,
                "uncertainty_multiplier": 1.0,
            }
            payload = None
        elif segment.source_coverage == "interpolation_candidate":
            assignment = {
                "coverage_type": "network_interpolated",
                "coverage_confidence": "medium",
                "routing_policy": "normal",
                "nearest_point_distance_m": float(
                    coverage.loc[
                        coverage["edge_id"] == segment.original_edge_id,
                        "interpolation_distance_m",
                    ].iloc[0]
                ),
                "source_edge_id": interpolation_sources[
                    segment.original_edge_id
                ],
                "uncertainty_multiplier": 1.15,
            }
            payload = None
        else:
            selected_distances = all_distances[segment_index]
            selected_indices = all_indices[segment_index]
            structure = _structure_key(
                segment.bridge, segment.tunnel, segment.layer
            )
            if not _is_surface(
                segment.bridge, segment.tunnel, segment.layer
            ) and structure in structure_trees:
                structure_tree, structure_indices = structure_trees[structure]
                count = min(
                    int(
                        expansion_rules["spatial_interpolation"][
                            "nearest_point_count"
                        ]
                    ),
                    len(structure_indices),
                )
                selected_distances, local_indices = structure_tree.query(
                    [[segment.midpoint_x, segment.midpoint_y]], k=count
                )
                selected_distances = np.atleast_1d(selected_distances).reshape(-1)
                selected_indices = structure_indices[
                    np.atleast_1d(local_indices).reshape(-1)
                ]
            nearest_distance = float(np.min(selected_distances))
            level = _distance_level(nearest_distance, levels)
            allowed_spatial = _is_surface(
                segment.bridge, segment.tunnel, segment.layer
            ) or structure in structure_trees
            if level is not None and allowed_spatial:
                coverage_type, confidence, routing_policy, upper = level
                if not _is_surface(
                    segment.bridge, segment.tunnel, segment.layer
                ):
                    coverage_type = "structure_interpolated"
                    confidence = "low"
                weights = 1.0 / np.square(
                    np.maximum(selected_distances.astype(float), 1.0)
                )
                payload = (
                    selected_indices.astype(int),
                    weights,
                    upper,
                )
                assignment = {
                    "coverage_type": coverage_type,
                    "coverage_confidence": confidence,
                    "routing_policy": routing_policy,
                    "nearest_point_distance_m": nearest_distance,
                    "source_edge_id": None,
                    "uncertainty_multiplier": _distance_multiplier(
                        coverage_type,
                        nearest_distance,
                        upper,
                        uncertainty_rules,
                    ),
                }
            else:
                payload = None
                assignment = {
                    "coverage_type": "prior_imputed",
                    "coverage_confidence": "very_low",
                    "routing_policy": expansion_rules["remaining_unknown"][
                        "routing_policy"
                    ],
                    "nearest_point_distance_m": nearest_distance,
                    "source_edge_id": None,
                    "uncertainty_multiplier": float(
                        uncertainty_rules["prior_imputed_multiplier"]
                    ),
                }
        assignments.append(assignment)
        spatial_payloads.append(payload)
    coverage_rows = []
    for segment, assignment in zip(segments, assignments, strict=True):
        coverage_rows.append(
            {
                "segment_id": segment.segment_id,
                "original_edge_id": segment.original_edge_id,
                "segment_index": segment.segment_index,
                "length_m": segment.length_m,
                "fclass": segment.fclass,
                "bridge": segment.bridge,
                "tunnel": segment.tunnel,
                "layer": segment.layer,
                "repair_type": segment.repair_type,
                **assignment,
                "formal_edge_cost_written": True,
            }
        )
    atomic_csv(outputs["segments_csv"], coverage_rows)
    write_segment_gdb(
        outputs["segments_gdb"],
        segments,
        assignments,
        target_epsg=target_epsg,
        overwrite=overwrite,
    )
    cost_columns = [
        "segment_id",
        "original_edge_id",
        "hour",
        "mode_network",
        "length_m",
        "shade_mean",
        "shade_std",
        "tmrt_mean",
        "tmrt_std",
        "utci_mean",
        "utci_std",
        "shade_discomfort_cost",
        "tmrt_cost",
        "utci_cost",
        "uncertainty_cost",
        "tmrt_cost_norm",
        "utci_cost_norm",
        "uncertainty_cost_norm",
        "support_point_count",
        "nearest_point_distance_m",
        "coverage_type",
        "source_edge_id",
        "interpolation_distance_m",
        "coverage_confidence",
        "routing_policy",
        "uncertainty_multiplier",
        "formal_edge_cost",
    ]
    temporary_cost = outputs["costs"].with_suffix(".csv.tmp")
    point_tmrt_min = float(predictions["tmrt_pred_mean"].min())
    point_tmrt_range = float(
        predictions["tmrt_pred_mean"].max() - point_tmrt_min
    )
    point_utci_min = float(predictions["utci_pred_mean"].min())
    point_utci_range = float(
        predictions["utci_pred_mean"].max() - point_utci_min
    )
    raw_uncertainty_values = np.sqrt(
        np.square(predictions["shade_pred_std"].to_numpy(float))
        + np.square(
            predictions["tmrt_pred_std"].to_numpy(float) / point_tmrt_range
        )
        + np.square(
            predictions["utci_pred_std"].to_numpy(float) / point_utci_range
        )
    )
    uncertainty_min = float(raw_uncertainty_values.min())
    uncertainty_range = float(
        raw_uncertainty_values.max() - uncertainty_min
    )
    row_count = 0
    nonfinite_count = 0
    with temporary_cost.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=cost_columns)
        writer.writeheader()
        for segment_index, (segment, assignment, payload) in enumerate(
            tqdm(
                zip(segments, assignments, spatial_payloads, strict=True),
                total=len(segments),
                desc="Writing formal hourly thermal costs",
                unit="segment",
                dynamic_ncols=True,
            )
        ):
            if assignment["coverage_type"] in {
                "direct",
                "network_interpolated",
            }:
                source_edge_id = str(assignment["source_edge_id"])
                hourly_values = []
                for hour in hours:
                    source = direct_lookup[(source_edge_id, hour)]
                    hourly_values.append(
                        (
                            np.array(
                                [
                                    source.shade_mean,
                                    source.tmrt_mean,
                                    source.utci_mean,
                                ],
                                dtype=float,
                            ),
                            np.array(
                                [
                                    source.shade_std,
                                    source.tmrt_std,
                                    source.utci_std,
                                ],
                                dtype=float,
                            )
                            * float(assignment["uncertainty_multiplier"]),
                            int(source.support_point_count),
                        )
                    )
            elif payload is not None:
                indices, weights, _ = payload
                output_mean, output_std = mixture(
                    means[indices], stds[indices], weights
                )
                output_std *= float(assignment["uncertainty_multiplier"])
                hourly_values = [
                    (output_mean[index], output_std[index], len(indices))
                    for index in range(len(hours))
                ]
            else:
                hourly_values = []
                for hour in hours:
                    prior = priors.get(
                        (segment.fclass, hour),
                        priors[("__global__", hour)],
                    )
                    hourly_values.append(
                        (
                            prior[0],
                            prior[1]
                            * float(assignment["uncertainty_multiplier"]),
                            prior[2],
                        )
                    )
            for hour_index, hour in enumerate(hours):
                output_mean, output_std, support_count = hourly_values[
                    hour_index
                ]
                shade_mean, tmrt_mean, utci_mean = map(float, output_mean)
                shade_std, tmrt_std, utci_std = map(float, output_std)
                uncertainty_cost = math.sqrt(
                    shade_std**2
                    + (tmrt_std / point_tmrt_range) ** 2
                    + (utci_std / point_utci_range) ** 2
                )
                row = {
                    "segment_id": segment.segment_id,
                    "original_edge_id": segment.original_edge_id,
                    "hour": hour,
                    "mode_network": "thermal_comfort_relevant",
                    "length_m": segment.length_m,
                    "shade_mean": shade_mean,
                    "shade_std": shade_std,
                    "tmrt_mean": tmrt_mean,
                    "tmrt_std": tmrt_std,
                    "utci_mean": utci_mean,
                    "utci_std": utci_std,
                    "shade_discomfort_cost": 1.0 - shade_mean,
                    "tmrt_cost": tmrt_mean,
                    "utci_cost": utci_mean,
                    "uncertainty_cost": uncertainty_cost,
                    "tmrt_cost_norm": (tmrt_mean - point_tmrt_min)
                    / point_tmrt_range,
                    "utci_cost_norm": (utci_mean - point_utci_min)
                    / point_utci_range,
                    "uncertainty_cost_norm": (
                        uncertainty_cost - uncertainty_min
                    )
                    / uncertainty_range,
                    "support_point_count": support_count,
                    "nearest_point_distance_m": assignment[
                        "nearest_point_distance_m"
                    ],
                    "coverage_type": assignment["coverage_type"],
                    "source_edge_id": assignment["source_edge_id"],
                    "interpolation_distance_m": (
                        assignment["nearest_point_distance_m"]
                        if assignment["coverage_type"]
                        == "network_interpolated"
                        else 0.0
                    ),
                    "coverage_confidence": assignment[
                        "coverage_confidence"
                    ],
                    "routing_policy": assignment["routing_policy"],
                    "uncertainty_multiplier": assignment[
                        "uncertainty_multiplier"
                    ],
                    "formal_edge_cost": True,
                }
                numeric = [
                    row[column]
                    for column in (
                        "length_m",
                        "shade_mean",
                        "shade_std",
                        "tmrt_mean",
                        "tmrt_std",
                        "utci_mean",
                        "utci_std",
                    )
                ]
                if not all(np.isfinite(numeric)):
                    nonfinite_count += 1
                writer.writerow(row)
                row_count += 1
    os.replace(temporary_cost, outputs["costs"])
    if failures:
        atomic_csv(outputs["failed"], failures)
    else:
        atomic_text(
            outputs["failed"],
            "category,object_id,error_type,error_message\n",
        )
    coverage_frame = pd.DataFrame(coverage_rows)
    total_length = float(coverage_frame["length_m"].sum())
    coverage_stats = {}
    for key, members in coverage_frame.groupby("coverage_type"):
        coverage_stats[str(key)] = {
            "segment_count": int(len(members)),
            "length_km": float(members["length_m"].sum() / 1000),
            "length_ratio": float(members["length_m"].sum() / total_length),
        }
    source_hashes_after = {
        "predictions": sha256_path(predictions_path),
        "mapping": sha256_path(mapping_path),
        "network": sha256_path(network_path),
    }
    qc = {
        "success": (
            len(failures) == 0
            and len(segments) > 0
            and row_count == len(segments) * len(hours)
            and nonfinite_count == 0
            and source_hashes_before == source_hashes_after
        ),
        "segment_count": len(segments),
        "expected_edge_hour_count": len(segments) * len(hours),
        "actual_edge_hour_count": row_count,
        "hours_per_segment": len(hours),
        "nonfinite_thermal_row_count": nonfinite_count,
        "failed_record_count": len(failures),
        "source_hashes_unchanged": source_hashes_before == source_hashes_after,
        "zero_fill_performed": False,
        "route_search_performed": False,
        "coverage_stats": coverage_stats,
    }
    atomic_json(outputs["qc"], qc)
    ended_at = now_text()
    summary = {
        "phase": "G5",
        "step": 26,
        "mode": "run",
        "script": "Step26_build_hourly_edge_costs.py",
        "script_version": "2.0.0",
        "rule_version": expansion_rules["rule_version"],
        "rule_status": expansion_rules["status"],
        "approved_by": expansion_rules["approval"]["approved_by"],
        "approved_at": expansion_rules["approval"]["approved_at"],
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "success": qc["success"],
        "ready_for_step27_check": qc["success"],
        "center_thermal_segment_count": len(segments),
        "routing_scope": str(
            network_config["spatial"].get("routing_scope", "center_boundary")
        ),
        "formal_edge_hour_count": row_count,
        "hours": hours,
        "center_network_length_km": total_length / 1000,
        "coverage_stats": coverage_stats,
        "prior_imputed_length_ratio": coverage_stats.get(
            "prior_imputed", {}
        ).get("length_ratio", 0.0),
        "failed_record_count": len(failures),
        "source_predictions_modified": False,
        "step25_mapping_modified": False,
        "step24_network_modified": False,
        "zero_fill_performed": False,
        "route_search_performed": False,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step26 Formal Thermal Segment Cost Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `SUCCESS = {str(qc['success']).upper()}`",
        f"- `READY_FOR_STEP27_CHECK = {str(qc['success']).upper()}`",
        f"- Center thermal segments: {len(segments):,}",
        f"- Formal segment-hour rows: {row_count:,}",
        f"- Center network length: {total_length / 1000:,.3f} km",
        f"- Failed records: {len(failures):,}",
        f"- Non-finite thermal rows: {nonfinite_count:,}",
        f"- Source hashes unchanged: {source_hashes_before == source_hashes_after}",
        "",
        "## Coverage",
        "",
        "| Coverage type | Segments | Length (km) | Length ratio |",
        "|---|---:|---:|---:|",
    ]
    for key, value in sorted(coverage_stats.items()):
        lines.append(
            f"| {key} | {value['segment_count']:,} | "
            f"{value['length_km']:,.3f} | {value['length_ratio']:.4%} |"
        )
    lines.extend(
        [
            "",
            "Roads were split into at most approximately 50 m thermal-cost "
            "segments. Direct and approved one-hop values were retained. "
            "Surface spatial estimates use four-point inverse-distance "
            "weighting with distance-dependent uncertainty. Bridge, tunnel "
            "and nonzero-layer segments do not inherit ordinary surface "
            "points. Prior-imputed segments are marked very-low confidence "
            "and fallback-only; no record was zero-filled.",
        ]
    )
    atomic_text(outputs["report"], "\n".join(lines) + "\n")
    logger.info(
        "Formal Step26 complete: success=%s segments=%s rows=%s prior_ratio=%.6f",
        qc["success"],
        len(segments),
        row_count,
        summary["prior_imputed_length_ratio"],
    )
    return summary
