"""Atomic Phase G1 outputs, audit geodatabase and Markdown reports."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from routing.src.road_schema import RoadFeature


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_csv(
        temporary, index=False, encoding="utf-8-sig"
    )
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _create_fields(layer: Any, rows: list[dict[str, Any]]) -> list[str]:
    from osgeo import ogr

    columns = list(rows[0]) if rows else ["audit_id"]
    for column in tqdm(
        columns,
        desc=f"Creating {layer.GetName()} fields",
        unit="field",
        dynamic_ncols=True,
    ):
        values = [row.get(column) for row in rows if row.get(column) is not None]
        if values and all(isinstance(value, (bool, int, np.integer)) for value in values):
            field = ogr.FieldDefn(column[:64], ogr.OFTInteger)
        elif values and all(
            isinstance(value, (bool, int, float, np.number)) for value in values
        ):
            field = ogr.FieldDefn(column[:64], ogr.OFTReal)
            field.SetWidth(18)
            field.SetPrecision(6)
        else:
            field = ogr.FieldDefn(column[:64], ogr.OFTString)
            field.SetWidth(254)
        if layer.CreateField(field) != 0:
            raise RuntimeError(f"Cannot create {layer.GetName()}.{column}")
    return columns


def _set_fields(feature: Any, columns: list[str], row: dict[str, Any]) -> None:
    for column in columns:
        value = row.get(column)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        elif isinstance(value, np.generic):
            value = value.item()
        feature.SetField(column[:64], value)


def _write_layer(
    dataset: Any,
    name: str,
    geometry_type: int,
    spatial_reference: Any,
    rows: list[dict[str, Any]],
    geometry_builder: Any,
) -> int:
    from osgeo import ogr

    layer = dataset.CreateLayer(name, spatial_reference, geometry_type)
    columns = _create_fields(layer, rows)
    definition = layer.GetLayerDefn()
    layer.StartTransaction()
    try:
        for row in tqdm(
            rows,
            desc=f"Writing {name}",
            unit="feature",
            dynamic_ncols=True,
        ):
            feature = ogr.Feature(definition)
            geometry = geometry_builder(row)
            if geometry is not None:
                feature.SetGeometry(geometry)
            _set_fields(feature, columns, row)
            if layer.CreateFeature(feature) != 0:
                raise RuntimeError(f"Cannot write {name}")
        layer.CommitTransaction()
    except Exception:
        layer.RollbackTransaction()
        raise
    return len(rows)


def write_audit_gdb(
    *,
    path: Path,
    roads: list[RoadFeature],
    geometry_qc_rows: list[dict[str, Any]],
    dead_ends: list[dict[str, Any]],
    crossings: list[dict[str, Any]],
    snap_pairs: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    overwrite: bool,
    target_epsg: int,
) -> dict[str, int]:
    from osgeo import ogr, osr

    ogr.UseExceptions()
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Audit geodatabase exists: {path}")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = ogr.GetDriverByName("OpenFileGDB")
    dataset = driver.CreateDataSource(str(path))
    if dataset is None:
        raise RuntimeError(f"Cannot create {path}")
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(target_epsg)
    road_lookup = {road.edge_id: road for road in roads}

    def road_geometry(row: dict[str, Any]) -> Any:
        edge_id = str(row.get("edge_id") or row.get("source_edge_id"))
        road = road_lookup.get(edge_id)
        return road.geometry.Clone() if road is not None else None

    def point_geometry(row: dict[str, Any]) -> Any:
        geometry = ogr.Geometry(ogr.wkbPoint)
        geometry.AddPoint(float(row["x"]), float(row["y"]))
        return geometry

    def crossing_geometry(row: dict[str, Any]) -> Any:
        return point_geometry(row)

    def snap_geometry(row: dict[str, Any]) -> Any:
        geometry = ogr.Geometry(ogr.wkbLineString)
        geometry.AddPoint(float(row["x_a"]), float(row["y_a"]))
        geometry.AddPoint(float(row["x_b"]), float(row["y_b"]))
        return geometry

    invalid = [
        row
        for row in geometry_qc_rows
        if not row["valid_geometry"] or row["invalid_coordinate"]
    ]
    short = [row for row in geometry_qc_rows if row["shorter_than_5m"]]
    covered = [
        row for row in coverage_rows if row.get("covered_within_75m")
    ]
    uncovered = [
        row for row in coverage_rows if not row.get("covered_within_75m")
    ]
    layers = [
        ("RoadInvalidGeometry", ogr.wkbLineString, invalid, road_geometry),
        ("RoadShortSegments", ogr.wkbLineString, short, road_geometry),
        ("DeadEndCandidates", ogr.wkbPoint, dead_ends, point_geometry),
        (
            "GradeSeparatedCrossings",
            ogr.wkbPoint,
            crossings,
            crossing_geometry,
        ),
        ("PossibleSnapPairs", ogr.wkbLineString, snap_pairs, snap_geometry),
        (
            "ThermalCoverageCandidates",
            ogr.wkbLineString,
            covered,
            road_geometry,
        ),
        (
            "UncoveredRoadCandidates",
            ogr.wkbLineString,
            uncovered,
            road_geometry,
        ),
    ]
    counts: dict[str, int] = {}
    for name, geometry_type, rows, builder in tqdm(
        layers,
        desc="Creating road audit geodatabase",
        unit="layer",
        dynamic_ncols=True,
    ):
        counts[name] = _write_layer(
            dataset,
            name,
            geometry_type,
            spatial_reference,
            rows,
            builder,
        )
    dataset.FlushCache()
    dataset = None
    return counts


def write_reports(
    *,
    reports_dir: Path,
    summary: dict[str, Any],
    road_class_rows: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> list[str]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    main = reports_dir / "STEP22_NETWORK_PREFLIGHT_REPORT.md"
    classes = reports_dir / "ROAD_CLASSIFICATION_REVIEW.md"
    topology = reports_dir / "TOPOLOGY_AUDIT_REPORT.md"
    coverage = reports_dir / "THERMAL_COVERAGE_AUDIT_REPORT.md"
    atomic_text(
        main,
        "\n".join(
            [
                "# Step22 Network Preflight Report",
                "",
                f"Generated: {summary['ended_at']}",
                "",
                f"- `READY_FOR_MODE_RULE_REVIEW = {str(summary['ready_for_mode_rule_review']).upper()}`",
                f"- Blocking errors / warnings: {summary['blocker_count']} / {summary['warning_count']}",
                f"- Roads loaded: {summary['road_count']:,}",
                f"- Road CRS: {summary['road_source_crs']} (audit projection EPSG:{summary['target_epsg']})",
                f"- Center boundary features: {summary['boundary_feature_count']}",
                f"- Thermal points: {summary['thermal_point_count']:,}",
                f"- Routing point-hour records: {summary['routing_cost_record_count']:,}",
                f"- Invalid geometries: {summary['invalid_geometry_count']:,}",
                f"- Dead-end candidates: {summary['dead_end_count']:,}",
                f"- 2D crossing candidates: {summary['crossing_count']:,}",
                f"- Possible snap pairs within 3 m: {summary['snap_pair_count']:,}",
                "",
                "This was a read-only audit. No source road was modified or deleted; no final mode rule, topology repair, point-edge match, thermal edge cost, A*, Dijkstra, or route search was performed.",
                "",
                "## Dependency note",
                "",
                "- Shapely and GeoPandas are not installed in the protected arcpy35 environment.",
                "- They were not installed or upgraded. OGR/GDAL and NetworkX equivalents were used.",
                "",
                "## Decision",
                "",
                (
                    "The mandatory inputs and outputs passed. Phase G2 may begin only after manual review and explicit approval."
                    if summary["ready_for_mode_rule_review"]
                    else "One or more blocking checks failed. Phase G2 must not begin."
                ),
                "",
                "## Warnings",
                "",
                *[f"- {value}" for value in summary["warnings"]],
            ]
        )
        + "\n",
    )
    class_lines = [
        "# Road Classification Review",
        "",
        "All statuses are non-binding candidates. No road was removed.",
        "",
        "| fclass | fclass_cn | count | length km | walk | bike | shared | review |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in tqdm(
        road_class_rows,
        desc="Writing road class review",
        unit="class",
        dynamic_ncols=True,
    ):
        class_lines.append(
            f"| {row['fclass']} | {row['fclass_cn']} | {row['feature_count']} | "
            f"{row['total_length_km']:.3f} | {row['candidate_walk_status']} | "
            f"{row['candidate_bike_status']} | {row['candidate_shared_status']} | "
            f"{row['review_required']} |"
        )
    atomic_text(classes, "\n".join(class_lines) + "\n")
    topology_lines = [
        "# Topology Audit Report",
        "",
        "Graphs use original line endpoints only. Interior 2D crossings were not split or connected.",
        "",
        "| scope | nodes | edges | components | largest node ratio | degree 1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in tqdm(
        topology_rows,
        desc="Writing topology report",
        unit="scope",
        dynamic_ncols=True,
    ):
        topology_lines.append(
            f"| {row['network_scope']} | {row['node_count']} | {row['edge_count']} | "
            f"{row['connected_component_count']} | {row['largest_component_node_ratio']:.6f} | "
            f"{row['degree_1_nodes']} |"
        )
    atomic_text(topology, "\n".join(topology_lines) + "\n")
    overall = [
        row
        for row in coverage_rows
        if row["group_type"] == "overall"
    ]
    coverage_lines = [
        "# Thermal Coverage Audit Report",
        "",
        "Distances are candidate support tests only; no formal point-edge match was created.",
        "",
        "| radius m | matched roads | matched length km | feature ratio | length ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in tqdm(
        overall,
        desc="Writing thermal coverage report",
        unit="radius",
        dynamic_ncols=True,
    ):
        coverage_lines.append(
            f"| {row['match_distance_m']} | {row['matched_road_count']} | "
            f"{row['matched_road_length_km']:.3f} | "
            f"{row['feature_coverage_ratio']:.6f} | {row['length_coverage_ratio']:.6f} |"
        )
    atomic_text(coverage, "\n".join(coverage_lines) + "\n")
    return [str(main), str(classes), str(topology), str(coverage)]
