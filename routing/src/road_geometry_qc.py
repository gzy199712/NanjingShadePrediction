"""Road geometry, class, and bridge/tunnel inventories."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from tqdm import tqdm

from routing.src.road_schema import RoadFeature


def flag_value(value: Any) -> str:
    text = str(value).strip().upper() if value is not None else ""
    if text in {"T", "TRUE", "1", "YES", "Y"}:
        return "T"
    if text in {"F", "FALSE", "0", "NO", "N"}:
        return "F"
    if not text:
        return "NULL"
    return f"INVALID:{text}"


def geometry_qc(
    roads: list[RoadFeature],
    short_thresholds: list[float],
    long_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    base_rows: list[dict[str, Any]] = []
    issue_ids: dict[str, list[str]] = defaultdict(list)
    for road in tqdm(
        roads,
        desc="Checking road geometry quality",
        unit="road",
        dynamic_ncols=True,
    ):
        digest = hashlib.sha256(bytes(road.geometry.ExportToWkb())).hexdigest()
        hashes[digest].append(road.edge_id)
        multipart = road.geometry.GetGeometryName().upper().startswith("MULTI")
        self_intersection = not bool(road.geometry.IsSimple())
        coordinates = np.asarray(
            [
                coordinate
                for part in (
                    [road.geometry.GetGeometryRef(i) for i in range(road.geometry.GetGeometryCount())]
                    if multipart
                    else [road.geometry]
                )
                for coordinate in [
                    part.GetPoint(index)[:2]
                    for index in range(part.GetPointCount())
                ]
            ],
            dtype=float,
        )
        invalid_coordinate = bool(
            coordinates.size == 0
            or not np.isfinite(coordinates).all()
            or (np.abs(coordinates) > 1.0e8).any()
        )
        row = {
            "edge_id": road.edge_id,
            "source_fid": road.source_fid,
            "fclass": road.attrs.get("fclass"),
            "fclass_cn": road.attrs.get("fclass_cn"),
            "length_m": road.length_m,
            "zero_length": road.length_m == 0,
            "shorter_than_0_5m": road.length_m < 0.5,
            "shorter_than_1m": road.length_m < 1,
            "shorter_than_2m": road.length_m < 2,
            "shorter_than_5m": road.length_m < 5,
            "abnormally_long": road.length_m > long_threshold,
            "valid_geometry": bool(road.geometry.IsValid()),
            "self_intersection": self_intersection,
            "multipart": multipart,
            "invalid_coordinate": invalid_coordinate,
            "duplicate_geometry": False,
            "duplicate_group_size": 1,
        }
        base_rows.append(row)
        if not row["valid_geometry"] or invalid_coordinate:
            issue_ids["invalid"].append(road.edge_id)
        if road.length_m < max(short_thresholds):
            issue_ids["short"].append(road.edge_id)
        if multipart:
            issue_ids["multipart"].append(road.edge_id)
        if self_intersection:
            issue_ids["self_intersection"].append(road.edge_id)
        if row["abnormally_long"]:
            issue_ids["long"].append(road.edge_id)
    duplicate_lookup = {
        edge_id: len(edge_ids)
        for edge_ids in hashes.values()
        if len(edge_ids) > 1
        for edge_id in edge_ids
    }
    for row in tqdm(
        base_rows,
        desc="Annotating duplicate geometries",
        unit="road",
        dynamic_ncols=True,
    ):
        size = duplicate_lookup.get(row["edge_id"], 1)
        row["duplicate_geometry"] = size > 1
        row["duplicate_group_size"] = size
        if size > 1:
            issue_ids["duplicate"].append(row["edge_id"])
    return base_rows, dict(issue_ids)


def road_class_inventory(
    roads: list[RoadFeature],
    covered_by_75m: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RoadFeature]] = defaultdict(list)
    for road in tqdm(
        roads,
        desc="Grouping road classes",
        unit="road",
        dynamic_ncols=True,
    ):
        grouped[
            (
                str(road.attrs.get("fclass") or "NULL"),
                str(road.attrs.get("fclass_cn") or ""),
            )
        ].append(road)
    rows: list[dict[str, Any]] = []
    for (fclass, fclass_cn), members in tqdm(
        sorted(grouped.items()),
        desc="Summarizing road classes",
        unit="class",
        dynamic_ncols=True,
    ):
        lengths = np.asarray([road.length_m for road in members], dtype=float)
        center = [road for road in members if road.inside_center]
        rows.append(
            {
                "fclass": fclass,
                "fclass_cn": fclass_cn,
                "feature_count": len(members),
                "total_length_km": float(lengths.sum() / 1000),
                "mean_length_m": float(lengths.mean()),
                "median_length_m": float(np.median(lengths)),
                "bridge_count": sum(flag_value(road.attrs.get("bridge")) == "T" for road in members),
                "tunnel_count": sum(flag_value(road.attrs.get("tunnel")) == "T" for road in members),
                "center_feature_count": len(center),
                "center_length_km": float(sum(road.length_m for road in center) / 1000),
                "near_thermal_point_count": sum(
                    road.edge_id in covered_by_75m for road in members
                ),
                "candidate_walk_status": members[0].walk_status,
                "candidate_bike_status": members[0].bike_status,
                "candidate_shared_status": members[0].shared_status,
                "review_required": True,
                "review_note": (
                    "Candidate status only; verify legal access, bridge/tunnel "
                    "conditions, names and local transport rules in Phase G2."
                ),
            }
        )
    return rows


def bridge_tunnel_inventory(roads: list[RoadFeature]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[RoadFeature]] = defaultdict(list)
    for road in tqdm(
        roads,
        desc="Grouping bridge and tunnel attributes",
        unit="road",
        dynamic_ncols=True,
    ):
        bridge = flag_value(road.attrs.get("bridge"))
        tunnel = flag_value(road.attrs.get("tunnel"))
        fclass = str(road.attrs.get("fclass") or "NULL")
        fclass_cn = str(road.attrs.get("fclass_cn") or "")
        name = str(road.attrs.get("name") or "")
        grouped[(bridge, tunnel, fclass, fclass_cn, name)].append(road)
    rows: list[dict[str, Any]] = []
    for key, members in tqdm(
        sorted(grouped.items()),
        desc="Summarizing bridge and tunnel inventory",
        unit="group",
        dynamic_ncols=True,
    ):
        bridge, tunnel, fclass, fclass_cn, name = key
        lower_name = name.lower()
        rows.append(
            {
                "bridge": bridge,
                "tunnel": tunnel,
                "fclass": fclass,
                "fclass_cn": fclass_cn,
                "name": name,
                "feature_count": len(members),
                "total_length_km": sum(road.length_m for road in members) / 1000,
                "footway_or_cycleway": fclass in {"footway", "cycleway", "path"},
                "motor_only_candidate": fclass in {
                    "motorway",
                    "motorway_link",
                    "trunk",
                    "trunk_link",
                },
                "cross_river_name_candidate": any(
                    token in lower_name or token in name
                    for token in ("bridge", "桥", "长江", "江")
                ),
                "elevated_name_candidate": any(
                    token in lower_name or token in name
                    for token in ("elevated", "高架", "立交")
                ),
                "abnormal_flag": bridge.startswith("INVALID")
                or tunnel.startswith("INVALID"),
                "manual_review": bridge == "T"
                or tunnel == "T"
                or bridge.startswith("INVALID")
                or tunnel.startswith("INVALID"),
            }
        )
    return rows
