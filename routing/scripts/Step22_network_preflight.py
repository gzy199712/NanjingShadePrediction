"""Step22: read-only road, topology, mode and thermal-coverage audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import psutil
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.src.mode_classification import (  # noqa: E402
    assign_candidate_status,
    rule_template,
)
from routing.src.reporting import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_text,
    write_audit_gdb,
    write_reports,
)
from routing.src.road_geometry_qc import (  # noqa: E402
    bridge_tunnel_inventory,
    flag_value,
    geometry_qc,
    road_class_inventory,
)
from routing.src.road_schema import (  # noqa: E402
    companion_file_qc,
    describe_dataset,
    field_inventory,
    load_points,
    load_projected_roads,
    load_union_geometry,
)
from routing.src.thermal_coverage import (  # noqa: E402
    center_scope_comparison,
    thermal_coverage,
)
from routing.src.topology_audit import (  # noqa: E402
    audit_network_scopes,
    dead_end_inventory,
    grade_separation_crossings,
    snap_candidates,
)

SCRIPT_VERSION = "1.0.0"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase G1 read-only network preflight."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/phase_g_network.yaml",
    )
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_g_step22")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for handler in (
        logging.FileHandler(
            path, mode="w" if overwrite else "a", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def dependency_inventory() -> list[dict[str, Any]]:
    modules = [
        "numpy",
        "pandas",
        "networkx",
        "psutil",
        "tqdm",
        "yaml",
        "osgeo",
        "shapely",
        "geopandas",
        "arcpy",
    ]
    rows: list[dict[str, Any]] = []
    for name in tqdm(
        modules,
        desc="Checking routing dependencies",
        unit="package",
        dynamic_ncols=True,
    ):
        if name == "arcpy":
            spec = importlib.util.find_spec("arcpy")
            rows.append(
                {
                    "check_category": "dependency",
                    "dataset": "arcpy35_environment",
                    "check": name,
                    "status": "AVAILABLE" if spec is not None else "NOT_AVAILABLE",
                    "detail": (
                        "module discoverable; not imported during read-only audit"
                        if spec is not None
                        else "module not discoverable"
                    ),
                }
            )
            continue
        try:
            module = __import__(name)
            status = "AVAILABLE"
            detail = getattr(module, "__version__", "available")
        except Exception as error:
            status = "NOT_AVAILABLE"
            detail = str(error)
        rows.append(
            {
                "check_category": "dependency",
                "dataset": "arcpy35_environment",
                "check": name,
                "status": status,
                "detail": detail,
            }
        )
    return rows


def dataset_qc_rows(
    descriptions: dict[str, dict[str, Any]],
    companions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, description in tqdm(
        descriptions.items(),
        desc="Summarizing spatial inputs",
        unit="dataset",
        dynamic_ncols=True,
    ):
        rows.append(
            {
                "check_category": "dataset",
                "dataset": name,
                "check": "readable_geometry_crs_extent",
                "status": "PASS",
                "detail": json.dumps(description, ensure_ascii=False),
            }
        )
    for row in companions:
        passed = row["exists"] or not row["required"]
        rows.append(
            {
                "check_category": "companion_file",
                "dataset": row["dataset"],
                "check": row["component"],
                "status": "PASS" if passed else "FAIL",
                "detail": json.dumps(row, ensure_ascii=False),
            }
        )
    return rows


def run_check(
    config: dict[str, Any],
    *,
    overwrite: bool,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    del resume
    started_at = now_text()
    started = time.perf_counter()
    root = Path(str(config["project_root"])).resolve()
    inputs = config["inputs"]
    spatial = config["spatial"]
    outputs = config["outputs"]
    runtime = config["runtime"]
    target_epsg = int(str(spatial["target_crs"]).split(":")[-1])
    road_path = resolve(root, inputs["osm_road_shp"])
    boundary_path = resolve(root, inputs["center_boundary_shp"])
    point_path = resolve(root, inputs["thermal_points_shp"])
    costs_path = resolve(root, inputs["routing_point_costs_csv"])
    output_root = resolve(root, outputs["output_root"])
    reports_dir = resolve(root, outputs["reports_dir"])
    gdb_path = resolve(root, outputs["audit_gdb"])
    output_root.mkdir(parents=True, exist_ok=True)
    expected_files = [
        output_root / name
        for name in (
            "road_schema_qc.csv",
            "road_field_inventory.csv",
            "road_class_inventory.csv",
            "bridge_tunnel_inventory.csv",
            "road_geometry_qc.csv",
            "connected_components_qc.csv",
            "network_degree_qc.csv",
            "dead_end_inventory.csv",
            "snap_tolerance_simulation.csv",
            "grade_separation_crossing_qc.csv",
            "mode_classification_rule_template.csv",
            "thermal_coverage_by_distance.csv",
            "road_thermal_coverage_qc.csv",
            "center_scope_comparison.csv",
            "step22_failed_records.csv",
        )
    ]
    report_files = [
        reports_dir / name
        for name in (
            "STEP22_NETWORK_PREFLIGHT_REPORT.md",
            "ROAD_CLASSIFICATION_REVIEW.md",
            "TOPOLOGY_AUDIT_REPORT.md",
            "THERMAL_COVERAGE_AUDIT_REPORT.md",
        )
    ]
    if (
        any(path.exists() for path in expected_files + report_files)
        or gdb_path.exists()
    ) and not overwrite:
        raise FileExistsError("Step22 outputs exist; pass --overwrite")
    failed_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    logger.info("Phase G1 read-only audit started")
    dependencies = dependency_inventory()
    missing_optional = [
        row["check"]
        for row in dependencies
        if row["status"] == "NOT_AVAILABLE"
        and row["check"] in {"shapely", "geopandas", "arcpy"}
    ]
    if missing_optional:
        warnings.append(
            "Protected arcpy35 environment lacks "
            + ", ".join(missing_optional)
            + "; OGR/GDAL and NetworkX equivalents were used without installation."
        )
    companions = []
    for path in tqdm(
        (road_path, boundary_path, point_path),
        desc="Checking Shapefile packages",
        unit="dataset",
        dynamic_ncols=True,
    ):
        companions.extend(companion_file_qc(path))
    required_missing = [
        row["path"]
        for row in companions
        if row["required"] and not row["exists"]
    ]
    if required_missing:
        blockers.append("Missing required Shapefile components: " + ", ".join(required_missing))
    descriptions: dict[str, dict[str, Any]] = {}
    for name, path in tqdm(
        (
            ("osm_roads", road_path),
            ("center_boundary", boundary_path),
            ("thermal_points", point_path),
        ),
        desc="Reading spatial dataset metadata",
        unit="dataset",
        dynamic_ncols=True,
    ):
        try:
            descriptions[name] = describe_dataset(path)
        except Exception as error:
            blockers.append(f"{name} cannot be read: {error}")
            failed_rows.append(
                {
                    "category": "dataset",
                    "object_id": name,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    if blockers:
        raise RuntimeError("; ".join(blockers))
    field_rows = field_inventory(road_path)
    available = {
        str(row["field_name"]).lower()
        for row in field_rows
        if row["availability"] == "AVAILABLE"
    }
    if not {"fclass", "fclass_cn"} & available:
        blockers.append("Neither fclass nor fclass_cn is available")
    if not {"bridge", "tunnel"} <= available:
        blockers.append("bridge/tunnel fields are unavailable")
    missing_access = sorted(
        {"access", "foot", "bicycle", "surface", "z_level", "level"} - available
    )
    if missing_access:
        warnings.append(
            "Mode/access fields marked NOT_AVAILABLE: " + ", ".join(missing_access)
        )
    center = load_union_geometry(boundary_path, target_epsg)
    roads, projection_info = load_projected_roads(
        road_path, target_epsg, center
    )
    if projection_info["projection_copy_required"]:
        warnings.append(
            "OSM roads are not EPSG:32650; only an in-memory/audit projection copy was used."
        )
    if len(roads) < 1000:
        blockers.append(f"Road network scale is suspicious: {len(roads)} roads")
    invalid_load_ratio = projection_info["failed_count"] / max(
        projection_info["source_count"], 1
    )
    if invalid_load_ratio > float(runtime["maximum_invalid_geometry_ratio"]):
        blockers.append(
            f"Too many road geometries failed loading: {invalid_load_ratio:.3%}"
        )
    assign_candidate_status(roads)
    points = load_points(point_path, target_epsg)
    if len(points) != int(runtime["expected_thermal_points"]):
        blockers.append(
            f"Thermal point count mismatch: {len(points)}"
        )
    costs = pd.read_csv(costs_path, low_memory=False)
    cost_points = costs["point_id"].astype(str).nunique()
    cost_hours = sorted(costs["hour"].astype(int).unique().tolist())
    if cost_points != int(runtime["expected_thermal_points"]):
        blockers.append(f"Routing cost point count mismatch: {cost_points}")
    if len(cost_hours) != int(runtime["expected_hours"]):
        blockers.append(f"Routing cost hour count mismatch: {cost_hours}")
    geometry_rows, geometry_issue_ids = geometry_qc(
        roads,
        [float(value) for value in spatial["short_segment_thresholds_m"]],
        float(spatial["long_segment_threshold_m"]),
    )
    invalid_geometry_count = sum(
        (not row["valid_geometry"]) or row["invalid_coordinate"]
        for row in geometry_rows
    )
    if invalid_geometry_count / max(len(roads), 1) > float(
        runtime["maximum_invalid_geometry_ratio"]
    ):
        blockers.append(
            f"Invalid geometry ratio exceeds threshold: {invalid_geometry_count}/{len(roads)}"
        )
    bridge_values = {flag_value(road.attrs.get("bridge")) for road in roads}
    tunnel_values = {flag_value(road.attrs.get("tunnel")) for road in roads}
    if not bridge_values <= {"T", "F", "NULL"} or not tunnel_values <= {
        "T",
        "F",
        "NULL",
    }:
        warnings.append(
            f"Unexpected bridge/tunnel values: {bridge_values} / {tunnel_values}"
        )
    coverage_road_rows, coverage_summary_rows, covered = thermal_coverage(
        roads,
        points,
        [float(value) for value in spatial["thermal_match_distance_candidates_m"]],
    )
    if not covered[max(covered)]:
        blockers.append("No reasonable thermal point-road overlap within 75 m")
    topology_rows = audit_network_scopes(
        roads, int(spatial["small_component_edge_threshold"])
    )
    snap_pairs, snap_rows = snap_candidates(
        roads, max(map(float, spatial["topology_tolerance_candidates_m"]))
    )
    dead_end_rows = dead_end_inventory(
        roads,
        center,
        float(spatial["endpoint_search_radius_m"]),
        float(spatial["geometry_grid_size_m"]),
    )
    crossing_rows = grade_separation_crossings(
        roads, float(spatial["geometry_grid_size_m"])
    )
    mode_rows = rule_template(roads)
    class_rows = road_class_inventory(roads, covered[75.0])
    bridge_rows = bridge_tunnel_inventory(roads)
    scope_rows = center_scope_comparison(
        roads,
        points,
        center,
        covered[75.0],
        int(spatial["small_component_edge_threshold"]),
    )
    schema_rows = dependencies + dataset_qc_rows(descriptions, companions)
    atomic_csv(output_root / "road_schema_qc.csv", schema_rows)
    atomic_csv(output_root / "road_field_inventory.csv", field_rows)
    atomic_csv(output_root / "road_class_inventory.csv", class_rows)
    atomic_csv(output_root / "bridge_tunnel_inventory.csv", bridge_rows)
    atomic_csv(output_root / "road_geometry_qc.csv", geometry_rows)
    atomic_csv(output_root / "connected_components_qc.csv", topology_rows)
    atomic_csv(
        output_root / "network_degree_qc.csv",
        [
            {
                key: row[key]
                for key in (
                    "network_scope",
                    "node_count",
                    "edge_count",
                    "degree_1_nodes",
                    "degree_2_nodes",
                    "degree_ge_3_nodes",
                    "isolated_edges",
                    "small_component_count",
                )
            }
            for row in topology_rows
        ],
    )
    atomic_csv(output_root / "dead_end_inventory.csv", dead_end_rows)
    atomic_csv(output_root / "snap_tolerance_simulation.csv", snap_rows)
    atomic_csv(
        output_root / "grade_separation_crossing_qc.csv", crossing_rows
    )
    atomic_csv(
        output_root / "mode_classification_rule_template.csv", mode_rows
    )
    atomic_csv(
        output_root / "thermal_coverage_by_distance.csv",
        coverage_summary_rows,
    )
    atomic_csv(
        output_root / "road_thermal_coverage_qc.csv", coverage_road_rows
    )
    atomic_csv(output_root / "center_scope_comparison.csv", scope_rows)
    if failed_rows:
        atomic_csv(output_root / "step22_failed_records.csv", failed_rows)
    else:
        atomic_text(
            output_root / "step22_failed_records.csv",
            "category,object_id,error_type,error_message\n",
        )
    gdb_counts = write_audit_gdb(
        path=gdb_path,
        roads=roads,
        geometry_qc_rows=geometry_rows,
        dead_ends=dead_end_rows,
        crossings=crossing_rows,
        snap_pairs=snap_pairs,
        coverage_rows=coverage_road_rows,
        overwrite=overwrite,
        target_epsg=target_epsg,
    )
    required_layer_names = {
        "RoadInvalidGeometry",
        "RoadShortSegments",
        "DeadEndCandidates",
        "GradeSeparatedCrossings",
        "PossibleSnapPairs",
        "ThermalCoverageCandidates",
        "UncoveredRoadCandidates",
    }
    if set(gdb_counts) != required_layer_names:
        blockers.append("Audit geodatabase layers are incomplete")
    summary = {
        "phase": "G1",
        "step": 22,
        "mode": "check",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "started_at": started_at,
        "ended_at": now_text(),
        "elapsed_seconds": time.perf_counter() - started,
        "ready_for_mode_rule_review": len(blockers) == 0,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "road_count": len(roads),
        "road_source_feature_count": projection_info["source_count"],
        "road_source_crs": projection_info["source_crs"],
        "road_source_authority": projection_info["source_authority"],
        "audit_projection_required": projection_info[
            "projection_copy_required"
        ],
        "target_epsg": target_epsg,
        "boundary_feature_count": descriptions["center_boundary"][
            "feature_count"
        ],
        "thermal_point_count": len(points),
        "routing_cost_record_count": len(costs),
        "routing_cost_point_count": cost_points,
        "routing_cost_hours": cost_hours,
        "invalid_geometry_count": invalid_geometry_count,
        "geometry_issue_counts": {
            key: len(value) for key, value in geometry_issue_ids.items()
        },
        "dead_end_count": len(dead_end_rows),
        "crossing_count": len(crossing_rows),
        "snap_pair_count": len(snap_pairs),
        "coverage_75m_road_count": len(covered[75.0]),
        "coverage_75m_road_ratio": len(covered[75.0]) / len(roads),
        "gdb_path": str(gdb_path),
        "gdb_layer_counts": gdb_counts,
        "failed_record_count": len(failed_rows),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
            "process_peak_rss_gb": psutil.Process().memory_info().rss / 1024**3,
        },
        "read_only_audit": True,
        "source_roads_modified": False,
        "formal_mode_rules_applied": False,
        "topology_repairs_performed": False,
        "formal_point_edge_match_performed": False,
        "route_search_performed": False,
    }
    report_paths = write_reports(
        reports_dir=reports_dir,
        summary=summary,
        road_class_rows=class_rows,
        topology_rows=topology_rows,
        coverage_rows=coverage_summary_rows,
    )
    summary["report_paths"] = report_paths
    summary_path = output_root / "step22_network_preflight_summary.json"
    atomic_json(summary_path, summary)
    logger.info(
        "Phase G1 complete: ready=%s roads=%s points=%s blockers=%s warnings=%s",
        summary["ready_for_mode_rule_review"],
        len(roads),
        len(points),
        len(blockers),
        len(warnings),
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    root = Path(str(config["project_root"])).resolve()
    logger = setup_logger(
        resolve(root, config["outputs"]["log_dir"]) / "step22.log",
        args.overwrite,
    )
    output_root = resolve(root, config["outputs"]["output_root"])
    failure_path = output_root / "step22_failure.json"
    try:
        if args.mode == "run":
            raise PermissionError(
                "Phase G1 currently authorizes --mode check only; Phase G2 "
                "requires manual rule review and explicit approval."
            )
        summary = run_check(
            config,
            overwrite=args.overwrite,
            resume=args.resume,
            logger=logger,
        )
        if failure_path.exists():
            failure_path.unlink()
        terminal = {
            "READY_FOR_MODE_RULE_REVIEW": summary[
                "ready_for_mode_rule_review"
            ],
            "road_count": summary["road_count"],
            "thermal_point_count": summary["thermal_point_count"],
            "routing_cost_record_count": summary["routing_cost_record_count"],
            "invalid_geometry_count": summary["invalid_geometry_count"],
            "dead_end_count": summary["dead_end_count"],
            "crossing_count": summary["crossing_count"],
            "snap_pair_count": summary["snap_pair_count"],
            "coverage_75m_road_count": summary["coverage_75m_road_count"],
            "blocker_count": summary["blocker_count"],
            "warning_count": summary["warning_count"],
            "report_paths": summary["report_paths"],
            "audit_gdb": summary["gdb_path"],
            "formal_routing_performed": False,
        }
        print(json.dumps(terminal, ensure_ascii=False, indent=2))
        return 0 if summary["ready_for_mode_rule_review"] else 2
    except Exception as error:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            failure_path,
            {
                "status": "FAIL",
                "time": now_text(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        logger.exception("Step22 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
