"""Step24: protected topology-repair review and approved execution gate."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.src.reporting import atomic_csv, atomic_json, atomic_text  # noqa: E402

SCRIPT_VERSION = "2.0.0"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase G3 protected topology-repair review."
    )
    parser.add_argument(
        "--network-config",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/phase_g_network.yaml",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/topology_repair_draft.yaml",
    )
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_g_step24")
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


def _node(x: float, y: float) -> tuple[float, float]:
    return round(float(x), 3), round(float(y), 3)


def read_base_graph(
    gdb_path: Path,
    layer_name: str = "ThermalComfortRelevantRoads",
) -> tuple[nx.MultiGraph, int]:
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(gdb_path), 0)
    if dataset is None:
        raise FileNotFoundError(gdb_path)
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise KeyError(layer_name)
    graph = nx.MultiGraph()
    null_geometry = 0
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Reading Step23 thermal road endpoints",
        unit="road",
        dynamic_ncols=True,
    ):
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            null_geometry += 1
            continue
        edge_id = str(feature.GetField("edge_id"))
        if geometry.GetGeometryType() in {
            ogr.wkbMultiLineString,
            ogr.wkbMultiLineString25D,
        }:
            first_part = geometry.GetGeometryRef(0)
            last_part = geometry.GetGeometryRef(geometry.GetGeometryCount() - 1)
            start = first_part.GetPoint(0)
            end = last_part.GetPoint(last_part.GetPointCount() - 1)
        else:
            start = geometry.GetPoint(0)
            end = geometry.GetPoint(geometry.GetPointCount() - 1)
        graph.add_edge(
            _node(start[0], start[1]),
            _node(end[0], end[1]),
            key=edge_id,
            edge_id=edge_id,
        )
    dataset = None
    return graph, null_geometry


def read_snap_pairs(gdb_path: Path) -> pd.DataFrame:
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(gdb_path), 0)
    if dataset is None:
        raise FileNotFoundError(gdb_path)
    layer = dataset.GetLayerByName("PossibleSnapPairs")
    if layer is None:
        raise KeyError("PossibleSnapPairs")
    definition = layer.GetLayerDefn()
    fields = [
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    ]
    rows: list[dict[str, Any]] = []
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Reading Step22 snap candidates",
        unit="pair",
        dynamic_ncols=True,
    ):
        rows.append({field: feature.GetField(field) for field in fields})
    dataset = None
    return pd.DataFrame(rows)


def classify_candidates(
    pairs: pd.DataFrame,
    classification: pd.DataFrame,
    rules: dict[str, Any],
) -> pd.DataFrame:
    lookup = classification.set_index("edge_id").to_dict("index")
    records: list[dict[str, Any]] = []
    angle_limit = float(rules["manual_review"]["direction_difference_over_deg"])
    for row in tqdm(
        pairs.to_dict("records"),
        desc="Applying protected topology rules",
        unit="pair",
        dynamic_ncols=True,
    ):
        edge_a = str(row["edge_id_a"])
        edge_b = str(row["edge_id_b"])
        first = lookup.get(edge_a)
        second = lookup.get(edge_b)
        reasons: list[str] = []
        hard_reject = False
        if edge_a == edge_b:
            reasons.append("same_source_edge")
            hard_reject = True
        if first is None or second is None:
            reasons.append("missing_edge_reference")
            hard_reject = True
        if first is not None and second is not None:
            if (
                first["thermal_comfort_class_preview"]
                != "thermal_comfort_relevant"
                or second["thermal_comfort_class_preview"]
                != "thermal_comfort_relevant"
            ):
                reasons.append("motor_vehicle_only_edge")
                hard_reject = True
            if bool(row["bridge_conflict"]):
                reasons.append("bridge_status_conflict")
                hard_reject = True
            if bool(row["tunnel_conflict"]):
                reasons.append("tunnel_status_conflict")
                hard_reject = True
            layer_a = first.get("layer")
            layer_b = second.get("layer")
            if (
                pd.notna(layer_a)
                and pd.notna(layer_b)
                and not math.isclose(float(layer_a), float(layer_b))
            ):
                reasons.append("layer_conflict")
                hard_reject = True
            if bool(first.get("grade_separated")) != bool(
                second.get("grade_separated")
            ):
                reasons.append("grade_separation_conflict")
                hard_reject = True
        if not hard_reject:
            if not bool(row["same_fclass"]):
                reasons.append("different_fclass_manual_review")
            if bool(row["parallel_candidate"]):
                reasons.append("parallel_or_collinear_manual_review")
            if float(row["direction_difference_deg"]) > angle_limit:
                reasons.append("large_direction_difference_manual_review")
            if not reasons:
                reasons.append("no_hard_conflict_manual_review")
        decision = (
            "hard_reject"
            if hard_reject
            else "eligible_for_manual_approval"
        )
        records.append(
            {
                **row,
                "edge_a_thermal_class": (
                    first.get("thermal_comfort_class_preview")
                    if first is not None
                    else None
                ),
                "edge_b_thermal_class": (
                    second.get("thermal_comfort_class_preview")
                    if second is not None
                    else None
                ),
                "layer_a": first.get("layer") if first is not None else None,
                "layer_b": second.get("layer") if second is not None else None,
                "grade_separated_a": (
                    first.get("grade_separated")
                    if first is not None
                    else None
                ),
                "grade_separated_b": (
                    second.get("grade_separated")
                    if second is not None
                    else None
                ),
                "check_decision": decision,
                "decision_reasons": ";".join(reasons),
                "approved_for_repair": False,
                "repair_performed": False,
            }
        )
    return pd.DataFrame(records)


def metrics(graph: nx.MultiGraph, scope: str) -> dict[str, Any]:
    components = list(nx.connected_components(graph))
    component_edges = [
        graph.subgraph(nodes).number_of_edges() for nodes in components
    ]
    degrees = dict(graph.degree())
    largest_nodes = max((len(nodes) for nodes in components), default=0)
    largest_edges = max(component_edges, default=0)
    return {
        "scope": scope,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "connected_component_count": len(components),
        "largest_component_node_count": largest_nodes,
        "largest_component_node_ratio": (
            largest_nodes / graph.number_of_nodes()
            if graph.number_of_nodes()
            else 0
        ),
        "largest_component_edge_count": largest_edges,
        "largest_component_edge_ratio": (
            largest_edges / graph.number_of_edges()
            if graph.number_of_edges()
            else 0
        ),
        "degree_1_nodes": sum(value == 1 for value in degrees.values()),
        "degree_2_nodes": sum(value == 2 for value in degrees.values()),
        "degree_ge_3_nodes": sum(value >= 3 for value in degrees.values()),
    }


def tolerance_simulation(
    base_graph: nx.MultiGraph,
    reviewed: pd.DataFrame,
    tolerances: list[float],
) -> list[dict[str, Any]]:
    eligible = reviewed[
        reviewed["check_decision"] == "eligible_for_manual_approval"
    ]
    rows: list[dict[str, Any]] = []
    base = metrics(base_graph, "before_repair")
    for tolerance in tqdm(
        tolerances,
        desc="Simulating protected snap tolerances",
        unit="tolerance",
        dynamic_ncols=True,
    ):
        candidates = eligible[eligible["distance_m"] <= tolerance]
        simulated = base_graph.copy()
        for row in candidates.to_dict("records"):
            simulated.add_edge(
                _node(row["x_a"], row["y_a"]),
                _node(row["x_b"], row["y_b"]),
                simulated_snap=True,
            )
        current = metrics(simulated, f"simulated_{tolerance:g}m")
        rows.append(
            {
                "tolerance_m": tolerance,
                "eligible_candidate_count": len(candidates),
                "formal_repairs_performed": 0,
                "base_component_count": base["connected_component_count"],
                "simulated_component_count": current[
                    "connected_component_count"
                ],
                "component_reduction": (
                    base["connected_component_count"]
                    - current["connected_component_count"]
                ),
                "base_degree_1_nodes": base["degree_1_nodes"],
                "simulated_degree_1_nodes": current["degree_1_nodes"],
                "degree_1_reduction": (
                    base["degree_1_nodes"] - current["degree_1_nodes"]
                ),
                "base_largest_component_node_ratio": base[
                    "largest_component_node_ratio"
                ],
                "simulated_largest_component_node_ratio": current[
                    "largest_component_node_ratio"
                ],
            }
        )
    return rows


def run_check(
    config_path: Path,
    rules_path: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    config = load_yaml(config_path)
    rules = load_yaml(rules_path)
    root = Path(str(config["project_root"])).resolve()
    topology_dir = root / "routing/data/topology"
    report_path = root / "routing/reports/STEP24_TOPOLOGY_CHECK_REPORT.md"
    outputs = {
        "review": topology_dir / "topology_repair_candidates.csv",
        "shortlist": topology_dir / "topology_2m_candidate_shortlist.csv",
        "rejected": topology_dir / "topology_rejected_candidates.csv",
        "tolerance": topology_dir / "topology_tolerance_comparison.csv",
        "metrics": topology_dir / "topology_pre_repair_metrics.csv",
        "failed": topology_dir / "step24_failed_records.csv",
        "summary": topology_dir / "step24_check_summary.json",
    }
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise FileExistsError("Step24 check outputs exist; pass --overwrite")
    step23_summary = json.loads(
        (
            root
            / "routing/data/candidate_network/step23_formal_summary.json"
        ).read_text(encoding="utf-8")
    )
    blockers: list[str] = []
    if not step23_summary.get("ready_for_step24_check"):
        blockers.append("Step23 formal output is not ready for Step24 check")
    candidate_gdb = (
        root
        / "routing/data/candidate_network/step23_candidate_network.gdb"
    )
    audit_gdb = root / "routing/data/audit/road_audit.gdb"
    classification = pd.read_csv(
        root
        / "routing/data/candidate_network/road_thermal_classification.csv",
        dtype={"edge_id": str},
        low_memory=False,
    )
    graph, null_geometry = read_base_graph(candidate_gdb)
    if null_geometry:
        blockers.append(f"Thermal road layer has {null_geometry} null geometries")
    pairs = read_snap_pairs(audit_gdb)
    reviewed = classify_candidates(pairs, classification, rules)
    rejected = reviewed[reviewed["check_decision"] == "hard_reject"].copy()
    eligible = reviewed[
        reviewed["check_decision"] == "eligible_for_manual_approval"
    ].copy()
    approved_tolerance = rules.get("approved_tolerance_m")
    shortlist = (
        eligible[eligible["distance_m"] <= float(approved_tolerance)].copy()
        if approved_tolerance is not None
        else eligible.iloc[0:0].copy()
    )
    tolerance_rows = tolerance_simulation(
        graph,
        reviewed,
        [float(value) for value in rules["tolerance_candidates_m"]],
    )
    base_metrics = metrics(graph, "thermal_comfort_relevant_before_repair")
    base_metrics["formal_repairs_performed"] = 0
    topology_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(outputs["review"], reviewed.to_dict("records"))
    if shortlist.empty:
        atomic_text(outputs["shortlist"], ",".join(reviewed.columns) + "\n")
    else:
        atomic_csv(outputs["shortlist"], shortlist.to_dict("records"))
    if rejected.empty:
        atomic_text(outputs["rejected"], ",".join(reviewed.columns) + "\n")
    else:
        atomic_csv(outputs["rejected"], rejected.to_dict("records"))
    atomic_csv(outputs["tolerance"], tolerance_rows)
    atomic_csv(outputs["metrics"], [base_metrics])
    atomic_text(
        outputs["failed"],
        "category,object_id,error_type,error_message\n",
    )
    ended_at = now_text()
    reason_counts = (
        reviewed["decision_reasons"]
        .str.get_dummies(sep=";")
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )
    summary = {
        "phase": "G3",
        "step": 24,
        "mode": "check",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "ready_for_topology_rule_approval": len(blockers) == 0,
        "ready_for_formal_topology_run": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": 0,
        "warnings": [],
        "thermal_road_count": int(step23_summary["thermal_comfort_relevant_count"]),
        "base_graph_metrics": base_metrics,
        "snap_candidate_count": len(reviewed),
        "hard_reject_count": len(rejected),
        "eligible_for_manual_approval_count": len(eligible),
        "approved_tolerance_shortlist_count": len(shortlist),
        "approved_repair_count": 0,
        "repair_performed_count": 0,
        "decision_reason_counts": {
            str(key): int(value) for key, value in reason_counts.items()
        },
        "tolerance_simulation": tolerance_rows,
        "approved_tolerance_m": rules.get("approved_tolerance_m"),
        "rule_status": rules.get("status"),
        "rule_approved": bool(rules.get("approval", {}).get("approved")),
        "source_roads_modified": False,
        "step23_network_modified": False,
        "topology_repairs_performed": False,
        "route_search_performed": False,
        "output_paths": {key: str(value) for key, value in outputs.items()},
        "report_path": str(report_path),
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step24 Topology Check Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `READY_FOR_TOPOLOGY_RULE_APPROVAL = {str(summary['ready_for_topology_rule_approval']).upper()}`",
        "- `READY_FOR_FORMAL_TOPOLOGY_RUN = FALSE`",
        f"- Thermal-comfort candidate roads: {summary['thermal_road_count']:,}",
        f"- Endpoint snap candidates reviewed: {len(reviewed):,}",
        f"- Hard rejected: {len(rejected):,}",
        f"- Eligible only for manual approval: {len(eligible):,}",
        f"- Approved tolerance: {approved_tolerance if approved_tolerance is not None else 'NOT_SELECTED'} m",
        f"- Candidates within approved tolerance: {len(shortlist):,}",
        "- Repairs performed: 0",
        "",
        "## Tolerance simulation",
        "",
        "| tolerance m | eligible pairs | component reduction | degree-1 reduction |",
        "|---:|---:|---:|---:|",
    ]
    for row in tolerance_rows:
        lines.append(
            f"| {row['tolerance_m']:.1f} | {row['eligible_candidate_count']} | "
            f"{row['component_reduction']} | {row['degree_1_reduction']} |"
        )
    lines.extend(
        [
            "",
            "No tolerance was selected. No endpoint was moved, no connector was written, no line was split and no topology repair was performed.",
            "",
            "Before formal execution, review `topology_repair_candidates.csv`, select an approved tolerance and explicitly approve the individual repair candidates or a documented subset rule.",
        ]
    )
    atomic_text(report_path, "\n".join(lines) + "\n")
    logger.info(
        "Step24 check complete: ready=%s candidates=%s rejected=%s eligible=%s",
        summary["ready_for_topology_rule_approval"],
        len(reviewed),
        len(rejected),
        len(eligible),
    )
    return summary


def _copy_fields(source_definition: Any, target_layer: Any) -> list[str]:
    names: list[str] = []
    for index in range(source_definition.GetFieldCount()):
        field = source_definition.GetFieldDefn(index)
        target_layer.CreateField(field)
        names.append(field.GetName())
    return names


def _copy_feature_values(
    source_feature: Any,
    target_feature: Any,
    field_names: list[str],
) -> None:
    for field_name in field_names:
        target_feature.SetField(
            field_name, source_feature.GetField(field_name)
        )


def run_formal(
    config_path: Path,
    rules_path: Path,
    overwrite: bool,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Write approved <=2 m gap connectors without moving source roads."""
    from osgeo import ogr

    ogr.UseExceptions()
    started_at = now_text()
    started = time.perf_counter()
    config = load_yaml(config_path)
    rules = load_yaml(rules_path)
    root = Path(str(config["project_root"])).resolve()
    topology_dir = root / "routing/data/topology"
    input_gdb = (
        root
        / "routing/data/candidate_network/step23_candidate_network.gdb"
    )
    output_gdb = topology_dir / "step24_repaired_network.gdb"
    repairs_path = topology_dir / "topology_repairs.csv"
    metrics_path = topology_dir / "topology_repair_metrics.csv"
    failed_path = topology_dir / "step24_formal_failed_records.csv"
    summary_path = topology_dir / "step24_formal_summary.json"
    qc_path = topology_dir / "step24_formal_qc.json"
    report_path = root / "routing/reports/STEP24_FORMAL_RUN_REPORT.md"
    outputs = [
        output_gdb,
        repairs_path,
        metrics_path,
        failed_path,
        summary_path,
        qc_path,
        report_path,
    ]
    if resume and all(path.exists() for path in outputs):
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        logger.info("Resume: using validated Step24 formal outputs")
        return prior
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError(
            "Step24 formal outputs exist; use --resume or --overwrite"
        )
    if output_gdb.exists():
        ogr.GetDriverByName("OpenFileGDB").DeleteDataSource(str(output_gdb))
    for path in outputs[1:]:
        if path.exists():
            path.unlink()

    approved_tolerance = float(rules["approved_tolerance_m"])
    shortlist = pd.read_csv(
        topology_dir / "topology_2m_candidate_shortlist.csv",
        dtype={"edge_id_a": str, "edge_id_b": str},
        low_memory=False,
    )
    approved_candidate_count = len(shortlist)
    if (shortlist["distance_m"] > approved_tolerance).any():
        raise ValueError("Approved shortlist contains distances over tolerance")
    if (shortlist["check_decision"] != "eligible_for_manual_approval").any():
        raise ValueError("Approved shortlist contains a hard-rejected candidate")

    step23_summary = json.loads(
        (root / "routing/data/candidate_network/step23_formal_summary.json").read_text(
            encoding="utf-8"
        )
    )
    expected_thermal_count = int(step23_summary["thermal_comfort_relevant_count"])
    expected_motor_count = int(step23_summary["motor_vehicle_only_count"])
    tolerance_rows = pd.read_csv(
        topology_dir / "topology_tolerance_comparison.csv", low_memory=False
    )
    tolerance_row = tolerance_rows.loc[
        np.isclose(tolerance_rows["tolerance_m"].astype(float), approved_tolerance)
    ]
    if len(tolerance_row) != 1:
        raise ValueError("Approved tolerance is missing or duplicated in Step24 simulation")
    expected_component_reduction = int(tolerance_row.iloc[0]["component_reduction"])
    expected_degree_one_reduction = int(tolerance_row.iloc[0]["degree_1_reduction"])

    source_ds = ogr.Open(str(input_gdb), 0)
    if source_ds is None:
        raise FileNotFoundError(input_gdb)
    source_thermal = source_ds.GetLayerByName(
        "ThermalComfortRelevantRoads"
    )
    source_motor = source_ds.GetLayerByName("MotorVehicleOnlyRoads")
    if source_thermal is None or source_motor is None:
        raise KeyError("Step23 formal layers are incomplete")
    driver = ogr.GetDriverByName("OpenFileGDB")
    output_ds = driver.CreateDataSource(str(output_gdb))
    if output_ds is None:
        raise RuntimeError(f"Cannot create {output_gdb}")
    srs = source_thermal.GetSpatialRef()
    repaired = output_ds.CreateLayer(
        "ThermalComfortNetworkRepaired",
        srs=srs,
        geom_type=source_thermal.GetLayerDefn().GetGeomType(),
        options=["TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER"],
    )
    thermal_fields = _copy_fields(source_thermal.GetLayerDefn(), repaired)
    for name, field_type, width in (
        ("repair_id", ogr.OFTString, 32),
        ("repair_typ", ogr.OFTString, 32),
        ("source_a", ogr.OFTString, 64),
        ("source_b", ogr.OFTString, 64),
        ("snap_dist", ogr.OFTReal, 0),
    ):
        field = ogr.FieldDefn(name, field_type)
        if width:
            field.SetWidth(width)
        repaired.CreateField(field)

    motor_copy = output_ds.CreateLayer(
        "MotorVehicleOnlyRoads",
        srs=source_motor.GetSpatialRef(),
        geom_type=source_motor.GetLayerDefn().GetGeomType(),
        options=["TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER"],
    )
    motor_fields = _copy_fields(source_motor.GetLayerDefn(), motor_copy)
    connector_layer = output_ds.CreateLayer(
        "TopologyRepairConnectors",
        srs=srs,
        geom_type=ogr.wkbLineString,
        options=["TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER"],
    )
    for name, field_type, width in (
        ("repair_id", ogr.OFTString, 32),
        ("pair_id", ogr.OFTString, 32),
        ("edge_id_a", ogr.OFTString, 64),
        ("edge_id_b", ogr.OFTString, 64),
        ("distance_m", ogr.OFTReal, 0),
        ("rule_ver", ogr.OFTString, 48),
    ):
        field = ogr.FieldDefn(name, field_type)
        if width:
            field.SetWidth(width)
        connector_layer.CreateField(field)

    failed_records: list[dict[str, Any]] = []
    original_written = 0
    source_thermal.ResetReading()
    for feature in tqdm(
        source_thermal,
        total=source_thermal.GetFeatureCount(),
        desc="Copying thermal network before repairs",
        unit="road",
        dynamic_ncols=True,
    ):
        try:
            target = ogr.Feature(repaired.GetLayerDefn())
            _copy_feature_values(feature, target, thermal_fields)
            target.SetField("repair_typ", "original")
            target.SetGeometry(feature.GetGeometryRef().Clone())
            repaired.CreateFeature(target)
            original_written += 1
        except Exception as error:
            failed_records.append(
                {
                    "category": "copy_thermal_road",
                    "object_id": feature.GetField("edge_id"),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    motor_written = 0
    source_motor.ResetReading()
    for feature in tqdm(
        source_motor,
        total=source_motor.GetFeatureCount(),
        desc="Copying protected motor-only roads",
        unit="road",
        dynamic_ncols=True,
    ):
        try:
            target = ogr.Feature(motor_copy.GetLayerDefn())
            _copy_feature_values(feature, target, motor_fields)
            target.SetGeometry(feature.GetGeometryRef().Clone())
            motor_copy.CreateFeature(target)
            motor_written += 1
        except Exception as error:
            failed_records.append(
                {
                    "category": "copy_motor_road",
                    "object_id": feature.GetField("edge_id"),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    repair_records: list[dict[str, Any]] = []
    connectors_written = 0
    for index, row in enumerate(
        tqdm(
            shortlist.to_dict("records"),
            desc="Writing approved topology connectors",
            unit="repair",
            dynamic_ncols=True,
        ),
        start=1,
    ):
        repair_id = f"TR{index:05d}"
        try:
            geometry = ogr.Geometry(ogr.wkbLineString)
            geometry.AddPoint_2D(float(row["x_a"]), float(row["y_a"]))
            geometry.AddPoint_2D(float(row["x_b"]), float(row["y_b"]))
            synthetic_edge_id = f"SNAP_{row['pair_id']}"

            target = ogr.Feature(repaired.GetLayerDefn())
            for field_name, value in (
                ("osm_id", synthetic_edge_id),
                ("fclass", "topology_connector"),
                ("name", "approved_topology_gap_connector"),
                ("edge_id", synthetic_edge_id),
                ("therm_class", "thermal_comfort_relevant"),
                ("class_basis", "approved_2m_topology_repair"),
                ("grade_sep", 0),
                ("rule_ver", rules["rule_version"]),
                ("repair_id", repair_id),
                ("repair_typ", "gap_connector"),
                ("source_a", str(row["edge_id_a"])),
                ("source_b", str(row["edge_id_b"])),
                ("snap_dist", float(row["distance_m"])),
            ):
                if target.GetFieldIndex(field_name) >= 0:
                    target.SetField(field_name, value)
            target.SetGeometry(geometry)
            repaired.CreateFeature(target)

            audit_feature = ogr.Feature(connector_layer.GetLayerDefn())
            for field_name, value in (
                ("repair_id", repair_id),
                ("pair_id", str(row["pair_id"])),
                ("edge_id_a", str(row["edge_id_a"])),
                ("edge_id_b", str(row["edge_id_b"])),
                ("distance_m", float(row["distance_m"])),
                ("rule_ver", rules["rule_version"]),
            ):
                audit_feature.SetField(field_name, value)
            audit_feature.SetGeometry(geometry.Clone())
            connector_layer.CreateFeature(audit_feature)
            connectors_written += 1
            repair_records.append(
                {
                    "repair_id": repair_id,
                    "pair_id": row["pair_id"],
                    "edge_id_a": row["edge_id_a"],
                    "edge_id_b": row["edge_id_b"],
                    "x_a": row["x_a"],
                    "y_a": row["y_a"],
                    "x_b": row["x_b"],
                    "y_b": row["y_b"],
                    "distance_m": row["distance_m"],
                    "repair_type": "explicit_gap_connector",
                    "approved_tolerance_m": approved_tolerance,
                    "hard_conflict_filtered": True,
                    "repair_performed": True,
                }
            )
        except Exception as error:
            failed_records.append(
                {
                    "category": "write_topology_connector",
                    "object_id": row["pair_id"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    output_ds = None
    source_ds = None

    before_graph, before_null = read_base_graph(input_gdb)
    after_graph, after_null = read_base_graph(
        output_gdb, "ThermalComfortNetworkRepaired"
    )
    before_metrics = metrics(before_graph, "before_repair")
    after_metrics = metrics(after_graph, "after_repair")
    atomic_csv(repairs_path, repair_records)
    atomic_csv(metrics_path, [before_metrics, after_metrics])
    if failed_records:
        atomic_csv(failed_path, failed_records)
    else:
        atomic_text(
            failed_path,
            "category,object_id,error_type,error_message\n",
        )

    blockers: list[str] = []
    if failed_records:
        blockers.append(f"{len(failed_records)} formal records failed")
    if original_written != expected_thermal_count:
        blockers.append(f"Thermal source copy count is {original_written}")
    if motor_written != expected_motor_count:
        blockers.append(f"Motor-only copy count is {motor_written}")
    if connectors_written != approved_candidate_count:
        blockers.append(f"Connector count is {connectors_written}")
    expected_after_edges = original_written + connectors_written
    if after_graph.number_of_edges() != expected_after_edges:
        blockers.append(
            f"Repaired graph edges {after_graph.number_of_edges()} != "
            f"{expected_after_edges}"
        )
    if before_null or after_null:
        blockers.append(
            f"Null geometry before/after: {before_null}/{after_null}"
        )
    component_reduction = (
        before_metrics["connected_component_count"]
        - after_metrics["connected_component_count"]
    )
    degree_one_reduction = (
        before_metrics["degree_1_nodes"] - after_metrics["degree_1_nodes"]
    )
    if (
        component_reduction != expected_component_reduction
        or degree_one_reduction != expected_degree_one_reduction
    ):
        blockers.append(
            "Formal topology metrics differ from approved simulation: "
            f"components={component_reduction}, degree1={degree_one_reduction}"
        )
    ended_at = now_text()
    summary = {
        "phase": "G3",
        "step": 24,
        "mode": "run",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "rule_version": rules["rule_version"],
        "rule_status": rules["status"],
        "approved_tolerance_m": approved_tolerance,
        "approved_candidate_count": approved_candidate_count,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "success": not blockers,
        "ready_for_step25_check": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": 0,
        "warnings": [],
        "thermal_source_road_count": original_written,
        "motor_only_preserved_count": motor_written,
        "repair_connector_count": connectors_written,
        "repaired_network_edge_count": after_graph.number_of_edges(),
        "failed_record_count": len(failed_records),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "component_reduction": component_reduction,
        "degree_1_reduction": degree_one_reduction,
        "maximum_repair_distance_m": (
            float(shortlist["distance_m"].max()) if approved_candidate_count else 0.0
        ),
        "source_roads_modified": False,
        "step23_network_modified": False,
        "road_endpoints_moved": False,
        "topology_repairs_performed": True,
        "formal_point_edge_match_performed": False,
        "route_search_performed": False,
        "outputs": {
            "gdb": str(output_gdb),
            "repaired_layer": (
                str(output_gdb) + "/ThermalComfortNetworkRepaired"
            ),
            "connector_layer": (
                str(output_gdb) + "/TopologyRepairConnectors"
            ),
            "motor_only_layer": (
                str(output_gdb) + "/MotorVehicleOnlyRoads"
            ),
            "repairs_csv": str(repairs_path),
            "metrics_csv": str(metrics_path),
            "failed_records_csv": str(failed_path),
            "summary_json": str(summary_path),
            "qc_json": str(qc_path),
            "report": str(report_path),
        },
    }
    qc = {
        "success": not blockers,
        "input_thermal_count": expected_thermal_count,
        "copied_thermal_count": original_written,
        "preserved_motor_only_count": motor_written,
        "approved_connector_count": approved_candidate_count,
        "written_connector_count": connectors_written,
        "repaired_network_edge_count": after_graph.number_of_edges(),
        "maximum_repair_distance_m": summary["maximum_repair_distance_m"],
        "approved_tolerance_m": approved_tolerance,
        "component_reduction_matches_simulation": (
            component_reduction == expected_component_reduction
        ),
        "degree_1_reduction_matches_simulation": (
            degree_one_reduction == expected_degree_one_reduction
        ),
        "null_geometry_before": before_null,
        "null_geometry_after": after_null,
        "failed_record_count": len(failed_records),
        "source_roads_modified": False,
        "step23_network_modified": False,
    }
    atomic_json(summary_path, summary)
    atomic_json(qc_path, qc)
    report_lines = [
        "# Step24 Formal Run Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- Success: `{str(summary['success']).upper()}`",
        f"- Ready for Step25 check: `{str(summary['ready_for_step25_check']).upper()}`",
        f"- Approved tolerance: {approved_tolerance:.1f} m",
        f"- Approved/written repair connectors: {approved_candidate_count} / {connectors_written}",
        f"- Maximum connector distance: {summary['maximum_repair_distance_m']:.3f} m",
        f"- Repaired thermal network edges: {after_graph.number_of_edges():,}",
        f"- Connected-component reduction: {component_reduction}",
        f"- Degree-1 node reduction: {degree_one_reduction}",
        f"- Failed records: {len(failed_records)}",
        "",
        "Repairs were written as explicit <=2 m gap connectors. Original road endpoints and the Step23 network were not moved or overwritten. Motor-only roads were copied unchanged for audit.",
        "",
        "No point-edge matching, thermal-cost assignment or route search was performed.",
    ]
    atomic_text(report_path, "\n".join(report_lines) + "\n")
    logger.info(
        "Step24 formal complete: success=%s connectors=%s components_reduced=%s",
        summary["success"],
        connectors_written,
        component_reduction,
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_yaml(args.network_config.resolve())
    root = Path(str(config["project_root"])).resolve()
    logger = setup_logger(root / "routing/logs/step24.log", args.overwrite)
    failure_path = root / "routing/data/topology/step24_failure.json"
    try:
        if args.mode == "run":
            rules = load_yaml(args.rules.resolve())
            if (
                not args.approved_by_user
                or not rules.get("approval", {}).get("approved")
                or rules.get("approved_tolerance_m") is None
            ):
                raise PermissionError(
                    "Formal Step24 run requires explicit user approval, an "
                    "approved rule file and approved_tolerance_m."
                )
            summary = run_formal(
                args.network_config.resolve(),
                args.rules.resolve(),
                args.overwrite,
                args.resume,
                logger,
            )
            if failure_path.exists():
                failure_path.unlink()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if summary["success"] else 2
        summary = run_check(
            args.network_config.resolve(),
            args.rules.resolve(),
            args.overwrite,
            logger,
        )
        if failure_path.exists():
            failure_path.unlink()
        print(
            json.dumps(
                {
                    "READY_FOR_TOPOLOGY_RULE_APPROVAL": summary[
                        "ready_for_topology_rule_approval"
                    ],
                    "READY_FOR_FORMAL_TOPOLOGY_RUN": False,
                    "snap_candidate_count": summary["snap_candidate_count"],
                    "hard_reject_count": summary["hard_reject_count"],
                    "eligible_for_manual_approval_count": summary[
                        "eligible_for_manual_approval_count"
                    ],
                    "repair_performed_count": 0,
                    "tolerance_simulation": summary["tolerance_simulation"],
                    "blocker_count": summary["blocker_count"],
                    "report_path": summary["report_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if summary["ready_for_topology_rule_approval"] else 2
    except Exception as error:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
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
        logger.exception("Step24 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
