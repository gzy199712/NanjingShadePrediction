"""Step23: review and (only after approval) freeze transport-mode networks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.src.mode_classification import (  # noqa: E402
    classify_mode_preview,
    load_mode_rules,
    read_road_attributes,
)
from routing.src.reporting import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_text,
)

SCRIPT_VERSION = "2.0.0"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase G2 mode-rule review and candidate-network builder."
    )
    parser.add_argument(
        "--network-config",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/phase_g_network.yaml",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/mode_rules_draft.yaml",
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


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_g_step23")
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


def build_impact(
    preview: pd.DataFrame,
    geometry: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    merged = preview.merge(
        geometry[["edge_id", "length_m"]],
        on="edge_id",
        how="left",
        validate="one_to_one",
    ).merge(
        coverage[
            [
                "edge_id",
                "inside_center",
                "covered_within_75m",
            ]
        ],
        on="edge_id",
        how="left",
        validate="one_to_one",
    )
    fields = [
        "fclass",
        "fclass_cn",
        "classification_basis",
        "mode_status_preview",
        "grade_separated",
    ]
    rows: list[dict[str, Any]] = []
    grouped = merged.groupby(fields, dropna=False, sort=True)
    for key, members in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Summarizing mode-rule impact",
        unit="group",
        dynamic_ncols=True,
    ):
        values = dict(zip(fields, key, strict=True))
        values.update(
            {
                "feature_count": len(members),
                "total_length_km": members["length_m"].sum() / 1000,
                "center_feature_count": int(
                    members["inside_center"].astype(bool).sum()
                ),
                "thermal_covered_feature_count_75m": int(
                    members["covered_within_75m"].astype(bool).sum()
                ),
                "formal_network_written": False,
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def decision_register(rules: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [
        (
            "always_by_fclass",
            rules["motor_vehicle_only_rules"]["always_by_fclass"],
            "motor_vehicle_only",
        ),
        (
            "when_grade_separated",
            rules["motor_vehicle_only_rules"]["when_grade_separated"],
            "motor_vehicle_only_if_grade_separated",
        ),
        (
            "preserve_active_travel",
            rules["thermal_comfort_relevant_rules"][
                "explicitly_preserve_even_when_grade_separated"
            ],
            "thermal_comfort_relevant",
        ),
    ]
    for group_name, classes, draft_class in tqdm(
        groups,
        desc="Building two-class decision register",
        unit="rule group",
        dynamic_ncols=True,
    ):
        for fclass in classes:
            rows.append(
                {
                    "rule_group": group_name,
                    "fclass": fclass,
                    "draft_output_class": draft_class,
                    "approved_output_class": "",
                    "reviewer_note": "",
                    "approved": False,
                }
            )
    return rows


def write_reports(
    reports_dir: Path,
    summary: dict[str, Any],
    impact: pd.DataFrame,
) -> list[str]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = reports_dir / "STEP23_MODE_RULE_CHECK_REPORT.md"
    checklist = reports_dir / "MODE_RULE_REVIEW_CHECKLIST.md"
    lines = [
        "# Step23 Mode Rule Check Report",
        "",
        f"Generated: {summary['ended_at']}",
        "",
        f"- `READY_FOR_MODE_RULE_APPROVAL = {str(summary['ready_for_mode_rule_approval']).upper()}`",
        f"- Roads classified exactly once: {summary['road_count']:,}",
        f"- Output classes: {summary['output_class_count']}",
        f"- Thermal-comfort-relevant roads: {summary['thermal_comfort_relevant_count']:,}",
        f"- Motor-vehicle-only roads: {summary['motor_vehicle_only_count']:,}",
        "",
        "This is a classification preview only. No formal candidate network was written and no road was deleted.",
        "",
        "## Preview totals",
        "",
        "| status | roads | length km |",
        "|---|---:|---:|",
    ]
    totals = (
        impact.groupby("mode_status_preview", as_index=False)
        .agg(feature_count=("feature_count", "sum"), total_length_km=("total_length_km", "sum"))
        .sort_values("mode_status_preview")
    )
    for row in tqdm(
        totals.to_dict("records"),
        desc="Writing mode preview report",
        unit="status",
        dynamic_ncols=True,
    ):
        lines.append(
            f"| {row['mode_status_preview']} | {int(row['feature_count'])} | "
            f"{row['total_length_km']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Required decision",
            "",
            "Confirm the two-class rule in `routing/configs/mode_rules_draft.yaml`. Phase G2 formal run remains blocked until explicit approval.",
        ]
    )
    atomic_text(report, "\n".join(lines) + "\n")
    atomic_text(
        checklist,
        "\n".join(
            [
                "# Mode Rule Review Checklist",
                "",
                "- [ ] Confirm exactly two output classes.",
                "- [ ] Confirm motorway/trunk and links as motor-vehicle-only.",
                "- [ ] Confirm grade-separated primary/secondary/tertiary roads as motor-vehicle-only.",
                "- [ ] Confirm active-travel structures remain thermal-comfort-relevant.",
                "- [ ] Confirm all remaining roads default to thermal-comfort-relevant.",
                "- [ ] Record reviewer, approval time and notes in the rule YAML.",
                "- [ ] Explicitly approve Phase G2 formal run.",
                "",
                "Approval of mode rules does not approve topology repair, snapping, thermal matching or route search.",
            ]
        )
        + "\n",
    )
    return [str(report), str(checklist)]


def run_check(
    *,
    network_config_path: Path,
    rules_path: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    config = load_yaml(network_config_path)
    rules = load_mode_rules(rules_path)
    root = Path(str(config["project_root"])).resolve()
    output_dir = root / "routing/data/mode_rules"
    reports_dir = root / "routing/reports"
    outputs = {
        "preview": output_dir / "road_mode_preview.csv",
        "impact": output_dir / "mode_rule_impact_preview.csv",
        "conflicts": output_dir / "mode_rule_conflicts.csv",
        "decisions": output_dir / "mode_rule_decision_register.csv",
        "failed": output_dir / "step23_failed_records.csv",
        "summary": output_dir / "step23_mode_rule_check_summary.json",
    }
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise FileExistsError("Step23 check outputs exist; pass --overwrite")
    step22_path = root / "routing/data/audit/step22_network_preflight_summary.json"
    step22 = json.loads(step22_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    warnings: list[str] = []
    if not step22.get("ready_for_mode_rule_review"):
        blockers.append("Step22 did not approve mode-rule review")
    road_path = resolve(root, config["inputs"]["osm_road_shp"])
    roads = read_road_attributes(road_path)
    roads["edge_id"] = roads["edge_id"].astype(str)
    observed_classes = set(roads["fclass"].fillna("unknown").astype(str).str.lower())
    configured_classes = set(
        rules["motor_vehicle_only_rules"]["always_by_fclass"]
    ) | set(rules["motor_vehicle_only_rules"]["when_grade_separated"]) | set(
        rules["thermal_comfort_relevant_rules"][
            "explicitly_preserve_even_when_grade_separated"
        ]
    )
    extra_rules = sorted(configured_classes - observed_classes)
    missing_rules: list[str] = []
    if extra_rules:
        warnings.append("Configured classes without observed roads: " + ", ".join(extra_rules))
    preview, conflicts = classify_mode_preview(roads, rules)
    if len(preview) != len(roads) or not preview["edge_id"].is_unique:
        blockers.append("Road classification preview is incomplete or duplicated")
    geometry = pd.read_csv(
        root / "routing/data/audit/road_geometry_qc.csv",
        dtype={"edge_id": str},
        low_memory=False,
    )
    coverage = pd.read_csv(
        root / "routing/data/audit/road_thermal_coverage_qc.csv",
        dtype={"edge_id": str},
        low_memory=False,
    )
    impact = build_impact(preview, geometry, coverage)
    statuses = set(preview["mode_status_preview"])
    required_statuses = {"thermal_comfort_relevant", "motor_vehicle_only"}
    missing_statuses = sorted(required_statuses - statuses)
    if missing_statuses:
        warnings.append(
            "Draft produces no roads for statuses: " + ", ".join(missing_statuses)
        )
    if rules.get("approval", {}).get("approved"):
        warnings.append(
            "Rule file is marked approved, but this check does not execute it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(outputs["preview"], preview.to_dict("records"))
    atomic_csv(outputs["impact"], impact.to_dict("records"))
    if conflicts.empty:
        atomic_text(
            outputs["conflicts"],
            "edge_id,conflict_type,fclass,bridge,tunnel,detail\n",
        )
    else:
        atomic_csv(outputs["conflicts"], conflicts.to_dict("records"))
    atomic_csv(outputs["decisions"], decision_register(rules))
    atomic_text(
        outputs["failed"],
        "category,object_id,error_type,error_message\n",
    )
    ended_at = now_text()
    summary = {
        "phase": "G2",
        "step": 23,
        "mode": "check",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "ready_for_mode_rule_approval": len(blockers) == 0,
        "ready_for_formal_mode_network_run": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "road_count": len(preview),
        "observed_class_count": len(observed_classes),
        "rule_class_count": len(configured_classes),
        "output_class_count": 2,
        "missing_rule_classes": missing_rules,
        "extra_rule_classes": extra_rules,
        "mode_status_counts": {
            str(key): int(value)
            for key, value in preview["mode_status_preview"].value_counts().items()
        },
        "thermal_comfort_relevant_count": int(
            preview["thermal_comfort_relevant"].sum()
        ),
        "motor_vehicle_only_count": int(preview["motor_vehicle_only"].sum()),
        "walk_candidate_count": int(preview["thermal_comfort_relevant"].sum()),
        "bike_candidate_count": int(preview["thermal_comfort_relevant"].sum()),
        "shared_candidate_count": int(preview["thermal_comfort_relevant"].sum()),
        "motor_only_candidate_count": int(
            preview["motor_vehicle_only"].sum()
        ),
        "structure_review_count": int(preview["structure_review"].sum()),
        "manual_review_count": int(preview["manual_review"].sum()),
        "conflict_count": len(conflicts),
        "rule_status": rules.get("status"),
        "rule_approved": bool(rules.get("approval", {}).get("approved")),
        "source_roads_modified": False,
        "formal_networks_written": False,
        "topology_repairs_performed": False,
        "route_search_performed": False,
        "output_paths": {key: str(value) for key, value in outputs.items()},
    }
    report_paths = write_reports(reports_dir, summary, impact)
    summary["report_paths"] = report_paths
    atomic_json(outputs["summary"], summary)
    logger.info(
        "Step23 check complete: ready=%s roads=%s manual_review=%s blockers=%s",
        summary["ready_for_mode_rule_approval"],
        summary["road_count"],
        summary["manual_review_count"],
        summary["blocker_count"],
    )
    return summary


def run_formal(
    *,
    network_config_path: Path,
    rules_path: Path,
    overwrite: bool,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Write the approved two-class candidate network without topology edits."""
    from osgeo import ogr, osr

    ogr.UseExceptions()
    started_at = now_text()
    started = time.perf_counter()
    config = load_yaml(network_config_path)
    rules = load_mode_rules(rules_path)
    root = Path(str(config["project_root"])).resolve()
    preview_path = root / "routing/data/mode_rules/road_mode_preview.csv"
    check_summary_path = (
        root / "routing/data/mode_rules/step23_mode_rule_check_summary.json"
    )
    if not preview_path.exists() or not check_summary_path.exists():
        raise FileNotFoundError("Approved Step23 check outputs are missing")
    check_summary = json.loads(check_summary_path.read_text(encoding="utf-8"))
    if not check_summary.get("ready_for_mode_rule_approval"):
        raise RuntimeError("Step23 check is not ready for formal rule approval")
    if check_summary.get("output_class_count") != 2:
        raise RuntimeError("Step23 check did not produce exactly two classes")

    output_dir = root / "routing/data/candidate_network"
    gdb_path = output_dir / "step23_candidate_network.gdb"
    classification_path = output_dir / "road_thermal_classification.csv"
    failed_path = output_dir / "step23_failed_records.csv"
    summary_path = output_dir / "step23_formal_summary.json"
    report_path = root / "routing/reports/STEP23_FORMAL_RUN_REPORT.md"
    expected_outputs = [
        gdb_path,
        classification_path,
        failed_path,
        summary_path,
        report_path,
    ]
    if resume and all(path.exists() for path in expected_outputs):
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        logger.info("Resume: validated existing Step23 formal summary")
        return prior
    if any(path.exists() for path in expected_outputs) and not overwrite:
        raise FileExistsError(
            "Step23 formal outputs already exist; use --resume or --overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if gdb_path.exists():
        ogr.GetDriverByName("OpenFileGDB").DeleteDataSource(str(gdb_path))
    for path in (classification_path, failed_path, summary_path, report_path):
        if path.exists():
            path.unlink()

    preview = pd.read_csv(preview_path, dtype={"edge_id": str}, low_memory=False)
    required_columns = {
        "edge_id",
        "thermal_comfort_class_preview",
        "classification_basis",
        "grade_separated",
    }
    missing_columns = sorted(required_columns - set(preview.columns))
    if missing_columns:
        raise ValueError(
            "Preview lacks formal classification columns: "
            + ", ".join(missing_columns)
        )
    if len(preview) != int(check_summary["road_count"]):
        raise ValueError("Preview count differs from approved check summary")
    if not preview["edge_id"].is_unique:
        raise ValueError("Preview edge_id is not unique")
    allowed_classes = {"thermal_comfort_relevant", "motor_vehicle_only"}
    observed_classes = set(preview["thermal_comfort_class_preview"])
    if observed_classes != allowed_classes:
        raise ValueError(
            f"Formal output requires exactly {sorted(allowed_classes)}; "
            f"observed {sorted(observed_classes)}"
        )
    rule_lookup = preview.set_index("edge_id")[
        [
            "thermal_comfort_class_preview",
            "classification_basis",
            "grade_separated",
        ]
    ].to_dict("index")

    source_path = resolve(root, config["inputs"]["osm_road_shp"])
    source_ds = ogr.Open(str(source_path), 0)
    if source_ds is None:
        raise FileNotFoundError(source_path)
    source_layer = source_ds.GetLayer(0)
    source_defn = source_layer.GetLayerDefn()
    source_fields = [
        source_defn.GetFieldDefn(index)
        for index in range(source_defn.GetFieldCount())
    ]
    source_field_names = [field.GetName() for field in source_fields]
    edge_source_field = (
        "osm_id"
        if "osm_id" in source_field_names
        else source_layer.GetFIDColumn()
    )

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(int(config["spatial"]["target_crs"].split(":")[1]))
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    driver = ogr.GetDriverByName("OpenFileGDB")
    output_ds = driver.CreateDataSource(str(gdb_path))
    if output_ds is None:
        raise RuntimeError(f"Cannot create {gdb_path}")
    layer_specs = {
        "thermal_comfort_relevant": "ThermalComfortRelevantRoads",
        "motor_vehicle_only": "MotorVehicleOnlyRoads",
    }
    output_layers: dict[str, Any] = {}
    for class_name, layer_name in tqdm(
        layer_specs.items(),
        desc="Creating formal Step23 layers",
        unit="layer",
        dynamic_ncols=True,
    ):
        layer = output_ds.CreateLayer(
            layer_name,
            srs=target_srs,
            geom_type=source_defn.GetGeomType(),
            options=["TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER"],
        )
        for source_field in source_fields:
            layer.CreateField(source_field)
        for name, field_type, width in (
            ("edge_id", ogr.OFTString, 64),
            ("therm_class", ogr.OFTString, 40),
            ("class_basis", ogr.OFTString, 64),
            ("grade_sep", ogr.OFTInteger, 0),
            ("rule_ver", ogr.OFTString, 40),
        ):
            field = ogr.FieldDefn(name, field_type)
            if width:
                field.SetWidth(width)
            layer.CreateField(field)
        output_layers[class_name] = layer

    failed_records: list[dict[str, Any]] = []
    written_counts = {key: 0 for key in layer_specs}
    seen_edge_ids: set[str] = set()
    source_layer.ResetReading()
    for source_feature in tqdm(
        source_layer,
        total=source_layer.GetFeatureCount(),
        desc="Writing approved two-class networks",
        unit="road",
        dynamic_ncols=True,
    ):
        edge_id = ""
        try:
            raw_edge_id = (
                source_feature.GetField(edge_source_field)
                if edge_source_field
                else source_feature.GetFID()
            )
            edge_id = str(raw_edge_id)
            decision = rule_lookup[edge_id]
            class_name = str(decision["thermal_comfort_class_preview"])
            target_layer = output_layers[class_name]
            target_feature = ogr.Feature(target_layer.GetLayerDefn())
            for field_name in source_field_names:
                target_feature.SetField(
                    field_name, source_feature.GetField(field_name)
                )
            target_feature.SetField("edge_id", edge_id)
            target_feature.SetField("therm_class", class_name)
            target_feature.SetField(
                "class_basis", str(decision["classification_basis"])
            )
            target_feature.SetField(
                "grade_sep",
                int(str(decision["grade_separated"]).lower() == "true"),
            )
            target_feature.SetField("rule_ver", str(rules["rule_version"]))
            geometry = source_feature.GetGeometryRef()
            if geometry is None:
                raise ValueError("Null geometry")
            geometry_copy = geometry.Clone()
            geometry_copy.Transform(transform)
            target_feature.SetGeometry(geometry_copy)
            target_layer.CreateFeature(target_feature)
            written_counts[class_name] += 1
            seen_edge_ids.add(edge_id)
        except Exception as error:
            failed_records.append(
                {
                    "category": "road_write",
                    "object_id": edge_id or source_feature.GetFID(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    output_ds = None
    source_ds = None

    missing_edge_ids = sorted(set(rule_lookup) - seen_edge_ids)
    for edge_id in missing_edge_ids:
        failed_records.append(
            {
                "category": "preview_edge_missing_from_source",
                "object_id": edge_id,
                "error_type": "MissingSourceFeature",
                "error_message": "Approved preview edge was not found in source roads.",
            }
        )
    formal_classification = preview.copy()
    formal_classification["formal_rule_applied"] = True
    formal_classification["rule_version"] = rules["rule_version"]
    atomic_csv(classification_path, formal_classification.to_dict("records"))
    if failed_records:
        atomic_csv(failed_path, failed_records)
    else:
        atomic_text(
            failed_path,
            "category,object_id,error_type,error_message\n",
        )

    total_written = sum(written_counts.values())
    blockers: list[str] = []
    if failed_records:
        blockers.append(f"{len(failed_records)} roads failed formal output")
    if total_written != len(preview):
        blockers.append(
            f"Output feature total {total_written} != preview total {len(preview)}"
        )
    for class_name, expected in preview[
        "thermal_comfort_class_preview"
    ].value_counts().items():
        actual = written_counts.get(str(class_name), 0)
        if actual != int(expected):
            blockers.append(
                f"{class_name}: output {actual} != preview {int(expected)}"
            )
    ended_at = now_text()
    summary = {
        "phase": "G2",
        "step": 23,
        "mode": "run",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "rule_version": rules["rule_version"],
        "rule_status": rules["status"],
        "approved": bool(rules["approval"]["approved"]),
        "approved_by": rules["approval"]["approved_by"],
        "approved_at": rules["approval"]["approved_at"],
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "success": not blockers,
        "ready_for_step24_check": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": 0,
        "warnings": [],
        "source_feature_count": len(preview),
        "output_feature_count": total_written,
        "output_class_count": 2,
        "thermal_comfort_relevant_count": written_counts[
            "thermal_comfort_relevant"
        ],
        "motor_vehicle_only_count": written_counts["motor_vehicle_only"],
        "failed_record_count": len(failed_records),
        "target_crs": config["spatial"]["target_crs"],
        "source_roads_modified": False,
        "topology_repairs_performed": False,
        "formal_point_edge_match_performed": False,
        "route_search_performed": False,
        "outputs": {
            "gdb": str(gdb_path),
            "thermal_comfort_relevant_layer": (
                str(gdb_path) + "/ThermalComfortRelevantRoads"
            ),
            "motor_vehicle_only_layer": (
                str(gdb_path) + "/MotorVehicleOnlyRoads"
            ),
            "classification_csv": str(classification_path),
            "failed_records_csv": str(failed_path),
            "summary_json": str(summary_path),
            "report": str(report_path),
        },
    }
    atomic_json(summary_path, summary)
    report_lines = [
        "# Step23 Formal Run Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- Success: `{str(summary['success']).upper()}`",
        f"- Ready for Step24 check: `{str(summary['ready_for_step24_check']).upper()}`",
        f"- Rule version: `{rules['rule_version']}`",
        f"- Output CRS: `{summary['target_crs']}`",
        f"- Source/output roads: {len(preview):,} / {total_written:,}",
        f"- Thermal-comfort-relevant roads: {written_counts['thermal_comfort_relevant']:,}",
        f"- Motor-vehicle-only roads: {written_counts['motor_vehicle_only']:,}",
        f"- Failed records: {len(failed_records):,}",
        "",
        "The source OSM roads were not modified. No topology repair, point-edge matching, thermal-cost mapping or route search was performed.",
    ]
    atomic_text(report_path, "\n".join(report_lines) + "\n")
    logger.info(
        "Step23 formal run complete: success=%s output=%s failed=%s",
        summary["success"],
        total_written,
        len(failed_records),
    )
    return summary


def main() -> int:
    args = parse_args()
    network_config = load_yaml(args.network_config.resolve())
    root = Path(str(network_config["project_root"])).resolve()
    logger = setup_logger(root / "routing/logs/step23.log", args.overwrite)
    failure_path = root / "routing/data/mode_rules/step23_failure.json"
    try:
        if args.mode == "run":
            rules = load_mode_rules(args.rules.resolve())
            if not args.approved_by_user or not rules.get("approval", {}).get(
                "approved"
            ):
                raise PermissionError(
                    "Formal Phase G2 run requires explicit user approval and "
                    "approval.approved=true in the reviewed rule file."
                )
            summary = run_formal(
                network_config_path=args.network_config.resolve(),
                rules_path=args.rules.resolve(),
                overwrite=args.overwrite,
                resume=args.resume,
                logger=logger,
            )
            if failure_path.exists():
                failure_path.unlink()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if summary["success"] else 2
        summary = run_check(
            network_config_path=args.network_config.resolve(),
            rules_path=args.rules.resolve(),
            overwrite=args.overwrite,
            logger=logger,
        )
        if failure_path.exists():
            failure_path.unlink()
        print(
            json.dumps(
                {
                    "READY_FOR_MODE_RULE_APPROVAL": summary[
                        "ready_for_mode_rule_approval"
                    ],
                    "READY_FOR_FORMAL_MODE_NETWORK_RUN": summary[
                        "ready_for_formal_mode_network_run"
                    ],
                    "road_count": summary["road_count"],
                    "rule_class_count": summary["rule_class_count"],
                    "mode_status_counts": summary["mode_status_counts"],
                    "walk_candidate_count": summary["walk_candidate_count"],
                    "bike_candidate_count": summary["bike_candidate_count"],
                    "shared_candidate_count": summary["shared_candidate_count"],
                    "motor_only_candidate_count": summary[
                        "motor_only_candidate_count"
                    ],
                    "manual_review_count": summary["manual_review_count"],
                    "conflict_count": summary["conflict_count"],
                    "blocker_count": summary["blocker_count"],
                    "warning_count": summary["warning_count"],
                    "report_paths": summary["report_paths"],
                    "formal_networks_written": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if summary["ready_for_mode_rule_approval"] else 2
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
        logger.exception("Step23 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
