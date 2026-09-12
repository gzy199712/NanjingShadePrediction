"""Step24c: apply the single user-approved od_025 aligned topology repair.

The existing Step24 output is backed up before an atomic directory swap. The
original road features are not moved or edited; one explicit connector is added.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = ROOT / "routing/configs/topology_gap_repair_approved.yaml"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve(value: str) -> Path:
    return (ROOT / value).resolve()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step24c_gap_repair")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in (
        logging.FileHandler(path, mode="w" if overwrite else "a", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def feature_count_and_match(gdb: Path, layer_name: str, edge_id: str) -> tuple[int, int, float | None]:
    from osgeo import ogr

    dataset = ogr.Open(str(gdb), 0)
    if dataset is None:
        raise FileNotFoundError(gdb)
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise KeyError(layer_name)
    count = int(layer.GetFeatureCount())
    matches = 0
    length = None
    layer.SetAttributeFilter(f"edge_id = '{edge_id}'")
    for feature in layer:
        matches += 1
        geometry = feature.GetGeometryRef()
        length = float(geometry.Length()) if geometry is not None else None
    dataset = None
    return count, matches, length


def append_connector(gdb: Path, config: dict[str, Any]) -> None:
    from osgeo import ogr

    ogr.UseExceptions()
    repair = config["repair"]
    dataset = ogr.Open(str(gdb), 1)
    if dataset is None:
        raise RuntimeError(f"Cannot open staged GDB for update: {gdb}")
    repaired = dataset.GetLayerByName(config["input"]["repaired_layer"])
    audit = dataset.GetLayerByName(config["input"]["connector_layer"])
    if repaired is None or audit is None:
        raise KeyError("Required Step24 layers are missing")

    geometry = ogr.Geometry(ogr.wkbLineString)
    geometry.AddPoint_2D(float(repair["source_x"]), float(repair["source_y"]))
    geometry.AddPoint_2D(float(repair["target_x"]), float(repair["target_y"]))

    feature = ogr.Feature(repaired.GetLayerDefn())
    values = {
        "osm_id": repair["synthetic_edge_id"],
        "fclass": "topology_connector",
        "name": "approved_aligned_missing_road_connector",
        "edge_id": repair["synthetic_edge_id"],
        "therm_class": "thermal_comfort_relevant",
        "class_basis": "approved_od025_aligned_gap_repair",
        "grade_sep": 0,
        "rule_ver": config["rule_version"],
        "repair_id": repair["repair_id"],
        "repair_typ": "aligned_gap_connector",
        "source_a": repair["source_edge_id"],
        "source_b": repair["target_edge_id"],
        "snap_dist": float(repair["distance_m"]),
    }
    for field, value in values.items():
        if feature.GetFieldIndex(field) >= 0:
            feature.SetField(field, value)
    feature.SetGeometry(geometry)
    repaired.CreateFeature(feature)

    audit_feature = ogr.Feature(audit.GetLayerDefn())
    audit_values = {
        "repair_id": repair["repair_id"],
        "pair_id": repair["candidate_id"],
        "edge_id_a": repair["source_edge_id"],
        "edge_id_b": repair["target_edge_id"],
        "distance_m": float(repair["distance_m"]),
        "rule_ver": config["rule_version"],
    }
    for field, value in audit_values.items():
        audit_feature.SetField(field, value)
    audit_feature.SetGeometry(geometry.Clone())
    audit.CreateFeature(audit_feature)
    dataset = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply approved od_025 topology gap repair")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    logger = setup_logger(resolve(config["outputs"]["log"]), args.overwrite)
    started_at = now_text()
    started = time.perf_counter()
    failed_path = resolve(config["outputs"]["failed_csv"])
    try:
        if not args.approved_by_user or config.get("status") != "APPROVED" or not config.get("approval", {}).get("approved"):
            raise PermissionError("Step24c formal repair requires explicit approval")
        source = resolve(config["input"]["formal_network_gdb"])
        backup = resolve(config["outputs"]["backup_gdb"])
        staging = source.with_name(source.stem + "_step24c_staging.gdb")
        edge_id = str(config["repair"]["synthetic_edge_id"])
        before_count, existing, _ = feature_count_and_match(source, config["input"]["repaired_layer"], edge_id)
        if existing:
            raise RuntimeError(f"Approved connector already exists in formal network: {edge_id}")
        if backup.exists():
            raise FileExistsError(f"Backup already exists; refusing to overwrite: {backup}")
        if staging.exists():
            if not args.overwrite:
                raise FileExistsError(staging)
            shutil.rmtree(staging)

        logger.info("Copying formal Step24 GDB to staging")
        shutil.copytree(source, staging)
        append_connector(staging, config)
        after_count, match_count, measured_length = feature_count_and_match(staging, config["input"]["repaired_layer"], edge_id)
        expected_length = float(config["repair"]["distance_m"])
        blockers: list[str] = []
        if after_count != before_count + 1:
            blockers.append(f"feature count {after_count} != {before_count + 1}")
        if match_count != 1:
            blockers.append(f"synthetic edge match count {match_count} != 1")
        if measured_length is None or not math.isclose(measured_length, expected_length, abs_tol=0.01):
            blockers.append(f"connector length {measured_length} != {expected_length}")
        if blockers:
            raise RuntimeError("; ".join(blockers))

        logger.info("Validation passed; performing versioned GDB swap")
        source.rename(backup)
        try:
            staging.rename(source)
        except Exception:
            backup.rename(source)
            raise

        audit = [{
            "repair_id": config["repair"]["repair_id"],
            "candidate_id": config["repair"]["candidate_id"],
            "synthetic_edge_id": edge_id,
            "source_node": config["repair"]["source_node"],
            "target_node": config["repair"]["target_node"],
            "source_edge_id": config["repair"]["source_edge_id"],
            "target_edge_id": config["repair"]["target_edge_id"],
            "source_x": config["repair"]["source_x"],
            "source_y": config["repair"]["source_y"],
            "target_x": config["repair"]["target_x"],
            "target_y": config["repair"]["target_y"],
            "distance_m": expected_length,
            "repair_type": "approved_aligned_missing_road_connector",
            "approved": True,
        }]
        write_csv(resolve(config["outputs"]["audit_csv"]), audit, list(audit[0]))
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text("record_id,stage,error_message\n", encoding="utf-8")
        finished_at = now_text()
        summary = {
            "step": "24c", "mode": "run", "success": True,
            "started_at": started_at, "finished_at": finished_at,
            "elapsed_seconds": time.perf_counter() - started,
            "rule_version": config["rule_version"], "approved_candidate": config["repair"]["candidate_id"],
            "synthetic_edge_id": edge_id, "formal_network_edge_count_before": before_count,
            "formal_network_edge_count_after": after_count, "connector_length_m": measured_length,
            "original_roads_modified": False, "backup_gdb": str(backup), "formal_gdb": str(source),
            "ready_for_step25_rerun": True, "failed_record_count": 0,
        }
        write_json(resolve(config["outputs"]["summary"]), summary)
        write_json(resolve(config["outputs"]["qc"]), {
            "success": True, "edge_count_increment": after_count - before_count,
            "synthetic_edge_match_count": match_count, "connector_length_matches": True,
            "backup_exists": backup.exists(), "original_roads_modified": False,
        })
        report = [
            "# Step24c 正式拓扑缺口修复报告", "", f"完成时间：{finished_at}", "",
            f"- 获批候选：`{config['repair']['candidate_id']}`",
            f"- 新增连接：`{edge_id}`，长度 {measured_length:.2f} 米",
            f"- 正式路网道路数：{before_count:,} → {after_count:,}",
            "- 原道路几何未移动、未改写。",
            f"- 修复前备份：`{backup}`", "", "`READY_FOR_STEP25_RERUN = TRUE`",
        ]
        report_path = resolve(config["outputs"]["report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        logger.info("Step24c complete: %s", json.dumps(summary, ensure_ascii=False))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        logger.error("Step24c failed: %s", error)
        logger.error(traceback.format_exc())
        write_csv(failed_path, [{"record_id": "gap_0001", "stage": "formal_repair", "error_message": str(error)}], ["record_id", "stage", "error_message"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
