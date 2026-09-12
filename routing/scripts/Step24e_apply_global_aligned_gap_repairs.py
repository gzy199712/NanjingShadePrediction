"""Step24e: versioned application of the approved global aligned-gap shortlist."""

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

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "routing/configs/global_aligned_gap_repair_approved.yaml"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve(value: str) -> Path:
    return (ROOT / value).resolve()


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step24e_global_gap_repair"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="w" if overwrite else "a", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in tqdm(rows, total=len(rows), desc=f"Writing {path.name}", unit="repair", dynamic_ncols=True):
            writer.writerow(row)


def layer_counts(gdb: Path, repaired_name: str, connector_name: str, edge_ids: set[str]) -> dict[str, Any]:
    from osgeo import ogr
    dataset = ogr.Open(str(gdb), 0)
    if dataset is None:
        raise FileNotFoundError(gdb)
    repaired, connectors = dataset.GetLayerByName(repaired_name), dataset.GetLayerByName(connector_name)
    if repaired is None or connectors is None:
        raise KeyError("Required formal topology layers missing")
    found: dict[str, float] = {}
    for feature in repaired:
        edge_id = str(feature.GetField("edge_id"))
        if edge_id in edge_ids:
            geometry = feature.GetGeometryRef(); found[edge_id] = float(geometry.Length()) if geometry else math.nan
    result = {"repaired_count": int(repaired.GetFeatureCount()), "connector_count": int(connectors.GetFeatureCount()), "found_lengths": found}
    dataset = None
    return result


def append_repairs(gdb: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from osgeo import ogr
    ogr.UseExceptions()
    dataset = ogr.Open(str(gdb), 1)
    repaired = dataset.GetLayerByName(config["input"]["repaired_layer"])
    connectors = dataset.GetLayerByName(config["input"]["connector_layer"])
    audits: list[dict[str, Any]] = []
    for index, row in enumerate(tqdm(rows, total=len(rows), desc="Writing approved global gap connectors", unit="repair", dynamic_ncols=True), start=1):
        edge_id = f"EXT_GLOBAL_{index:05d}"
        repair_id = f"TR{58 + index:05d}"
        geometry = ogr.Geometry(ogr.wkbLineString)
        geometry.AddPoint_2D(float(row["source_x"]), float(row["source_y"])); geometry.AddPoint_2D(float(row["target_x"]), float(row["target_y"]))
        feature = ogr.Feature(repaired.GetLayerDefn())
        values = {
            "osm_id": edge_id, "fclass": "topology_connector", "name": "approved_global_aligned_gap_connector",
            "edge_id": edge_id, "therm_class": "thermal_comfort_relevant",
            "class_basis": "approved_global_aligned_gap_repair", "grade_sep": 0,
            "rule_ver": config["rule_version"], "repair_id": repair_id, "repair_typ": "aligned_gap_connector",
            "source_a": str(row["source_edge_id"]), "source_b": str(row["target_edge_id"]), "snap_dist": float(row["gap_m"]),
        }
        for field, value in values.items():
            if feature.GetFieldIndex(field) >= 0:
                feature.SetField(field, value)
        feature.SetGeometry(geometry); repaired.CreateFeature(feature)
        audit_feature = ogr.Feature(connectors.GetLayerDefn())
        for field, value in {
            "repair_id": repair_id, "pair_id": str(row["candidate_id"]), "edge_id_a": str(row["source_edge_id"]),
            "edge_id_b": str(row["target_edge_id"]), "distance_m": float(row["gap_m"]), "rule_ver": config["rule_version"],
        }.items():
            audit_feature.SetField(field, value)
        audit_feature.SetGeometry(geometry.Clone()); connectors.CreateFeature(audit_feature)
        audits.append({
            "repair_id": repair_id, "candidate_id": row["candidate_id"], "synthetic_edge_id": edge_id,
            "source_node": row["source_node"], "target_node": row["target_node"],
            "source_edge_id": row["source_edge_id"], "target_edge_id": row["target_edge_id"],
            "source_x": row["source_x"], "source_y": row["source_y"], "target_x": row["target_x"], "target_y": row["target_y"],
            "distance_m": row["gap_m"], "source_angle_deg": row["source_angle_deg"], "target_angle_deg": row["target_angle_deg"],
            "repair_type": "approved_global_aligned_gap_connector", "approved": True,
        })
    dataset = None
    return audits


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply approved global aligned road-gap repairs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    logger = setup_logger(resolve(config["outputs"]["log"]), args.overwrite)
    started_at, started = now_text(), time.perf_counter()
    failed_path = resolve(config["outputs"]["failed_csv"])
    try:
        if not args.approved_by_user or config.get("status") != "APPROVED" or not config.get("approval", {}).get("approved"):
            raise PermissionError("Step24e requires explicit user approval")
        frame = pd.read_csv(resolve(config["input"]["shortlist"]), dtype=str).fillna("")
        expected = int(config["validation"]["expected_candidate_count"])
        if len(frame) != expected or not (frame["high_confidence"].str.lower() == "true").all():
            raise ValueError(f"Approved shortlist validation failed: expected {expected}, found {len(frame)}")
        if (frame["gap_m"].astype(float) > float(config["validation"]["maximum_gap_m"])).any():
            raise ValueError("Shortlist contains an over-length automatic repair")
        if (frame[["source_angle_deg", "target_angle_deg"]].astype(float) > float(config["validation"]["maximum_endpoint_angle_deg"])).any().any():
            raise ValueError("Shortlist contains an over-angle repair")
        rows = frame.to_dict("records")
        source = resolve(config["input"]["formal_network_gdb"]); backup = resolve(config["outputs"]["backup_gdb"])
        staging = source.with_name(source.stem + "_step24e_staging.gdb")
        edge_ids = {f"EXT_GLOBAL_{index:05d}" for index in range(1, expected + 1)}
        before = layer_counts(source, config["input"]["repaired_layer"], config["input"]["connector_layer"], edge_ids)
        if before["found_lengths"]:
            raise RuntimeError("One or more Step24e synthetic edges already exist")
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
        if staging.exists():
            if not args.overwrite:
                raise FileExistsError(staging)
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        audits = append_repairs(staging, config, rows)
        after = layer_counts(staging, config["input"]["repaired_layer"], config["input"]["connector_layer"], edge_ids)
        blockers = []
        if after["repaired_count"] != before["repaired_count"] + expected:
            blockers.append("repaired feature count mismatch")
        if after["connector_count"] != before["connector_count"] + expected:
            blockers.append("connector audit count mismatch")
        if set(after["found_lengths"]) != edge_ids:
            blockers.append("synthetic edge ID mismatch")
        for audit in audits:
            if not math.isclose(after["found_lengths"][audit["synthetic_edge_id"]], float(audit["distance_m"]), abs_tol=0.01):
                blockers.append(f"length mismatch: {audit['synthetic_edge_id']}")
        if blockers:
            raise RuntimeError("; ".join(blockers))
        source.rename(backup)
        try:
            staging.rename(source)
        except Exception:
            backup.rename(source); raise
        write_csv(resolve(config["outputs"]["audit_csv"]), audits, list(audits[0]))
        write_csv(failed_path, [], ["record_id", "stage", "error_message"])
        summary = {
            "step": "24e", "mode": "run", "success": True, "started_at": started_at, "finished_at": now_text(),
            "elapsed_seconds": time.perf_counter() - started, "rule_version": config["rule_version"],
            "approved_connector_count": expected, "total_connector_length_m": sum(float(row["gap_m"]) for row in rows),
            "formal_network_edge_count_before": before["repaired_count"], "formal_network_edge_count_after": after["repaired_count"],
            "connector_audit_count_before": before["connector_count"], "connector_audit_count_after": after["connector_count"],
            "original_roads_modified": False, "backup_gdb": str(backup), "formal_gdb": str(source),
            "ready_for_step25_rerun": True, "failed_record_count": 0,
        }
        atomic_json(resolve(config["outputs"]["summary"]), summary)
        atomic_json(resolve(config["outputs"]["qc"]), {"success": True, "blockers": [], "synthetic_edges": sorted(edge_ids), "original_roads_modified": False, "backup_exists": backup.exists()})
        report = [
            "# Step24e 全局共线断点正式修复报告", "", f"完成时间：{summary['finished_at']}", "",
            f"- 正式连接：{expected}条", f"- 总长度：{summary['total_connector_length_m']:.2f}米",
            f"- 正式路网：{before['repaired_count']:,} → {after['repaired_count']:,}",
            f"- 连接审计：{before['connector_count']:,} → {after['connector_count']:,}",
            "- 原道路未移动、未改写。", f"- 修复前备份：`{backup}`", "", "`READY_FOR_STEP25_RERUN = TRUE`",
        ]
        report_path = resolve(config["outputs"]["report"]); report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        logger.info("Step24e complete: %s", json.dumps(summary, ensure_ascii=False)); print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        logger.error("Step24e failed: %s", error); logger.error(traceback.format_exc())
        write_csv(failed_path, [{"record_id": "step24e", "stage": "formal_repair", "error_message": str(error)}], ["record_id", "stage", "error_message"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
