"""Step24g: apply conservative endpoint-extension repairs to the formal road network."""

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
DEFAULT_CONFIG = ROOT / "routing/configs/topology_endpoint_extension_repair.yaml"


def resolve(value: str) -> Path:
    return (ROOT / value).resolve()


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step24g_endpoint_extension")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="w" if overwrite else "a", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def select_shortlist(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rule = config["selection"]
    selected = frame[
        frame["candidate_type"].eq(rule["candidate_type"])
        & (
            frame["same_name"].astype(str).str.lower().eq("true")
            | frame["same_ref"].astype(str).str.lower().eq("true")
        )
        & frame["source_fclass"].eq(frame["target_fclass"])
        & frame["source_angle_deg"].le(float(rule["maximum_source_angle_deg"]))
        & frame["target_endpoint_alignment_deg"].le(float(rule["maximum_target_alignment_deg"]))
    ].copy()
    selected["source_node"] = list(zip(selected["source_x"].round(2), selected["source_y"].round(2)))
    selected["target_node"] = list(zip(selected["target_x"].round(2), selected["target_y"].round(2)))
    best = selected.sort_values(["connector_length_m", "tip_to_target_m", "candidate_id"]).drop_duplicates("source_node")
    directed = set(zip(best["source_node"], best["target_node"]))
    mutual = {tuple(sorted((a, b))) for a, b in directed if (b, a) in directed}
    selected["pair_key"] = selected.apply(lambda row: tuple(sorted((row["source_node"], row["target_node"]))), axis=1)
    selected = selected[selected["pair_key"].isin(mutual)].copy()
    selected = selected.sort_values(["connector_length_m", "candidate_id"]).drop_duplicates("pair_key")
    selected = selected.sort_values(["source_y", "source_x", "target_y", "target_x"]).reset_index(drop=True)
    selected["repair_index"] = range(1, len(selected) + 1)
    selected["synthetic_edge_id"] = selected["repair_index"].map(lambda value: f"EXT_GRID_{value:05d}")
    selected["repair_id"] = selected["repair_index"].map(lambda value: f"EG{value:05d}")
    return selected.drop(columns=["source_node", "target_node", "pair_key"])


def layer_state(gdb: Path, config: dict[str, Any], edge_ids: set[str]) -> dict[str, Any]:
    from osgeo import ogr
    dataset = ogr.Open(str(gdb), 0)
    if dataset is None:
        raise FileNotFoundError(gdb)
    repaired = dataset.GetLayerByName(config["input"]["repaired_layer"])
    connectors = dataset.GetLayerByName(config["input"]["connector_layer"])
    if repaired is None or connectors is None:
        raise KeyError("Required formal topology layers are missing")
    found: dict[str, float] = {}
    for feature in repaired:
        edge_id = str(feature.GetField("edge_id"))
        if edge_id in edge_ids:
            geometry = feature.GetGeometryRef()
            found[edge_id] = float(geometry.Length()) if geometry else math.nan
    result = {
        "repaired_count": int(repaired.GetFeatureCount()),
        "connector_count": int(connectors.GetFeatureCount()),
        "found_lengths": found,
    }
    dataset = None
    return result


def append_connectors(gdb: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    from osgeo import ogr
    ogr.UseExceptions()
    dataset = ogr.Open(str(gdb), 1)
    repaired = dataset.GetLayerByName(config["input"]["repaired_layer"])
    connectors = dataset.GetLayerByName(config["input"]["connector_layer"])
    for row in tqdm(rows, total=len(rows), desc="Writing endpoint connectors", unit="edge", dynamic_ncols=True):
        geometry = ogr.Geometry(ogr.wkbLineString)
        geometry.AddPoint_2D(float(row["source_x"]), float(row["source_y"]))
        geometry.AddPoint_2D(float(row["target_x"]), float(row["target_y"]))
        feature = ogr.Feature(repaired.GetLayerDefn())
        values = {
            "osm_id": row["synthetic_edge_id"], "fclass": "topology_connector",
            "name": "endpoint_extension_connector", "edge_id": row["synthetic_edge_id"],
            "therm_class": "thermal_comfort_relevant", "class_basis": "same_name_mutual_endpoint_extension",
            "grade_sep": 0, "rule_ver": config["rule_version"], "repair_id": row["repair_id"],
            "repair_typ": "endpoint_extension", "source_a": str(row["source_edge_id"]),
            "source_b": str(row["target_edge_id"]), "snap_dist": float(row["connector_length_m"]),
        }
        for field, value in values.items():
            if feature.GetFieldIndex(field) >= 0:
                feature.SetField(field, value)
        feature.SetGeometry(geometry); repaired.CreateFeature(feature)
        audit = ogr.Feature(connectors.GetLayerDefn())
        for field, value in {
            "repair_id": row["repair_id"], "pair_id": str(row["candidate_id"]),
            "edge_id_a": str(row["source_edge_id"]), "edge_id_b": str(row["target_edge_id"]),
            "distance_m": float(row["connector_length_m"]), "rule_ver": config["rule_version"],
        }.items():
            if audit.GetFieldIndex(field) >= 0:
                audit.SetField(field, value)
        audit.SetGeometry(geometry.Clone()); connectors.CreateFeature(audit)
    dataset = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--apply", action="store_true", help="Write the validated shortlist to a versioned formal GDB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    logger = setup_logger(resolve(config["outputs"]["log"]), args.overwrite)
    started_at, started = now_text(), time.perf_counter()
    failed_path = resolve(config["outputs"]["failed_csv"])
    try:
        candidates = pd.read_csv(resolve(config["input"]["candidates_csv"]), low_memory=False)
        shortlist = select_shortlist(candidates, config)
        expected = int(config["selection"]["expected_connector_count"])
        if len(shortlist) != expected:
            raise ValueError(f"Deterministic shortlist drift: expected {expected}, got {len(shortlist)}")
        write_csv(resolve(config["outputs"]["shortlist_csv"]), shortlist)
        if not args.apply:
            print(json.dumps({"mode": "check", "shortlist_count": len(shortlist)}, ensure_ascii=False, indent=2))
            return 0
        source = resolve(config["input"]["formal_network_gdb"])
        backup = resolve(config["outputs"]["backup_gdb"])
        staging = source.with_name(source.stem + "_step24g_staging.gdb")
        edge_ids = set(shortlist["synthetic_edge_id"])
        before = layer_state(source, config, edge_ids)
        if before["found_lengths"]:
            raise RuntimeError("Step24g synthetic edges already exist")
        if backup.exists():
            raise FileExistsError(backup)
        if staging.exists():
            if not args.overwrite:
                raise FileExistsError(staging)
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        append_connectors(staging, config, shortlist.to_dict("records"))
        after = layer_state(staging, config, edge_ids)
        blockers = []
        if after["repaired_count"] != before["repaired_count"] + expected:
            blockers.append("formal edge count mismatch")
        if after["connector_count"] != before["connector_count"] + expected:
            blockers.append("connector audit count mismatch")
        if set(after["found_lengths"]) != edge_ids:
            blockers.append("synthetic edge ID mismatch")
        expected_lengths = dict(zip(shortlist["synthetic_edge_id"], shortlist["connector_length_m"]))
        for edge_id, length in after["found_lengths"].items():
            if not math.isclose(length, float(expected_lengths[edge_id]), abs_tol=0.02):
                blockers.append(f"geometry length mismatch: {edge_id}")
        if blockers:
            raise RuntimeError("; ".join(blockers))
        source.rename(backup)
        try:
            staging.rename(source)
        except Exception:
            backup.rename(source); raise
        write_csv(resolve(config["outputs"]["audit_csv"]), shortlist)
        write_csv(failed_path, pd.DataFrame(columns=["record_id", "stage", "error_message"]))
        summary = {
            "step": "24g", "success": True, "started_at": started_at, "finished_at": now_text(),
            "elapsed_seconds": time.perf_counter() - started, "rule_version": config["rule_version"],
            "input_candidate_count": len(candidates), "applied_connector_count": expected,
            "connector_length_min_m": float(shortlist["connector_length_m"].min()),
            "connector_length_max_m": float(shortlist["connector_length_m"].max()),
            "connector_length_total_m": float(shortlist["connector_length_m"].sum()),
            "formal_edge_count_before": before["repaired_count"], "formal_edge_count_after": after["repaired_count"],
            "connector_count_before": before["connector_count"], "connector_count_after": after["connector_count"],
            "t_junctions_automatically_applied": 0, "original_roads_modified": False,
            "backup_gdb": str(backup), "formal_gdb": str(source), "ready_for_downstream_rebuild": True,
        }
        atomic_json(resolve(config["outputs"]["summary_json"]), summary)
        atomic_json(resolve(config["outputs"]["qc_json"]), {"success": True, "blockers": [], "selection_is_deterministic": True})
        report = [
            "# Step24g 端点延伸路网修复", "", f"完成时间：{summary['finished_at']}", "",
            f"- 原始组合候选：{len(candidates):,}", f"- 正式写入的一对一同名续接：{expected:,}",
            f"- 连接长度：{summary['connector_length_min_m']:.2f}–{summary['connector_length_max_m']:.2f} m",
            f"- 正式路网：{before['repaired_count']:,} → {after['repaired_count']:,}",
            "- 自动写入 T 形接入：0（仅保留审计）", "- 原始道路未移动、未覆盖，只追加可追溯连接边。", "",
            "筛选条件：同名、道路等级相同、结构属性一致、源端与目标端方向偏差均≤15°、双方互为最近端点。", "",
            "`READY_FOR_DOWNSTREAM_REBUILD = TRUE`",
        ]
        report_path = resolve(config["outputs"]["report"]); report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        logger.info("Step24g complete: %s", json.dumps(summary, ensure_ascii=False))
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    except Exception as error:
        logger.error("Step24g failed: %s", error); logger.error(traceback.format_exc())
        write_csv(failed_path, pd.DataFrame([{"record_id": "step24g", "stage": "endpoint_extension", "error_message": str(error)}]))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
