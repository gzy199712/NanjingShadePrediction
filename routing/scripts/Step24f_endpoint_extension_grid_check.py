"""Step24f: full-network endpoint extension and buffered-tip connection audit."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "routing/configs/topology_endpoint_extension_grid_check.yaml"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--mode", required=True, choices=("check", "run")); parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def logger_for(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True); logger = logging.getLogger("step24f")
    logger.handlers.clear(); logger.setLevel(logging.INFO); handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")); logger.addHandler(handler); return logger


def line_parts(geometry: Any) -> Iterable[np.ndarray]:
    from osgeo import ogr
    flat = ogr.GT_Flatten(geometry.GetGeometryType())
    if flat in {ogr.wkbLineString, ogr.wkbLinearRing}:
        points = np.asarray([(geometry.GetX(i), geometry.GetY(i)) for i in range(geometry.GetPointCount())], dtype=np.float64)
        if len(points) >= 2: yield points
    elif flat in {ogr.wkbMultiLineString, ogr.wkbGeometryCollection}:
        for index in range(geometry.GetGeometryCount()): yield from line_parts(geometry.GetGeometryRef(index))


def norm_text(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "t", "true", "yes", "y"}


def node_key(point: np.ndarray, precision: float) -> tuple[int, int]:
    return tuple(np.rint(point / precision).astype(np.int64).tolist())


def unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector)); return vector / length if length > 1e-9 else None


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None: return 180.0
    return float(math.degrees(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0)))))


def project(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, float, float]:
    vector = second - first; denom = float(np.dot(vector, vector))
    fraction = float(np.clip(np.dot(point-first, vector)/denom, 0.0, 1.0)) if denom > 0 else 0.0
    nearest = first + fraction * vector; return nearest, fraction, float(np.linalg.norm(point-nearest))


def structure(edge: dict[str, Any]) -> tuple[int, bool, bool, int]:
    return int(edge["layer"] or 0), truthy(edge["bridge"]), truthy(edge["tunnel"]), int(edge["grade_sep"] or 0)


def main() -> int:
    args = arguments(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); source = ROOT / cfg["input"]["network_gdb"]
    ready = source.is_dir()
    if args.mode == "check":
        print(json.dumps({"status": "READY" if ready else "BLOCKED", "network": str(source),
                          "extension_distances_m": cfg["parameters"]["extension_distances_m"],
                          "tip_search_buffers_m": cfg["parameters"]["tip_search_buffers_m"], "formal_write": False}, ensure_ascii=False, indent=2)); return 0 if ready else 1
    if not ready: raise FileNotFoundError(source)
    output = ROOT / cfg["outputs"]["directory"]
    if output.exists() and any(output.iterdir()) and not args.overwrite: raise RuntimeError("outputs exist; use --overwrite")
    if output.exists() and args.overwrite: shutil.rmtree(output)
    output.mkdir(parents=True); log = logger_for(ROOT / cfg["outputs"]["log"]); started = datetime.now().astimezone(); tick = time.perf_counter()
    os.environ.setdefault("GDAL_PAM_PROXY_DIR", str((output / "gdal_proxy").resolve())); Path(os.environ["GDAL_PAM_PROXY_DIR"]).mkdir(parents=True, exist_ok=True)
    from osgeo import ogr
    ogr.UseExceptions(); dataset = ogr.Open(str(source), 0); layer = dataset.GetLayerByName(cfg["input"]["layer"])
    if layer is None: raise ValueError("network layer unavailable")
    edges: list[dict[str, Any]] = []; failed: list[dict[str, Any]] = []
    fields = ["osm_id","edge_id","fclass","name","ref","layer","bridge","tunnel","grade_sep","repair_typ"]
    for feature in tqdm(layer, total=layer.GetFeatureCount(), desc="Reading current road network", unit="road", dynamic_ncols=True):
        try:
            geom = feature.GetGeometryRef()
            for part_index, coords in enumerate(line_parts(geom)):
                record = {field: feature.GetField(field) for field in fields}; record.update({"fid": int(feature.GetFID()), "part": part_index, "coords": coords})
                edges.append(record)
        except Exception as error: failed.append({"feature_id": feature.GetFID(), "error_message": f"{type(error).__name__}: {error}"})
    precision = float(cfg["parameters"]["endpoint_rounding_m"]); degrees: Counter[tuple[int,int]] = Counter()
    for edge in tqdm(edges, desc="Counting topology endpoints", unit="road", dynamic_ncols=True):
        degrees[node_key(edge["coords"][0], precision)] += 1; degrees[node_key(edge["coords"][-1], precision)] += 1
    grid_size = float(cfg["parameters"]["spatial_grid_size_m"]); grid: dict[tuple[int,int], list[tuple[int,int]]] = defaultdict(list)
    for edge_index, edge in enumerate(tqdm(edges, desc="Building segment spatial grid", unit="road", dynamic_ncols=True)):
        coords = edge["coords"]
        for segment_index in range(len(coords)-1):
            low = np.minimum(coords[segment_index], coords[segment_index+1]); high = np.maximum(coords[segment_index], coords[segment_index+1])
            lo = np.floor(low/grid_size).astype(int); hi = np.floor(high/grid_size).astype(int)
            for gx in range(lo[0],hi[0]+1):
                for gy in range(lo[1],hi[1]+1): grid[(gx,gy)].append((edge_index,segment_index))
    terminals=[]
    motor=set(cfg["hard_reject"]["motor_vehicle_only_classes"])
    for edge_index, edge in enumerate(edges):
        if str(edge["fclass"] or "") in motor or str(edge["repair_typ"] or "").lower() not in {"", "original", "none"}: continue
        coords=edge["coords"]
        for label, endpoint, neighbor in (("start",coords[0],coords[1]),("end",coords[-1],coords[-2])):
            if degrees[node_key(endpoint,precision)]==1:
                direction=unit(endpoint-neighbor)
                if direction is not None: terminals.append((edge_index,label,endpoint,neighbor,direction))
    params=cfg["parameters"]; raw_hits=[]
    combinations=[(float(extension),float(buffer)) for extension in params["extension_distances_m"] for buffer in params["tip_search_buffers_m"]]
    for edge_index,label,endpoint,neighbor,direction in tqdm(terminals, desc="Testing endpoint extension grids", unit="endpoint", dynamic_ncols=True):
        source_edge=edges[edge_index]
        for extension,buffer in combinations:
            tip=endpoint+direction*extension; low=np.floor((tip-buffer)/grid_size).astype(int); high=np.floor((tip+buffer)/grid_size).astype(int)
            targets=set()
            for gx in range(low[0],high[0]+1):
                for gy in range(low[1],high[1]+1): targets.update(grid.get((gx,gy),()))
            best_by_edge={}
            for target_edge_index,segment_index in targets:
                if target_edge_index==edge_index: continue
                target=edges[target_edge_index]
                if str(target["repair_typ"] or "").lower() not in {"", "original", "none"}: continue
                if str(target["fclass"] or "") in motor or structure(source_edge)!=structure(target): continue
                first,target_second=target["coords"][segment_index:segment_index+2]
                nearest,fraction,tip_distance=project(tip,first,target_second)
                if tip_distance>buffer+1e-7: continue
                connector=nearest-endpoint; connector_distance=float(np.linalg.norm(connector))
                if connector_distance<float(params["minimum_connector_m"]): continue
                source_angle=angle_deg(direction,connector)
                if source_angle>float(params["maximum_source_angle_deg"]): continue
                tangent=target_second-first; crossing=min(angle_deg(connector,tangent),angle_deg(connector,-tangent))
                endpoint_tolerance=float(params["endpoint_fraction_tolerance"]); target_endpoint=fraction<=endpoint_tolerance or fraction>=1-endpoint_tolerance
                target_alignment=None
                if target_endpoint:
                    target_node=first if fraction<=endpoint_tolerance else target_second
                    target_inner=target_second if fraction<=endpoint_tolerance else first
                    if degrees[node_key(target_node,precision)]!=1: continue
                    target_alignment=angle_deg(target_node-target_inner,-connector)
                same_name=bool(norm_text(source_edge["name"]) and norm_text(source_edge["name"])==norm_text(target["name"]))
                same_ref=bool(norm_text(source_edge["ref"]) and norm_text(source_edge["ref"])==norm_text(target["ref"]))
                aligned_endpoint=bool(target_endpoint and target_alignment<=float(params["maximum_endpoint_alignment_deg"]) and (same_name or same_ref))
                t_junction=bool(not target_endpoint and crossing>=float(params["minimum_midline_crossing_angle_deg"]))
                auto=aligned_endpoint or t_junction
                row={"source_edge_id":source_edge["edge_id"],"source_osm_id":source_edge["osm_id"],"source_endpoint":label,
                     "source_name":source_edge["name"] or "","source_fclass":source_edge["fclass"],"target_edge_id":target["edge_id"],"target_osm_id":target["osm_id"],
                     "target_name":target["name"] or "","target_fclass":target["fclass"],"source_x":endpoint[0],"source_y":endpoint[1],
                     "target_x":nearest[0],"target_y":nearest[1],"extension_m":extension,"tip_buffer_m":buffer,"connector_length_m":connector_distance,
                     "tip_to_target_m":tip_distance,"source_angle_deg":source_angle,"target_crossing_angle_deg":crossing,
                     "target_is_endpoint":target_endpoint,"target_endpoint_alignment_deg":target_alignment,"same_name":same_name,"same_ref":same_ref,
                     "candidate_type":"same_name_endpoint_continuation" if aligned_endpoint else "midline_t_junction" if t_junction else "review_only",
                     "automatic_candidate":auto,"structure_signature":"|".join(map(str,structure(source_edge)))}
                key=(str(target["edge_id"]),round(float(nearest[0]),2),round(float(nearest[1]),2))
                if key not in best_by_edge or connector_distance<best_by_edge[key]["connector_length_m"]: best_by_edge[key]=row
            raw_hits.extend(best_by_edge.values())
    # Keep the smallest parameter combination that discovers each physical connection.
    best={}
    for row in raw_hits:
        key=(str(row["source_edge_id"]),row["source_endpoint"],str(row["target_edge_id"]),round(float(row["target_x"]),2),round(float(row["target_y"]),2))
        rank=(not row["automatic_candidate"],row["extension_m"]+row["tip_buffer_m"],row["connector_length_m"])
        if key not in best or rank<best[key][0]: best[key]=(rank,row)
    candidates=[item[1] for item in best.values()]; candidates.sort(key=lambda row:(not row["automatic_candidate"],row["candidate_type"],row["connector_length_m"]))
    for index,row in enumerate(candidates,1): row["candidate_id"]=f"ext_{index:05d}"
    fields_out=["candidate_id","candidate_type","automatic_candidate","source_edge_id","source_osm_id","source_endpoint","source_name","source_fclass",
                "target_edge_id","target_osm_id","target_name","target_fclass","source_x","source_y","target_x","target_y","extension_m","tip_buffer_m",
                "connector_length_m","tip_to_target_m","source_angle_deg","target_crossing_angle_deg","target_is_endpoint","target_endpoint_alignment_deg",
                "same_name","same_ref","structure_signature"]
    with (output/"endpoint_extension_candidates.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields_out);writer.writeheader();writer.writerows(candidates)
    with (output/"failed_files.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["feature_id","error_message"]);writer.writeheader();writer.writerows(failed)
    automatic=[row for row in candidates if row["automatic_candidate"]]; type_counts=Counter(row["candidate_type"] for row in candidates)
    summary={"status":"STEP24F_ENDPOINT_EXTENSION_GRID_CHECK_COMPLETE","started_at":started.isoformat(),"ended_at":datetime.now().astimezone().isoformat(),
             "runtime_seconds":round(time.perf_counter()-tick,3),"network_edge_count":len(edges),"degree_one_endpoint_count":len(terminals),
             "parameter_combinations":[{"extension_m":a,"tip_buffer_m":b} for a,b in combinations],"candidate_count":len(candidates),
             "automatic_candidate_count":len(automatic),"candidate_type_counts":dict(type_counts),"same_name_automatic_count":sum(r["candidate_type"]=="same_name_endpoint_continuation" for r in automatic),
             "midline_t_junction_automatic_count":sum(r["candidate_type"]=="midline_t_junction" for r in automatic),"failure_count":len(failed),
             "formal_network_modified":False,"claim_boundary":"Geometric and topology candidate audit only; no connector has been written to the formal network."}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    report=ROOT/cfg["outputs"]["report"]
    report.write_text("# Step24f：全路网端点延伸与缓冲搜索预检\n\n"+f"- 路段：{len(edges):,}\n- 度为1端点：{len(terminals):,}\n- 候选：{len(candidates):,}\n- 自动规则候选：{len(automatic):,}\n- 同名续接：{summary['same_name_automatic_count']:,}\n- T形接入：{summary['midline_t_junction_automatic_count']:,}\n- 正式路网修改：否\n\n"+
        "组合测试端点向前延伸10/20/30米，并在新端点构建5/10米搜索范围。桥梁、隧道、layer、grade separation、机动车专用道路与既有修复连接均硬排除。自动候选仅包括同名/同ref双端对齐续接，或满足交叉角门槛的道路中段T形接入。\n",encoding="utf-8")
    log.info("completed %s",json.dumps(summary,ensure_ascii=False));print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if not failed else 2


if __name__=="__main__": raise SystemExit(main())
