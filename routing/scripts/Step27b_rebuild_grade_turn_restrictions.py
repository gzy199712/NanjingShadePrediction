from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
COVERAGE = ROOT / "routing/data/thermal_edges/thermal_segment_coverage.csv"
SEGMENTS = ROOT / "routing/data/thermal_edges/step26_thermal_segments.gdb"
CLASSIFICATION = ROOT / "routing/data/candidate_network/road_thermal_classification.csv"
OUTPUT = ROOT / "routing/configs/approved_grade_turn_restrictions.csv"
ARCHIVE = ROOT / "routing/configs/archive/approved_grade_turn_restrictions_pre_latest_osm_20260803.csv"
SUMMARY = ROOT / "routing/data/graph/step27b_turn_restriction_summary.json"
FAILED = ROOT / "routing/data/graph/step27b_failed_records.csv"
LOG = ROOT / "routing/logs/step27b_rebuild_grade_turn_restrictions.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild deterministic grade-separated turn restrictions for the current Step26 graph.")
    parser.add_argument("--maximum-aligned-angle-deg", type=float, default=45.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    result = logging.getLogger("step27b_turn_restrictions"); result.handlers.clear(); result.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG, mode="w", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); result.addHandler(handler)
    return result


def endpoint_key(point) -> tuple[float, float]:
    return round(float(point[0]), 3), round(float(point[1]), 3)


def truthy(value: object) -> bool:
    return str(value or "").strip().upper() not in {"", "0", "F", "FALSE", "N", "NO", "NULL", "NONE"}


def main() -> int:
    args = parse_args(); log = setup_logger(); started = time.perf_counter(); failed = []
    try:
        from osgeo import ogr
        ogr.UseExceptions()
        if OUTPUT.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {OUTPUT}; use --overwrite")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        if OUTPUT.exists() and not ARCHIVE.exists():
            shutil.copy2(OUTPUT, ARCHIVE)
        coverage = pd.read_csv(COVERAGE, dtype={"segment_id": str, "original_edge_id": str}, low_memory=False)
        classification = pd.read_csv(CLASSIFICATION, dtype={"edge_id": str}, low_memory=False)
        road_meta = classification[["edge_id", "name", "ref"]].drop_duplicates("edge_id").set_index("edge_id")
        segment_ids = coverage["segment_id"].to_numpy(str)
        original_ids = coverage["original_edge_id"].to_numpy(str)
        lookup = {value: index for index, value in enumerate(segment_ids)}
        names = road_meta["name"].reindex(original_ids).fillna("").astype(str).to_numpy()
        refs = road_meta["ref"].reindex(original_ids).fillna("").astype(str).to_numpy()
        fclasses = coverage["fclass"].fillna("unknown").astype(str).to_numpy()
        bridge = coverage["bridge"].map(truthy).to_numpy(bool)
        tunnel = coverage["tunnel"].map(truthy).to_numpy(bool)
        layers = coverage["layer"].fillna(0).to_numpy(float)
        structure = list(zip(bridge.tolist(), tunnel.tolist(), np.round(layers, 6).tolist()))

        node_lookup: dict[tuple[float, float], int] = {}; node_xy = []
        edge_u = np.full(len(coverage), -1, np.int32); edge_v = np.full(len(coverage), -1, np.int32)
        adjacency: list[list[int]] = []
        ds = ogr.Open(str(SEGMENTS), 0); layer = ds.GetLayerByName("ThermalCostSegments")
        for feature in tqdm(layer, total=layer.GetFeatureCount(), desc="Indexing current Step26 nodes", unit="segment", dynamic_ncols=True):
            segment_id = str(feature.GetField("segment_id")); index = lookup[segment_id]
            geometry = feature.GetGeometryRef(); line = geometry.GetGeometryRef(0) if geometry.GetGeometryCount() else geometry
            first = endpoint_key(line.GetPoint(0)); second = endpoint_key(line.GetPoint(line.GetPointCount() - 1))
            for point in (first, second):
                if point not in node_lookup:
                    node_lookup[point] = len(node_xy); node_xy.append(point); adjacency.append([])
            edge_u[index] = node_lookup[first]; edge_v[index] = node_lookup[second]
            adjacency[edge_u[index]].append(index); adjacency[edge_v[index]].append(index)
        ds = None
        if (edge_u < 0).any() or (edge_v < 0).any(): raise ValueError("Some segments lack graph endpoints")
        xy = np.asarray(node_xy, dtype=float); rows = []; mixed_nodes = incompatible = 0
        for node, incident in enumerate(tqdm(adjacency, desc="Deriving grade-separated turn restrictions", unit="node", dynamic_ncols=True)):
            edges = sorted(set(incident))
            if len(edges) < 2 or len({structure[e] for e in edges}) < 2: continue
            mixed_nodes += 1
            for a_pos in range(len(edges) - 1):
                for b_pos in range(a_pos + 1, len(edges)):
                    first, second = edges[a_pos], edges[b_pos]
                    if structure[first] == structure[second]: continue
                    incompatible += 1
                    first_other = int(edge_v[first]) if int(edge_u[first]) == node else int(edge_u[first])
                    second_other = int(edge_v[second]) if int(edge_u[second]) == node else int(edge_u[second])
                    incoming = xy[node] - xy[first_other]; outgoing = xy[second_other] - xy[node]
                    denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
                    cosine = float(np.dot(incoming, outgoing) / denominator) if denominator else 1.0
                    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                    same_identity = bool((names[first] and names[first] == names[second]) or (refs[first] and refs[first] == refs[second]))
                    if same_identity or angle <= args.maximum_aligned_angle_deg: continue
                    rows.append({
                        "node_id": node, "node_x": xy[node, 0], "node_y": xy[node, 1],
                        "from_segment_id": segment_ids[first], "to_segment_id": segment_ids[second],
                        "from_original_edge_id": original_ids[first], "to_original_edge_id": original_ids[second],
                        "from_fclass": fclasses[first], "to_fclass": fclasses[second],
                        "from_name": names[first], "to_name": names[second], "from_ref": refs[first], "to_ref": refs[second],
                        "from_bridge": bool(bridge[first]), "to_bridge": bool(bridge[second]),
                        "from_tunnel": bool(tunnel[first]), "to_tunnel": bool(tunnel[second]),
                        "from_layer": float(layers[first]), "to_layer": float(layers[second]),
                        "turn_angle_deg": angle, "same_nonempty_name_or_ref": same_identity,
                        "review_decision": "suspicious_grade_jump", "formal_graph_repaired": False,
                    })
        fields = list(rows[0]) if rows else ["node_id", "from_segment_id", "to_segment_id"]
        with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        FAILED.parent.mkdir(parents=True, exist_ok=True)
        FAILED.write_text("object_id,error_message\n", encoding="utf-8")
        summary = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "method": "deterministic_bridge_tunnel_layer_conflict_with_name_ref_and_45deg_exemptions",
            "segment_count": len(segment_ids), "node_count": len(node_xy), "mixed_grade_node_count": mixed_nodes,
            "incompatible_structure_pair_count": incompatible, "restriction_count": len(rows),
            "maximum_aligned_angle_deg": args.maximum_aligned_angle_deg, "failed_count": 0,
            "source_old_restrictions_archive": str(ARCHIVE), "output": str(OUTPUT), "elapsed_seconds": time.perf_counter() - started,
        }
        SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Completed: %s", json.dumps(summary, ensure_ascii=False)); return 0
    except Exception as error:
        log.error("Failed: %s\n%s", error, traceback.format_exc()); return 1


if __name__ == "__main__":
    raise SystemExit(main())
