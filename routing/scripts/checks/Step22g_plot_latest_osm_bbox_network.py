from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
ROADS = ROOT / "data/raw/osm_geofabrik/2026-08-01/nanjing_center_bbox_roads_20260801.shp"
BOUNDARY = ROOT / "data/raw/Nanjing_center_boundary/Nanjing_center_UTM50N.shp"
WORK = ROOT / "routing/data/osm_latest_rect"
FIGURES = ROOT / "routing/outputs/figures"
LOG = ROOT / "routing/logs/step22g_latest_osm_bbox_network_qc.log"
MOTOR_ALWAYS = {"motorway", "motorway_link", "trunk", "trunk_link"}
GRADE_CLASSES = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}


def logger() -> logging.Logger:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    result = logging.getLogger("step22g_latest_osm_bbox_qc")
    result.handlers.clear(); result.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG, mode="w", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt); result.addHandler(handler)
    return result


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "t", "y"}


def lines(geometry):
    from osgeo import ogr
    flat = ogr.GT_Flatten(geometry.GetGeometryType())
    if flat in {ogr.wkbLineString, ogr.wkbLinearRing}:
        points = np.asarray([geometry.GetPoint(i)[:2] for i in range(geometry.GetPointCount())], dtype=np.float64)
        if len(points) >= 2:
            yield points
    elif flat in {ogr.wkbMultiLineString, ogr.wkbGeometryCollection}:
        for index in range(geometry.GetGeometryCount()):
            yield from lines(geometry.GetGeometryRef(index))


def main() -> int:
    log = logger(); started = time.perf_counter()
    failed: list[dict[str, str]] = []
    try:
        from osgeo import ogr, osr
        ogr.UseExceptions()
        WORK.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
        road_ds = ogr.Open(str(ROADS), 0)
        if road_ds is None: raise FileNotFoundError(ROADS)
        road_layer = road_ds.GetLayer(0)
        source_srs = road_layer.GetSpatialRef().Clone(); source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs = osr.SpatialReference(); target_srs.ImportFromEPSG(32650); target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source_srs, target_srs)
        route_lines: list[np.ndarray] = []; motor_lines: list[np.ndarray] = []
        graph = nx.Graph(); class_counts: Counter[str] = Counter(); lengths = Counter()
        for feature in tqdm(road_layer, total=road_layer.GetFeatureCount(), desc="Building latest OSM candidate topology", unit="road", dynamic_ncols=True):
            edge_id = str(feature.GetField("osm_id") or feature.GetFID())
            try:
                geometry = feature.GetGeometryRef()
                if geometry is None or geometry.IsEmpty(): raise ValueError("empty_geometry")
                geometry = geometry.Clone(); geometry.Transform(transform)
                fclass = str(feature.GetField("fclass") or "unknown").lower()
                grade = truthy(feature.GetField("bridge")) or truthy(feature.GetField("tunnel")) or int(feature.GetField("layer") or 0) != 0
                motor = fclass in MOTOR_ALWAYS or (fclass in GRADE_CLASSES and grade)
                target = motor_lines if motor else route_lines
                category = "motor_vehicle_only" if motor else "thermal_comfort_relevant"
                for part in lines(geometry):
                    target.append(part); lengths[category] += sum(float(np.linalg.norm(part[i + 1] - part[i])) for i in range(len(part) - 1))
                    if not motor:
                        for index in range(len(part) - 1):
                            a = tuple(np.round(part[index], 3)); b = tuple(np.round(part[index + 1], 3))
                            if a != b: graph.add_edge(a, b)
                class_counts[category] += 1
            except Exception as error:
                failed.append({"edge_id": edge_id, "error_message": str(error)})
        road_ds = None

        boundary_ds = ogr.Open(str(BOUNDARY), 0); boundary_layer = boundary_ds.GetLayer(0)
        boundary_lines: list[np.ndarray] = []
        for feature in tqdm(boundary_layer, total=boundary_layer.GetFeatureCount(), desc="Reading center boundary overlay", unit="feature", dynamic_ncols=True):
            geometry = feature.GetGeometryRef()
            if geometry is None: continue
            boundary_lines.extend(lines(geometry.Boundary()))
        boundary_ds = None
        boundary_points = np.vstack(boundary_lines)
        min_xy = boundary_points.min(axis=0); max_xy = boundary_points.max(axis=0)
        components = list(nx.connected_components(graph)); sizes = sorted((len(c) for c in components), reverse=True)
        degree_counts = Counter(dict(graph.degree()).values())
        summary = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": str(ROADS.resolve()), "scope": "center_boundary_axis_aligned_bounding_rectangle",
            "center_boundary_used_for_clipping": False, "center_boundary_used_for_plot_overlay_only": True,
            "road_feature_count": int(sum(class_counts.values())), "class_counts": dict(class_counts),
            "class_length_km": {key: value / 1000.0 for key, value in lengths.items()},
            "routing_node_count": graph.number_of_nodes(), "routing_segment_count": graph.number_of_edges(),
            "connected_component_count": len(components), "largest_component_node_count": sizes[0] if sizes else 0,
            "largest_component_node_ratio": sizes[0] / graph.number_of_nodes() if graph.number_of_nodes() else 0,
            "degree_1_node_count": int(degree_counts[1]), "failed_count": len(failed),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (WORK / "step22g_latest_osm_bbox_network_qc.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with (WORK / "step22g_failed_records.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["edge_id", "error_message"]); writer.writeheader(); writer.writerows(failed)

        figure, axis = plt.subplots(figsize=(15.5, 13.2), dpi=180)
        axis.set_facecolor("#fbfbf8")
        axis.add_collection(LineCollection(route_lines, colors="#315f78", linewidths=0.22, alpha=0.63, zorder=1))
        axis.add_collection(LineCollection(motor_lines, colors="#c8a5a0", linewidths=0.20, alpha=0.38, zorder=0))
        axis.add_collection(LineCollection(boundary_lines, colors="#111111", linewidths=1.35, alpha=0.95, zorder=3))
        axis.set_xlim(min_xy[0], max_xy[0]); axis.set_ylim(min_xy[1], max_xy[1]); axis.set_aspect("equal")
        axis.set_title("Latest OSM routing network — center-boundary bounding rectangle", fontsize=18, weight="bold", pad=14)
        axis.set_xlabel("Easting (m), EPSG:32650"); axis.set_ylabel("Northing (m), EPSG:32650")
        axis.grid(color="#d9d9d2", linewidth=0.35, alpha=0.45)
        axis.legend(handles=[
            Line2D([0], [0], color="#315f78", lw=2.0, label=f"Thermal-comfort routing roads ({class_counts['thermal_comfort_relevant']:,})"),
            Line2D([0], [0], color="#c8a5a0", lw=2.0, label=f"Motor-vehicle-only ({class_counts['motor_vehicle_only']:,})"),
            Line2D([0], [0], color="#111111", lw=1.6, label="Nanjing center boundary (overlay only)"),
        ], loc="lower right", frameon=True, framealpha=0.94, fontsize=9)
        figure.text(0.01, 0.008, "© OpenStreetMap contributors · Geofabrik extract · OSM data timestamp 2026-08-01 20:21 UTC", fontsize=7.5, color="#555555")
        figure.tight_layout(rect=(0, 0.02, 1, 1))
        png = FIGURES / "latest_osm_bbox_route_network_20260801.png"; pdf = FIGURES / "latest_osm_bbox_route_network_20260801.pdf"
        figure.savefig(png, dpi=240, bbox_inches="tight"); figure.savefig(pdf, bbox_inches="tight"); plt.close(figure)
        log.info("Completed: %s", json.dumps(summary, ensure_ascii=False))
        log.info("Figures: %s ; %s", png, pdf)
        return 0 if not failed else 2
    except Exception as error:
        log.error("Failed: %s\n%s", error, traceback.format_exc()); return 1


if __name__ == "__main__":
    raise SystemExit(main())
