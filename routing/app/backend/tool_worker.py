"""Long-lived worker for the deterministic routing tools."""

from __future__ import annotations

import csv
import base64
import heapq
import io
import json
import os
import shutil
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GDAL_PROXY = ROOT / "routing" / "data" / "application" / "gdal_proxy"
GDAL_PROXY.mkdir(parents=True, exist_ok=True)
os.environ["GDAL_PAM_PROXY_DIR"] = str(GDAL_PROXY)

from osgeo import ogr
from osgeo import gdal
import numpy as np
from PIL import Image
from routing.src.route_interface import RouteToolInterface


BUILDINGS = (
    ROOT
    / "data"
    / "raw"
    / "SOLWEIG20240729"
    / "BuildingFootprint"
    / "BuildingHeightVector.shp"
)
INDEXED_BUILDINGS = (
    ROOT / "routing" / "data" / "application" / "visual_cache"
    / "BuildingHeightVector_indexed.gpkg"
)
SHADOWS = (
    ROOT / "data" / "raw" / "SOLWEIG20240729" / "HourlyShade2m"
)
BUILDING_RASTER_CACHE: OrderedDict[
    tuple[float, float, float, float, int], dict[str, Any]
] = OrderedDict()
MAX_BUILDING_RASTER_CACHE_ITEMS = 16


def polygon_rings(geometry: Any) -> list[list[list[float]]]:
    rings: list[list[list[float]]] = []
    geometry_type = ogr.GT_Flatten(geometry.GetGeometryType())
    if geometry_type == ogr.wkbPolygon:
        if geometry.GetGeometryCount():
            ring = geometry.GetGeometryRef(0)
            coordinates = [
                [round(ring.GetX(index), 1), round(ring.GetY(index), 1)]
                for index in range(ring.GetPointCount())
            ]
            if len(coordinates) >= 4:
                rings.append(coordinates)
        return rings
    if geometry_type == ogr.wkbMultiPolygon:
        for index in range(geometry.GetGeometryCount()):
            rings.extend(polygon_rings(geometry.GetGeometryRef(index)))
    return rings


def rasterize_buildings(
    layer: Any, bbox: list[float], max_dimension: int = 1200
) -> dict[str, Any]:
    """Render every footprint in the requested extent into a lightweight RGBA tile."""
    min_x, min_y, max_x, max_y = bbox
    key = tuple(round(value, 1) for value in bbox) + (max_dimension,)
    cached = BUILDING_RASTER_CACHE.get(key)
    if cached is not None:
        BUILDING_RASTER_CACHE.move_to_end(key)
        return cached

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min(1.0, max_dimension / max(span_x, span_y))
    width = max(1, round(span_x * scale))
    height = max(1, round(span_y * scale))
    memory = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
    memory.SetGeoTransform((min_x, span_x / width, 0.0, max_y, 0.0, -span_y / height))
    spatial_reference = layer.GetSpatialRef()
    if spatial_reference is not None:
        memory.SetProjection(spatial_reference.ExportToWkt())
    band = memory.GetRasterBand(1)
    band.Fill(0)
    layer.ResetReading()
    status = gdal.RasterizeLayer(memory, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    if status != 0:
        raise RuntimeError("Unable to rasterize the citywide building layer")
    mask = band.ReadAsArray().astype(bool)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = 161
    rgba[..., 1] = 169
    rgba[..., 2] = 166
    rgba[..., 3] = np.where(mask, 82, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG", compress_level=3)
    result = {
        "building_extent": [min_x, min_y, max_x, max_y],
        "building_width": width,
        "building_height": height,
        "building_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
    BUILDING_RASTER_CACHE[key] = result
    while len(BUILDING_RASTER_CACHE) > MAX_BUILDING_RASTER_CACHE_ITEMS:
        BUILDING_RASTER_CACHE.popitem(last=False)
    return result


def build_visual_layers(
    bbox: list[float], hour: int, max_buildings: int,
    raster_max_dimension: int = 1200, include_vector_features: bool = True,
) -> dict[str, Any]:
    min_x, min_y, max_x, max_y = map(float, bbox)
    building_source = INDEXED_BUILDINGS if INDEXED_BUILDINGS.exists() else BUILDINGS
    dataset = ogr.Open(str(building_source), 0)
    if dataset is None:
        raise RuntimeError("Building height source is unavailable")
    layer = dataset.GetLayer(0)
    layer.SetSpatialFilterRect(min_x, min_y, max_x, max_y)
    matched_count = int(layer.GetFeatureCount())
    building_raster = rasterize_buildings(
        layer, [min_x, min_y, max_x, max_y], raster_max_dimension
    )
    simplify_tolerance = max(1.0, max(max_x - min_x, max_y - min_y) / 1800)
    candidates: list[tuple[float, int, float, list[list[list[float]]]]] = []
    serial = 0
    # Full-city requests use the complete raster tile. Vector extrusion remains
    # available for smaller extents and for backward-compatible API clients.
    include_vectors = include_vector_features and max(max_x - min_x, max_y - min_y) <= 10000
    layer.ResetReading()
    for feature in layer if include_vectors else ():
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        area = float(geometry.GetArea())
        if area < 15:
            continue
        simplified = geometry.SimplifyPreserveTopology(simplify_tolerance)
        rings = polygon_rings(simplified)
        if not rings:
            continue
        height = max(2.5, min(float(feature.GetField("Height") or 8.0), 150.0))
        item = (area, serial, height, rings)
        serial += 1
        if len(candidates) < max_buildings:
            heapq.heappush(candidates, item)
        elif area > candidates[0][0]:
            heapq.heapreplace(candidates, item)
    dataset = None
    buildings = [
        {"height": round(height, 1), "rings": rings}
        for _area, _serial, height, rings in sorted(candidates, reverse=True)
    ]

    raster_path = SHADOWS / f"Shadow_20240729_{hour:02d}00_LST.tif"
    shadow_dataset = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
    if shadow_dataset is None:
        raise RuntimeError("Hourly SOLWEIG shadow source is unavailable")
    transform = shadow_dataset.GetGeoTransform()
    raster_min_x = transform[0]
    raster_max_y = transform[3]
    pixel_width = transform[1]
    pixel_height = abs(transform[5])
    x_offset = max(0, int((min_x - raster_min_x) // pixel_width))
    y_offset = max(0, int((raster_max_y - max_y) // pixel_height))
    x_end = min(
        shadow_dataset.RasterXSize,
        int(np.ceil((max_x - raster_min_x) / pixel_width)),
    )
    y_end = min(
        shadow_dataset.RasterYSize,
        int(np.ceil((raster_max_y - min_y) / pixel_height)),
    )
    x_size = max(1, x_end - x_offset)
    y_size = max(1, y_end - y_offset)
    scale = min(1.0, raster_max_dimension / max(x_size, y_size))
    output_width = max(1, round(x_size * scale))
    output_height = max(1, round(y_size * scale))
    array = shadow_dataset.GetRasterBand(1).ReadAsArray(
        x_offset,
        y_offset,
        x_size,
        y_size,
        output_width,
        output_height,
    )
    shadow_dataset = None
    valid = array > -100
    shaded = valid & (array < 0.5)
    rgba = np.zeros((output_height, output_width, 4), dtype=np.uint8)
    rgba[..., 0] = 36
    rgba[..., 1] = 48
    rgba[..., 2] = 55
    rgba[..., 3] = np.where(shaded, 105, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG", compress_level=3)
    actual_extent = [
        raster_min_x + x_offset * pixel_width,
        raster_max_y - y_end * pixel_height,
        raster_min_x + x_end * pixel_width,
        raster_max_y - y_offset * pixel_height,
    ]
    return {
        "hour": hour,
        "bbox": [min_x, min_y, max_x, max_y],
        "buildings": buildings,
        "building_count": len(buildings),
        "building_raster_feature_count": matched_count,
        "building_match_count": matched_count,
        "building_truncated": matched_count > len(buildings),
        "building_render_mode": "all_footprints_raster",
        **building_raster,
        "shadow_extent": actual_extent,
        "shadow_width": output_width,
        "shadow_height": output_height,
        "shadow_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def read_gdb_geometry(path: Path) -> list[list[float]]:
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise RuntimeError("Cannot read official route export")
    layer = dataset.GetLayer(0)
    feature = layer.GetNextFeature()
    geometry = feature.GetGeometryRef()
    coordinates: list[list[float]] = []

    def append_points(value: Any) -> None:
        if value.GetPointCount() > 0:
            points = [
                [float(value.GetX(index)), float(value.GetY(index))]
                for index in range(value.GetPointCount())
            ]
            if coordinates and points and coordinates[-1] == points[0]:
                coordinates.extend(points[1:])
            else:
                coordinates.extend(points)
            return
        for index in range(value.GetGeometryCount()):
            append_points(value.GetGeometryRef(index))

    append_points(geometry)
    return coordinates


def export_derived(
    interface: RouteToolInterface,
    route_result: dict[str, Any],
    output_format: str,
    transient: bool,
) -> dict[str, Any]:
    if output_format in {"png", "pdf", "file_geodatabase"}:
        return interface.call(
            "export_route_map",
            {"route_result": route_result, "format": output_format},
        )
    official = interface.call(
        "export_route_map",
        {"route_result": route_result, "format": "file_geodatabase"},
    )
    source = Path(official["output_path"])
    coordinates = read_gdb_geometry(source)
    route_id = str(route_result["route_id"])
    if output_format == "geometry":
        shutil.rmtree(source)
        return {
            "type": "LineString",
            "crs": "EPSG:32650",
            "coordinates": coordinates,
        }
    if output_format == "geojson":
        path = interface.output_dir / f"route_{route_id}.geojson"
        feature = {
            "type": "Feature",
            "properties": {
                "route_id": route_id,
                "objective": route_result.get("objective"),
                "distance_m": route_result.get("distance_m"),
            },
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "crs": {"type": "name", "properties": {"name": "EPSG:32650"}},
        }
        path.write_text(json.dumps(feature, ensure_ascii=False), encoding="utf-8")
    elif output_format == "csv":
        path = interface.output_dir / f"route_{route_id}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["route_id", "sequence", "x", "y", "crs"])
            for index, (x, y) in enumerate(coordinates):
                writer.writerow([route_id, index, x, y, "EPSG:32650"])
    else:
        raise ValueError(f"Unsupported application export: {output_format}")
    shutil.rmtree(source)
    return {
        "format": output_format,
        "output_path": str(path.resolve()),
        "exists": path.exists(),
        "route_id": route_id,
        "transient": transient,
    }


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    interface = RouteToolInterface(ROOT)
    interface.output_dir = ROOT / "routing" / "outputs" / "application"
    interface.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"ready": True, "worker_id": uuid.uuid4().hex}), flush=True)
    for line in sys.stdin:
        request_id = ""
        try:
            message = json.loads(line)
            request_id = str(message["id"])
            operation = str(message["operation"])
            if operation == "tool":
                result = interface.call(str(message["tool"]), message["payload"])
            elif operation == "derived_export":
                result = export_derived(
                    interface,
                    message["route_result"],
                    str(message["format"]),
                    bool(message.get("transient", False)),
                )
            elif operation == "health":
                result = {
                    "graph_loaded": len(interface.engine.node_xy) > 0,
                    "costs_loaded": len(interface.engine.utci_mean) > 0,
                    "gazetteer_loaded": len(interface.aliases) > 0,
                    "tool_count": len(interface.schemas["tools"]),
                    "forbidden_turn_count": len(interface.engine.forbidden_turns),
                }
            elif operation == "visual_layers":
                result = build_visual_layers(
                    list(message["bbox"]),
                    int(message["hour"]),
                    int(message.get("max_buildings", 3500)),
                    int(message.get("raster_max_dimension", 1200)),
                    bool(message.get("include_vectors", True)),
                )
            elif operation == "shutdown":
                print(json.dumps({"id": request_id, "ok": True, "result": {}}), flush=True)
                return 0
            else:
                raise ValueError("Unknown worker operation")
            print(
                json.dumps(
                    {"id": request_id, "ok": True, "result": result},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "id": request_id,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
