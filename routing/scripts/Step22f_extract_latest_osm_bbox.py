from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PBF = ROOT / "data/raw/osm_geofabrik/2026-08-01/jiangsu-260801.osm.pbf"
DEFAULT_BOUNDARY = ROOT / "data/raw/Nanjing_center_boundary/Nanjing_center_UTM50N.shp"
DEFAULT_OUTPUT = ROOT / "data/raw/osm_geofabrik/2026-08-01/nanjing_center_bbox_roads_20260801.shp"
DEFAULT_WORK = ROOT / "routing/data/osm_latest_rect"
DEFAULT_LOG = ROOT / "routing/logs/step22f_extract_latest_osm_bbox.log"
SCRIPT_VERSION = "1.0.0"

# Match the road-family coverage of the former Geofabrik roads Shapefile.
# Non-routable or unfinished objects (construction/proposed/platform/etc.) are excluded.
ROUTABLE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "residential",
    "unclassified", "service", "living_street", "pedestrian", "track", "bridleway",
    "cycleway", "footway", "path", "steps", "road",
}
TAG_PATTERN = re.compile(r'"((?:\\.|[^"])*)"=>"((?:\\.|[^"])*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract current OSM roads within the center-boundary bounding rectangle.")
    parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--expected-md5", default="5ffa251202e362290f137f5ced2d9691")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step22f_osm_bbox")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tags_dict(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    return {
        key.replace(r'\"', '"').replace(r"\\", "\\"): val.replace(r'\"', '"').replace(r"\\", "\\")
        for key, val in TAG_PATTERN.findall(value)
    }


def axis_order(srs: Any) -> Any:
    from osgeo import osr
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def load_bbox(boundary_path: Path) -> tuple[Any, Any, dict[str, float]]:
    from osgeo import ogr, osr
    dataset = ogr.Open(str(boundary_path), 0)
    if dataset is None:
        raise FileNotFoundError(boundary_path)
    layer = dataset.GetLayer(0)
    source_srs = axis_order(layer.GetSpatialRef().Clone())
    extent = layer.GetExtent()  # minx, maxx, miny, maxy
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in (
        (extent[0], extent[2]), (extent[1], extent[2]), (extent[1], extent[3]),
        (extent[0], extent[3]), (extent[0], extent[2]),
    ):
        ring.AddPoint_2D(x, y)
    rectangle_projected = ogr.Geometry(ogr.wkbPolygon)
    rectangle_projected.AddGeometry(ring)
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    axis_order(wgs84)
    rectangle_wgs84 = rectangle_projected.Clone()
    rectangle_wgs84.Transform(osr.CoordinateTransformation(source_srs, wgs84))
    envelope = rectangle_wgs84.GetEnvelope()
    return rectangle_wgs84, wgs84, {
        "west": float(envelope[0]), "east": float(envelope[1]),
        "south": float(envelope[2]), "north": float(envelope[3]),
        "utm_min_x": float(extent[0]), "utm_max_x": float(extent[1]),
        "utm_min_y": float(extent[2]), "utm_max_y": float(extent[3]),
    }


def line_parts(geometry: Any) -> Iterable[Any]:
    from osgeo import ogr
    flat = ogr.GT_Flatten(geometry.GetGeometryType())
    if flat in {ogr.wkbLineString, ogr.wkbLinearRing}:
        if geometry.GetPointCount() >= 2 and geometry.Length() > 0:
            yield geometry.Clone()
    elif flat in {ogr.wkbMultiLineString, ogr.wkbGeometryCollection}:
        for index in range(geometry.GetGeometryCount()):
            yield from line_parts(geometry.GetGeometryRef(index))


def remove_shapefile(path: Path) -> None:
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()


def create_output(path: Path, srs: Any) -> tuple[Any, Any]:
    from osgeo import ogr
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(str(path))
    layer = dataset.CreateLayer(path.stem, srs=srs, geom_type=ogr.wkbLineString, options=["ENCODING=UTF-8"])
    for name, kind, width in (
        ("osm_id", ogr.OFTString, 48), ("fclass", ogr.OFTString, 32),
        ("name", ogr.OFTString, 160), ("ref", ogr.OFTString, 80),
        ("oneway", ogr.OFTString, 12), ("maxspeed", ogr.OFTString, 32),
        ("layer", ogr.OFTInteger, 0), ("bridge", ogr.OFTString, 12),
        ("tunnel", ogr.OFTString, 12), ("access", ogr.OFTString, 32),
        ("foot", ogr.OFTString, 32), ("bicycle", ogr.OFTString, 32),
        ("surface", ogr.OFTString, 40), ("tracktype", ogr.OFTString, 16),
        ("source_dt", ogr.OFTString, 12),
    ):
        field = ogr.FieldDefn(name, kind)
        if width:
            field.SetWidth(width)
        layer.CreateField(field)
    return dataset, layer


def main() -> int:
    args = parse_args()
    logger = setup_logger(DEFAULT_LOG)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.perf_counter()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    failed_path = args.work_dir / "step22f_failed_records.csv"
    summary_path = args.work_dir / "step22f_extract_summary.json"
    class_path = args.work_dir / "step22f_highway_class_counts.csv"
    failed: list[dict[str, str]] = []
    try:
        from osgeo import gdal, ogr
        gdal.UseExceptions()
        ogr.UseExceptions()
        os.environ["OGR_INTERLEAVED_READING"] = "YES"
        gdal.SetConfigOption("OGR_INTERLEAVED_READING", "YES")
        gdal.SetConfigOption("GDAL_PAM_PROXY_DIR", str(args.work_dir / "gdal_proxy"))
        (args.work_dir / "gdal_proxy").mkdir(parents=True, exist_ok=True)
        if not args.pbf.exists() or not args.boundary.exists():
            raise FileNotFoundError("PBF or boundary input is missing")
        source_md5 = md5(args.pbf)
        if args.expected_md5 and source_md5.lower() != args.expected_md5.lower():
            raise ValueError(f"PBF MD5 mismatch: {source_md5}")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}; use --overwrite")
        if args.overwrite:
            remove_shapefile(args.output)
        rectangle, wgs84, bbox = load_bbox(args.boundary)
        dataset = gdal.OpenEx(str(args.pbf), gdal.OF_VECTOR, open_options=["INTERLEAVED_READING=YES"])
        if dataset is None:
            raise RuntimeError(f"Cannot open PBF: {args.pbf}")
        sql = "SELECT osm_id,name,highway,z_order,other_tags FROM lines WHERE highway IS NOT NULL"
        source = dataset.ExecuteSQL(sql, spatialFilter=rectangle, dialect="OGRSQL")
        if source is None:
            raise RuntimeError("OSM lines query failed")
        output_ds, output_layer = create_output(args.output, wgs84)
        output_defn = output_layer.GetLayerDefn()
        counts: Counter[str] = Counter()
        excluded: Counter[str] = Counter()
        selected = written = 0
        # Wrap the OGR layer in a generator. Calling len()/GetFeatureCount() on
        # an interleaved OSM stream consumes the query cursor in some GDAL builds.
        feature_stream = (feature for feature in source)
        for feature in tqdm(feature_stream, desc="Extracting latest OSM roads", unit="way", dynamic_ncols=True):
            osm_id = str(feature.GetField("osm_id") or feature.GetFID())
            try:
                highway = str(feature.GetField("highway") or "").lower()
                if highway not in ROUTABLE_HIGHWAYS:
                    excluded[highway or "missing"] += 1
                    continue
                geometry = feature.GetGeometryRef()
                if geometry is None or geometry.IsEmpty():
                    raise ValueError("empty_geometry")
                clipped = geometry.Intersection(rectangle)
                if clipped is None or clipped.IsEmpty():
                    continue
                selected += 1
                tags = tags_dict(feature.GetField("other_tags"))
                tracktype = tags.get("tracktype", "")
                fclass = tracktype if highway == "track" and tracktype.startswith("grade") else highway
                if fclass == "road":
                    fclass = "unknown"
                parts = list(line_parts(clipped))
                for part_index, part in enumerate(parts, start=1):
                    out = ogr.Feature(output_defn)
                    out.SetGeometry(part)
                    out.SetField("osm_id", osm_id if len(parts) == 1 else f"{osm_id}_p{part_index}")
                    values = {
                        "fclass": fclass, "name": feature.GetField("name") or "",
                        "ref": tags.get("ref", ""), "oneway": tags.get("oneway", ""),
                        "maxspeed": tags.get("maxspeed", ""), "layer": int(tags.get("layer", "0") or 0),
                        "bridge": tags.get("bridge", "F"), "tunnel": tags.get("tunnel", "F"),
                        "access": tags.get("access", ""), "foot": tags.get("foot", ""),
                        "bicycle": tags.get("bicycle", ""), "surface": tags.get("surface", ""),
                        "tracktype": tracktype, "source_dt": "2026-08-01",
                    }
                    for key, value in values.items():
                        out.SetField(key, value)
                    output_layer.CreateFeature(out)
                    written += 1
                    counts[fclass] += 1
            except Exception as error:
                failed.append({"osm_id": osm_id, "error_message": str(error)})
        dataset.ReleaseResultSet(source)
        output_ds = None
        dataset = None
        if written == 0:
            raise RuntimeError("OSM extraction produced zero road features")
        with class_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["fclass", "feature_count"])
            writer.writeheader()
            for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                writer.writerow({"fclass": key, "feature_count": value})
        with failed_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["osm_id", "error_message"])
            writer.writeheader(); writer.writerows(failed)
        summary = {
            "step": "22f", "script_version": SCRIPT_VERSION,
            "started_at": started_at, "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - started,
            "source_pbf": str(args.pbf.resolve()), "source_md5": source_md5,
            "source_osm_data_timestamp_utc": "2026-08-01T20:21:21Z",
            "boundary": str(args.boundary.resolve()), "scope_rule": "axis_aligned_bbox_of_center_boundary",
            "bbox_wgs84": bbox, "output": str(args.output.resolve()),
            "selected_osm_way_count": selected, "written_feature_count": written,
            "failed_count": len(failed), "excluded_highway_counts": dict(sorted(excluded.items())),
            "highway_class_counts": dict(sorted(counts.items())), "success": written > 0 and not failed,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Completed: selected=%s written=%s failed=%s output=%s", selected, written, len(failed), args.output)
        return 0
    except Exception as error:
        logger.error("Failed: %s\n%s", error, traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
