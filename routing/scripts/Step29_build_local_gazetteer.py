from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[2]
GDAL_PROXY = ROOT / "routing" / "data" / "gazetteer" / "gdal_proxy"
GDAL_PROXY.mkdir(parents=True, exist_ok=True)
os.environ["GDAL_PAM_PROXY_DIR"] = str(GDAL_PROXY)

from osgeo import gdal, ogr, osr
from tqdm import tqdm


CONFIG = ROOT / "routing" / "configs" / "gazetteer_source_draft.yaml"
OSMCONF = ROOT / "routing" / "configs" / "osmconf_gazetteer.ini"
BOUNDARY = ROOT / "data" / "raw" / "Nanjing_center_boundary" / "Nanjing_center_UTM50N.shp"
SOURCE = ROOT / "routing" / "data" / "gazetteer" / "source" / "jiangsu-latest.osm.pbf"
OUT_DIR = ROOT / "routing" / "data" / "gazetteer" / "built"
LOG_PATH = ROOT / "routing" / "logs" / "step29_gazetteer.log"

NAME_FIELDS = ("name", "name:zh", "name:en", "alt_name", "short_name", "official_name")
CLASS_FIELDS = (
    "highway", "place", "amenity", "shop", "tourism", "leisure", "office",
    "healthcare", "historic", "public_transport", "railway", "station",
)
LAYERS = ("points", "lines", "multilinestrings", "multipolygons", "other_relations")
HSTORE_PAIR = re.compile(r'"((?:\\.|[^"])*)"=>"((?:\\.|[^"])*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the approved offline Nanjing gazetteer.")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--boundary", type=Path, default=BOUNDARY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--approved-by-user", action="store_true")
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step29_gazetteer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as handle, tqdm(
        total=total, desc="Hashing source", unit="B", unit_scale=True, dynamic_ncols=True
    ) as progress:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            progress.update(len(block))
    return digest.hexdigest()


def configure_gdal() -> None:
    gdal.SetConfigOption("GDAL_PAM_PROXY_DIR", str(GDAL_PROXY))
    gdal.SetConfigOption("OSM_CONFIG_FILE", str(OSMCONF))
    gdal.SetConfigOption("OGR_INTERLEAVED_READING", "YES")
    gdal.UseExceptions()
    ogr.UseExceptions()


def axis_order(srs: osr.SpatialReference) -> osr.SpatialReference:
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def transform(source: osr.SpatialReference, target: osr.SpatialReference):
    return osr.CoordinateTransformation(axis_order(source.Clone()), axis_order(target.Clone()))


def load_boundary(path: Path) -> tuple[ogr.Geometry, osr.SpatialReference]:
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise RuntimeError(f"Cannot open boundary: {path}")
    layer = dataset.GetLayer(0)
    srs = axis_order(layer.GetSpatialRef().Clone())
    geometries = []
    total = layer.GetFeatureCount()
    for feature in tqdm(layer, total=total, desc="Reading boundary", dynamic_ncols=True):
        geometry = feature.GetGeometryRef()
        if geometry is not None and not geometry.IsEmpty():
            geometries.append(geometry.Clone())
    dataset = None
    if not geometries:
        raise RuntimeError("Boundary contains no valid geometry.")
    union = geometries[0]
    for geometry in tqdm(
        geometries[1:],
        total=max(0, len(geometries) - 1),
        desc="Merging boundary",
        dynamic_ncols=True,
    ):
        union = union.Union(geometry)
    if union is None or union.IsEmpty():
        raise RuntimeError("Boundary contains no valid geometry.")
    return union, srs


def parse_other_tags(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    return {
        key.replace(r"\"", '"').replace(r"\\", "\\"):
        item.replace(r"\"", '"').replace(r"\\", "\\")
        for key, item in HSTORE_PAIR.findall(value)
    }


def value(feature: ogr.Feature, name: str, tags: dict[str, str]) -> str:
    index = feature.GetFieldIndex(name)
    if index >= 0 and feature.IsFieldSetAndNotNull(index):
        result = str(feature.GetField(index)).strip()
        if result:
            return result
    return str(tags.get(name, "")).strip()


def classify(values: dict[str, str]) -> tuple[str, str]:
    if values.get("highway"):
        return "road", values["highway"]
    if values.get("place"):
        return "place", values["place"]
    if values.get("public_transport") or values.get("railway") or values.get("station"):
        return (
            "transit",
            values.get("public_transport") or values.get("railway") or values.get("station") or "",
        )
    for key in ("amenity", "shop", "tourism", "leisure", "office", "healthcare", "historic"):
        if values.get(key):
            return "poi", f"{key}:{values[key]}"
    return "named_feature", ""


def normalize_name(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def split_aliases(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;；|]", text) if item.strip()]


def representative_point(geometry: ogr.Geometry, boundary: ogr.Geometry) -> ogr.Geometry | None:
    clipped = geometry.Intersection(boundary)
    if clipped is None or clipped.IsEmpty():
        return None
    if ogr.GT_Flatten(clipped.GetGeometryType()) == ogr.wkbPoint:
        return clipped.Clone()
    point = clipped.PointOnSurface()
    if point is None or point.IsEmpty():
        point = clipped.Centroid()
    return point


def extract_records(
    source: Path,
    boundary: ogr.Geometry,
    boundary_srs: osr.SpatialReference,
    logger: logging.Logger,
) -> tuple[list[dict], list[dict]]:
    dataset = ogr.Open(str(source), 0)
    if dataset is None:
        raise RuntimeError(f"GDAL cannot open OSM PBF: {source}")
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    axis_order(wgs84)
    boundary_wgs84 = boundary.Clone()
    boundary_wgs84.Transform(transform(boundary_srs, wgs84))
    records, failures, seen = [], [], set()
    object_types = {
        "points": "node", "lines": "way", "multilinestrings": "way",
        "multipolygons": "relation_or_way", "other_relations": "relation",
    }
    for layer_name in tqdm(LAYERS, total=len(LAYERS), desc="OSM layers", dynamic_ncols=True):
        layer = dataset.GetLayerByName(layer_name)
        if layer is None:
            logger.warning("Layer unavailable: %s", layer_name)
            continue
        layer.SetSpatialFilter(boundary_wgs84)
        total = layer.GetFeatureCount(force=1)
        source_srs = layer.GetSpatialRef() or wgs84
        to_projected = transform(source_srs, boundary_srs)
        to_wgs84 = transform(boundary_srs, wgs84)
        for feature in tqdm(
            layer, total=total, desc=f"Extracting {layer_name}", leave=False, dynamic_ncols=True
        ):
            osm_id = str(feature.GetField("osm_id") or feature.GetFID())
            feature_id = f"{layer_name}:{osm_id}"
            if feature_id in seen:
                continue
            seen.add(feature_id)
            try:
                tags = parse_other_tags(feature.GetField("other_tags"))
                values = {
                    field: value(feature, field, tags)
                    for field in (*NAME_FIELDS, *CLASS_FIELDS)
                }
                names = [values[field] for field in NAME_FIELDS if values[field]]
                if not names:
                    continue
                geometry = feature.GetGeometryRef()
                if geometry is None or geometry.IsEmpty():
                    raise ValueError("empty_geometry")
                projected = geometry.Clone()
                projected.Transform(to_projected)
                point = representative_point(projected, boundary)
                if point is None:
                    continue
                lonlat = point.Clone()
                lonlat.Transform(to_wgs84)
                category, subtype = classify(values)
                record = {
                    "feature_id": feature_id,
                    "osm_type": object_types[layer_name],
                    "osm_id": osm_id,
                    "source_layer": layer_name,
                    "name": values["name"] or names[0],
                    "name_zh": values["name:zh"],
                    "name_en": values["name:en"],
                    "alt_name": values["alt_name"],
                    "short_name": values["short_name"],
                    "official_name": values["official_name"],
                    "category": category,
                    "subtype": subtype,
                    "ref": value(feature, "ref", tags),
                    "x": point.GetX(), "y": point.GetY(),
                    "lon": lonlat.GetX(), "lat": lonlat.GetY(),
                }
                if all(math.isfinite(float(record[key])) for key in ("x", "y", "lon", "lat")):
                    records.append(record)
                else:
                    raise ValueError("non_finite_coordinate")
            except Exception as exc:
                failures.append(
                    {"source_layer": layer_name, "osm_id": osm_id, "error_message": str(exc)}
                )
        layer.SetSpatialFilter(None)
    dataset = None
    return records, failures


def build_aliases(records: list[dict]) -> list[dict]:
    aliases = []
    for record in tqdm(records, total=len(records), desc="Building aliases", dynamic_ncols=True):
        for field in NAME_FIELDS:
            for alias in split_aliases(str(record.get(field.replace(":", "_"), ""))):
                normalized = normalize_name(alias)
                if normalized:
                    aliases.append(
                        {
                            "normalized_name": normalized,
                            "display_name": alias,
                            "name_field": field,
                            "feature_id": record["feature_id"],
                            "category": record["category"],
                            "subtype": record["subtype"],
                            "x": record["x"], "y": record["y"],
                            "lon": record["lon"], "lat": record["lat"],
                        }
                    )
    unique = {}
    for alias in tqdm(aliases, total=len(aliases), desc="Deduplicating aliases", dynamic_ncols=True):
        unique[(alias["normalized_name"], alias["feature_id"], alias["name_field"])] = alias
    return list(unique.values())


def write_gpkg(records: list[dict], path: Path, overwrite: bool) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {path}")
        driver.DeleteDataSource(str(path))
    dataset = driver.CreateDataSource(str(path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32650)
    axis_order(srs)
    layer = dataset.CreateLayer("gazetteer_features", srs=srs, geom_type=ogr.wkbPoint)
    string_fields = (
        "feature_id", "osm_type", "osm_id", "source_layer", "name", "name_zh",
        "name_en", "alt_name", "short_name", "official_name", "category", "subtype", "ref",
    )
    for field in string_fields:
        definition = ogr.FieldDefn(field, ogr.OFTString)
        definition.SetWidth(254)
        layer.CreateField(definition)
    for field in ("x", "y", "lon", "lat"):
        layer.CreateField(ogr.FieldDefn(field, ogr.OFTReal))
    definition = layer.GetLayerDefn()
    dataset.StartTransaction()
    for record in tqdm(records, total=len(records), desc="Writing GeoPackage", dynamic_ncols=True):
        feature = ogr.Feature(definition)
        for key in (*string_fields, "x", "y", "lon", "lat"):
            feature.SetField(key, record[key])
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(float(record["x"]), float(record["y"]))
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
    dataset.CommitTransaction()
    dataset = None


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields or []).writeheader()


def main() -> int:
    args = parse_args()
    logger = setup_logging()
    started = datetime.now(timezone.utc).astimezone()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed_path = OUT_DIR / "step29_gazetteer_failed_records.csv"
    summary_path = OUT_DIR / "step29_gazetteer_summary.json"
    logger.info("Step29 gazetteer build started: %s", started.isoformat())
    try:
        if not args.approved_by_user:
            raise PermissionError("Formal build requires --approved-by-user.")
        if not args.source.exists():
            raise FileNotFoundError(f"Approved source PBF is missing: {args.source}")
        if args.source.stat().st_size < 10_000_000:
            raise ValueError("Source PBF is unexpectedly small; refusing formal build.")
        if not args.boundary.exists():
            raise FileNotFoundError(f"Boundary is missing: {args.boundary}")
        configure_gdal()
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        source_hash = sha256_file(args.source)
        boundary, boundary_srs = load_boundary(args.boundary)
        records, failures = extract_records(args.source, boundary, boundary_srs, logger)
        if not records:
            raise RuntimeError("No named OSM records intersect the approved boundary.")
        aliases = build_aliases(records)
        if not aliases:
            raise RuntimeError("No searchable aliases were generated.")
        write_gpkg(records, OUT_DIR / "nanjing_local_gazetteer.gpkg", args.overwrite)
        write_csv(OUT_DIR / "gazetteer_features.csv", records)
        write_csv(OUT_DIR / "gazetteer_aliases.csv", aliases)
        write_csv(failed_path, failures, ["source_layer", "osm_id", "error_message"])
        category_counts = Counter(record["category"] for record in records)
        qc_rows = [
            {"check": "source_size_gt_10mb", "value": args.source.stat().st_size, "pass": True},
            {"check": "source_sha256_length", "value": len(source_hash), "pass": len(source_hash) == 64},
            {"check": "feature_count_positive", "value": len(records), "pass": len(records) > 0},
            {"check": "alias_count_positive", "value": len(aliases), "pass": len(aliases) > 0},
            {"check": "normalized_names_empty", "value": sum(not x["normalized_name"] for x in aliases),
             "pass": all(x["normalized_name"] for x in aliases)},
            {"check": "failed_record_count", "value": len(failures), "pass": len(failures) == 0},
        ]
        write_csv(OUT_DIR / "step29_gazetteer_qc.csv", qc_rows)
        success = all(bool(row["pass"]) for row in qc_rows)
        summary = {
            "step": "Step29_local_gazetteer",
            "status": "PASS" if success else "FAIL",
            "started_at": started.isoformat(),
            "ended_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "source_path": str(args.source.resolve()),
            "source_url": config["source"]["download_url"],
            "source_size_bytes": args.source.stat().st_size,
            "source_sha256": source_hash,
            "boundary_path": str(args.boundary.resolve()),
            "feature_count": len(records),
            "alias_count": len(aliases),
            "category_counts": dict(sorted(category_counts.items())),
            "failed_record_count": len(failures),
            "attribution": config["source"]["required_attribution"],
            "license": config["source"]["license"],
            "ready_for_formal_step29_interface": success,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Build ended: success=%s features=%d aliases=%d failures=%d",
                    success, len(records), len(aliases), len(failures))
        return 0 if success else 2
    except Exception as exc:
        logger.exception("Step29 gazetteer build blocked: %s", exc)
        write_csv(failed_path, [{"source_layer": "", "osm_id": "", "error_message": str(exc)}])
        summary_path.write_text(
            json.dumps(
                {
                    "step": "Step29_local_gazetteer",
                    "status": "BLOCKED",
                    "started_at": started.isoformat(),
                    "ended_at": datetime.now(timezone.utc).astimezone().isoformat(),
                    "error": str(exc),
                    "ready_for_formal_step29_interface": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
