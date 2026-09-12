from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "routing" / "data" / "gazetteer" / "built"
SOURCE = ROOT / "routing" / "data" / "gazetteer" / "source" / "jiangsu-latest.osm.pbf"
BOUNDARY = ROOT / "data" / "raw" / "Nanjing_center_boundary" / "Nanjing_center_UTM50N.shp"
os.environ["GDAL_PAM_PROXY_DIR"] = str(ROOT / "routing" / "data" / "gazetteer" / "gdal_proxy")

from osgeo import ogr

ogr.UseExceptions()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as handle, tqdm(
        total=total, desc="QC source hash", unit="B", unit_scale=True, dynamic_ncols=True
    ) as progress:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            progress.update(len(block))
    return digest.hexdigest()


def boundary_union() -> ogr.Geometry:
    dataset = ogr.Open(str(BOUNDARY), 0)
    layer = dataset.GetLayer(0)
    geometries = []
    for feature in tqdm(
        layer, total=layer.GetFeatureCount(), desc="QC boundary", dynamic_ncols=True
    ):
        geometry = feature.GetGeometryRef()
        if geometry is not None and not geometry.IsEmpty():
            geometries.append(geometry.Clone())
    dataset = None
    result = geometries[0]
    for geometry in tqdm(
        geometries[1:],
        total=max(0, len(geometries) - 1),
        desc="QC boundary union",
        dynamic_ncols=True,
    ):
        result = result.Union(geometry)
    return result


def main() -> int:
    summary = json.loads((DATA / "step29_gazetteer_summary.json").read_text(encoding="utf-8"))
    features = pd.read_csv(DATA / "gazetteer_features.csv")
    aliases = pd.read_csv(DATA / "gazetteer_aliases.csv")
    source_hash = sha256(SOURCE)
    boundary = boundary_union()

    dataset = ogr.Open(str(DATA / "nanjing_local_gazetteer.gpkg"), 0)
    layer = dataset.GetLayerByName("gazetteer_features")
    gpkg_count = layer.GetFeatureCount()
    epsg = layer.GetSpatialRef().GetAuthorityCode(None)
    layer_type = layer.GetGeomType()
    empty_geometries = 0
    outside_boundary = 0
    max_outside_distance_m = 0.0
    non_point_geometries = 0
    non_2d_geometries = 0
    for feature in tqdm(
        layer, total=gpkg_count, desc="QC GeoPackage", dynamic_ncols=True
    ):
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            empty_geometries += 1
            continue
        if ogr.GT_Flatten(geometry.GetGeometryType()) != ogr.wkbPoint:
            non_point_geometries += 1
        if geometry.GetCoordinateDimension() != 2:
            non_2d_geometries += 1
        if not boundary.Intersects(geometry):
            outside_boundary += 1
            max_outside_distance_m = max(max_outside_distance_m, boundary.Distance(geometry))
    dataset = None

    alias_key_duplicates = int(
        aliases.duplicated(["normalized_name", "feature_id", "name_field"]).sum()
    )
    alias_orphans = int((~aliases["feature_id"].isin(features["feature_id"])).sum())
    feature_id_duplicates = int(features["feature_id"].duplicated().sum())
    non_finite_coordinates = int(
        sum(
            not math.isfinite(float(value))
            for value in tqdm(
                features[["x", "y", "lon", "lat"]].to_numpy().ravel(),
                total=len(features) * 4,
                desc="QC coordinates",
                dynamic_ncols=True,
            )
        )
    )
    categories = set(features["category"].dropna().astype(str))
    expected_categories = {"road", "place", "poi", "transit", "named_feature"}
    query_names = ["新街口", "鼓楼", "夫子庙", "玄武湖", "南京站"]
    query_rows = [
        {
            "query": query,
            "exact_candidate_count": int((aliases["normalized_name"] == query).sum()),
            "pass": int((aliases["normalized_name"] == query).sum()) > 0,
        }
        for query in tqdm(
            query_names, total=len(query_names), desc="QC known names", dynamic_ncols=True
        )
    ]
    pd.DataFrame(query_rows).to_csv(
        DATA / "step29_gazetteer_query_smoke_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    checks = [
        ("source_size_matches", SOURCE.stat().st_size, SOURCE.stat().st_size == summary["source_size_bytes"]),
        ("source_sha256_matches", source_hash, source_hash == summary["source_sha256"]),
        ("feature_csv_count", len(features), len(features) == summary["feature_count"]),
        ("alias_csv_count", len(aliases), len(aliases) == summary["alias_count"]),
        ("gpkg_feature_count", gpkg_count, gpkg_count == summary["feature_count"]),
        ("gpkg_epsg", epsg, epsg == "32650"),
        ("gpkg_layer_is_point", ogr.GeometryTypeToName(layer_type), ogr.GT_Flatten(layer_type) == ogr.wkbPoint),
        ("empty_geometries", empty_geometries, empty_geometries == 0),
        ("non_point_geometries", non_point_geometries, non_point_geometries == 0),
        ("non_2d_geometries", non_2d_geometries, non_2d_geometries == 0),
        (
            "strict_boundary_misses_within_1um_tolerance",
            f"count={outside_boundary};max_distance_m={max_outside_distance_m:.12g}",
            max_outside_distance_m <= 1e-6,
        ),
        ("feature_id_duplicates", feature_id_duplicates, feature_id_duplicates == 0),
        ("alias_key_duplicates", alias_key_duplicates, alias_key_duplicates == 0),
        ("alias_orphans", alias_orphans, alias_orphans == 0),
        ("non_finite_coordinates", non_finite_coordinates, non_finite_coordinates == 0),
        ("expected_categories", ",".join(sorted(categories)), expected_categories.issubset(categories)),
        ("known_name_queries", sum(row["pass"] for row in query_rows), all(row["pass"] for row in query_rows)),
        ("failed_record_count", summary["failed_record_count"], summary["failed_record_count"] == 0),
    ]
    qc = pd.DataFrame(checks, columns=["check", "value", "pass"])
    qc.to_csv(
        DATA / "step29_gazetteer_independent_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    passed = bool(qc["pass"].all())
    result = {
        "step": "Step29_local_gazetteer_independent_qc",
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "PASS" if passed else "FAIL",
        "check_count": len(qc),
        "passed_check_count": int(qc["pass"].sum()),
        "feature_count": len(features),
        "alias_count": len(aliases),
        "source_sha256": source_hash,
        "ready_for_formal_step29_interface": passed,
    }
    (DATA / "step29_gazetteer_independent_qc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
