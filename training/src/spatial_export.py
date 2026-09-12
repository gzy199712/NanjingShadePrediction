"""Spatial-source validation and Phase F GIS field definitions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


GIS_FIELD_MAPPING: list[dict[str, str]] = [
    {
        "full_field": "point_id",
        "shapefile_field": "POINT_ID",
        "dtype": "text",
        "unit": "",
        "description": "Street-view point identifier",
    },
    {
        "full_field": "hour",
        "shapefile_field": "HOUR",
        "dtype": "short",
        "unit": "hour",
        "description": "Local hour in Asia/Shanghai",
    },
    {
        "full_field": "shade_pred_mean",
        "shapefile_field": "SHD_MEAN",
        "dtype": "double",
        "unit": "proportion",
        "description": "Five-seed mean shade prediction",
    },
    {
        "full_field": "shade_pred_std",
        "shapefile_field": "SHD_STD",
        "dtype": "double",
        "unit": "proportion",
        "description": "Five-seed shade predictive uncertainty",
    },
    {
        "full_field": "tmrt_pred_mean",
        "shapefile_field": "TMR_MEAN",
        "dtype": "double",
        "unit": "degree_Celsius",
        "description": "Five-seed mean Tmrt prediction",
    },
    {
        "full_field": "tmrt_pred_std",
        "shapefile_field": "TMR_STD",
        "dtype": "double",
        "unit": "degree_Celsius",
        "description": "Five-seed Tmrt predictive uncertainty",
    },
    {
        "full_field": "utci_pred_mean",
        "shapefile_field": "UTC_MEAN",
        "dtype": "double",
        "unit": "degree_Celsius",
        "description": "Five-seed mean UTCI post-processing result",
    },
    {
        "full_field": "utci_pred_std",
        "shapefile_field": "UTC_STD",
        "dtype": "double",
        "unit": "degree_Celsius",
        "description": "Five-seed UTCI predictive uncertainty",
    },
    {
        "full_field": "solar_altitude",
        "shapefile_field": "SOL_ALT",
        "dtype": "double",
        "unit": "degree",
        "description": "Solar altitude",
    },
    {
        "full_field": "solar_azimuth",
        "shapefile_field": "SOL_AZ",
        "dtype": "double",
        "unit": "degree",
        "description": "Solar azimuth",
    },
    {
        "full_field": "air_temperature",
        "shapefile_field": "AIR_TEMP",
        "dtype": "double",
        "unit": "degree_Celsius",
        "description": "Air temperature",
    },
    {
        "full_field": "relative_humidity",
        "shapefile_field": "RH",
        "dtype": "double",
        "unit": "percent",
        "description": "Relative humidity",
    },
    {
        "full_field": "wind_speed",
        "shapefile_field": "WIND",
        "dtype": "double",
        "unit": "m/s",
        "description": "Wind speed",
    },
    {
        "full_field": "dominant_attention_azimuth",
        "shapefile_field": "ATT_AZ",
        "dtype": "short",
        "unit": "degree",
        "description": "Dominant ensemble-attention azimuth",
    },
    {
        "full_field": "dominant_attention_weight",
        "shapefile_field": "ATT_WGT",
        "dtype": "double",
        "unit": "weight",
        "description": "Dominant ensemble-attention weight",
    },
    {
        "full_field": "dataset_split",
        "shapefile_field": "SPLIT",
        "dtype": "text",
        "unit": "",
        "description": "Original spatial split metadata",
    },
]


def gis_field_mapping() -> list[dict[str, str]]:
    return [dict(row) for row in GIS_FIELD_MAPPING]


def probe_spatial_source(
    *,
    path: Path,
    expected_point_ids: list[str],
    external_python: Path,
    gdal_proxy_dir: Path,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["GDAL_PAM_PROXY_DIR"] = str(gdal_proxy_dir)
    command = [
        str(external_python),
        str(Path(__file__).resolve()),
        "--probe",
        str(path),
    ]
    completed = subprocess.run(
        command,
        input=json.dumps({"expected_point_ids": expected_point_ids}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Spatial probe failed with arcpy35 GDAL runtime: "
            + completed.stderr.strip()
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Spatial probe returned invalid JSON: {completed.stdout[:500]}"
        ) from error
    result["external_stderr"] = completed.stderr.strip()
    result["external_python"] = str(external_python)
    return result


def export_spatial_outputs(
    *,
    source_points: Path,
    prediction_csv: Path,
    gis_dir: Path,
    external_python: Path,
    gdal_proxy_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Create formal FileGDB and hourly Shapefiles in the ArcGIS/GDAL runtime."""
    environment = os.environ.copy()
    environment["GDAL_PAM_PROXY_DIR"] = str(gdal_proxy_dir)
    request = {
        "source_points": str(source_points),
        "prediction_csv": str(prediction_csv),
        "gis_dir": str(gis_dir),
        "overwrite": overwrite,
        "field_mapping": gis_field_mapping(),
    }
    completed = subprocess.run(
        [
            str(external_python),
            str(Path(__file__).resolve()),
            "--export-request",
        ],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Spatial export failed with arcpy35 GDAL runtime:\n"
            + completed.stderr[-4000:]
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Spatial export returned invalid JSON: {completed.stdout[:1000]}"
        ) from error
    result["external_stderr_tail"] = completed.stderr[-4000:]
    return result


def extract_spatial_coordinates(
    *,
    source_points: Path,
    external_python: Path,
    gdal_proxy_dir: Path,
    output_csv: Path,
) -> dict[str, Any]:
    """Export point-geometry UTM coordinates and WGS84 copies without mutation."""
    environment = os.environ.copy()
    environment["GDAL_PAM_PROXY_DIR"] = str(gdal_proxy_dir)
    completed = subprocess.run(
        [
            str(external_python),
            str(Path(__file__).resolve()),
            "--coordinates",
            str(source_points),
            "--coordinates-output",
            str(output_csv),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Coordinate export failed:\n" + completed.stderr[-4000:]
        )
    return json.loads(completed.stdout)


def _probe(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    from osgeo import ogr, osr
    from tqdm import tqdm

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise FileNotFoundError(path)
    layer = dataset.GetLayer(0)
    definition = layer.GetLayerDefn()
    fields = [
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    ]
    id_field = next(
        (field for field in fields if field.lower() in {"id", "point_id"}),
        None,
    )
    if id_field is None:
        raise RuntimeError(f"No point identifier field in {fields}")
    x_field = next((field for field in fields if field.lower() == "x"), None)
    y_field = next((field for field in fields if field.lower() == "y"), None)
    source = layer.GetSpatialRef()
    if source is None:
        raise RuntimeError("Spatial source has no coordinate reference system")
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    point_ids: list[str] = []
    null_geometry = 0
    nonpoint_geometry = 0
    longitude_values: list[float] = []
    latitude_values: list[float] = []
    utm_x_values: list[float] = []
    utm_y_values: list[float] = []
    attribute_longitude: list[float] = []
    attribute_latitude: list[float] = []
    longitude_errors: list[float] = []
    latitude_errors: list[float] = []
    layer.ResetReading()
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Checking point geometries",
        unit="point",
        dynamic_ncols=True,
        file=sys.stderr,
    ):
        point_ids.append(str(feature.GetField(id_field)))
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            null_geometry += 1
            continue
        flatten = getattr(ogr, "GT_Flatten", None)
        geometry_type = (
            flatten(geometry.GetGeometryType())
            if flatten is not None
            else geometry.GetGeometryType() & (~ogr.wkb25DBit)
        )
        if geometry_type != ogr.wkbPoint:
            nonpoint_geometry += 1
            continue
        utm_x = float(geometry.GetX())
        utm_y = float(geometry.GetY())
        transformed = geometry.Clone()
        transformed.Transform(transform)
        longitude = float(transformed.GetX())
        latitude = float(transformed.GetY())
        utm_x_values.append(utm_x)
        utm_y_values.append(utm_y)
        longitude_values.append(longitude)
        latitude_values.append(latitude)
        if x_field and y_field:
            x_value = float(feature.GetField(x_field))
            y_value = float(feature.GetField(y_field))
            attribute_longitude.append(x_value)
            attribute_latitude.append(y_value)
            longitude_errors.append(abs(longitude - x_value))
            latitude_errors.append(abs(latitude - y_value))
    expected = set(str(value) for value in expected_ids)
    actual = set(point_ids)
    duplicate_count = len(point_ids) - len(actual)
    authority = source.GetAuthorityCode(None) or source.GetAuthorityCode("PROJCS")
    return {
        "path": str(path),
        "record_count": int(layer.GetFeatureCount()),
        "point_id_field": id_field,
        "field_names": fields,
        "unique_point_ids": len(actual),
        "duplicate_point_ids": duplicate_count,
        "missing_expected_count": len(expected - actual),
        "extra_point_count": len(actual - expected),
        "missing_expected_sample": sorted(expected - actual)[:20],
        "extra_point_sample": sorted(actual - expected)[:20],
        "null_geometry_count": null_geometry,
        "nonpoint_geometry_count": nonpoint_geometry,
        "spatial_reference_name": source.GetName(),
        "spatial_reference_wkid": int(authority) if authority else None,
        "spatial_reference_wkt": source.ExportToWkt(),
        "utm_x_min": min(utm_x_values),
        "utm_x_max": max(utm_x_values),
        "utm_y_min": min(utm_y_values),
        "utm_y_max": max(utm_y_values),
        "longitude_min": min(longitude_values),
        "longitude_max": max(longitude_values),
        "latitude_min": min(latitude_values),
        "latitude_max": max(latitude_values),
        "attribute_xy_present": bool(x_field and y_field),
        "attribute_longitude_min": (
            min(attribute_longitude) if attribute_longitude else None
        ),
        "attribute_longitude_max": (
            max(attribute_longitude) if attribute_longitude else None
        ),
        "attribute_latitude_min": (
            min(attribute_latitude) if attribute_latitude else None
        ),
        "attribute_latitude_max": (
            max(attribute_latitude) if attribute_latitude else None
        ),
        "max_longitude_attribute_difference": (
            max(longitude_errors) if longitude_errors else None
        ),
        "max_latitude_attribute_difference": (
            max(latitude_errors) if latitude_errors else None
        ),
    }


def _ogr_field_type(series: Any) -> tuple[int, int | None, int | None]:
    from osgeo import ogr
    import pandas as pd

    if pd.api.types.is_integer_dtype(series.dtype):
        return ogr.OFTInteger, None, None
    if pd.api.types.is_numeric_dtype(series.dtype):
        return ogr.OFTReal, 18, 8
    return ogr.OFTString, 64, None


def _set_feature_value(feature: Any, name: str, value: Any) -> None:
    import pandas as pd

    if pd.isna(value):
        return
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        value = int(value)
    feature.SetField(name, value)


def _load_geometries(source_points: Path) -> tuple[dict[str, bytes], str]:
    from osgeo import ogr
    from tqdm import tqdm

    source_ds = ogr.Open(str(source_points), 0)
    if source_ds is None:
        raise RuntimeError(f"Cannot open spatial source: {source_points}")
    layer = source_ds.GetLayer(0)
    definition = layer.GetLayerDefn()
    fields = [
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    ]
    id_field = next(
        (field for field in fields if field.lower() in {"id", "point_id"}),
        None,
    )
    if id_field is None:
        raise RuntimeError("Spatial source has no point ID field")
    geometries: dict[str, bytes] = {}
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Loading formal point geometries",
        unit="point",
        dynamic_ncols=True,
        file=sys.stderr,
    ):
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue
        geometries[str(feature.GetField(id_field))] = bytes(geometry.ExportToWkb())
    wkt = layer.GetSpatialRef().ExportToWkt()
    source_ds = None
    return geometries, wkt


def _create_layer(
    *,
    driver_name: str,
    dataset_path: Path,
    layer_name: str,
    frame: Any,
    geometries: dict[str, bytes],
    spatial_wkt: str,
    field_names: list[tuple[str, str]],
    recreate_dataset: bool,
) -> int:
    from osgeo import ogr, osr
    from tqdm import tqdm

    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"OGR driver unavailable: {driver_name}")
    if recreate_dataset:
        if dataset_path.exists():
            if dataset_path.is_dir():
                shutil.rmtree(dataset_path)
            else:
                driver.DeleteDataSource(str(dataset_path))
        dataset = driver.CreateDataSource(str(dataset_path))
    else:
        dataset = ogr.Open(str(dataset_path), 1)
    if dataset is None:
        raise RuntimeError(f"Cannot create/open output dataset: {dataset_path}")
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(spatial_wkt)
    existing = dataset.GetLayerByName(layer_name)
    if existing is not None:
        dataset.DeleteLayer(layer_name)
    layer = dataset.CreateLayer(
        layer_name, spatial_reference, geom_type=ogr.wkbPoint
    )
    output_names: list[str] = []
    for source_name, output_name in field_names:
        field_type, width, precision = _ogr_field_type(frame[source_name])
        field = ogr.FieldDefn(output_name, field_type)
        if width:
            field.SetWidth(width)
        if precision:
            field.SetPrecision(precision)
        if layer.CreateField(field) != 0:
            raise RuntimeError(f"Cannot create field {output_name}")
        output_names.append(output_name)
    definition = layer.GetLayerDefn()
    layer.StartTransaction()
    written = 0
    try:
        for values in tqdm(
            frame.itertuples(index=False, name=None),
            total=len(frame),
            desc=f"Writing {layer_name}",
            unit="feature",
            dynamic_ncols=True,
            file=sys.stderr,
        ):
            row = dict(zip(frame.columns, values, strict=True))
            geometry_wkb = geometries.get(str(row["point_id"]))
            if geometry_wkb is None:
                raise RuntimeError(f"Missing geometry for point {row['point_id']}")
            feature = ogr.Feature(definition)
            feature.SetGeometry(ogr.CreateGeometryFromWkb(geometry_wkb))
            for (source_name, _), output_name in zip(
                field_names, output_names, strict=True
            ):
                _set_feature_value(feature, output_name, row[source_name])
            if layer.CreateFeature(feature) != 0:
                raise RuntimeError(
                    f"Cannot write {layer_name} feature {row.get('sample_id')}"
                )
            written += 1
        layer.CommitTransaction()
    except Exception:
        layer.RollbackTransaction()
        raise
    dataset.FlushCache()
    dataset = None
    return written


def _export(request: dict[str, Any]) -> dict[str, Any]:
    from osgeo import ogr
    import pandas as pd
    from tqdm import tqdm

    ogr.UseExceptions()
    source_points = Path(request["source_points"])
    prediction_csv = Path(request["prediction_csv"])
    gis_dir = Path(request["gis_dir"])
    overwrite = bool(request["overwrite"])
    gis_dir.mkdir(parents=True, exist_ok=True)
    gdb_path = gis_dir / "full_predictions.gdb"
    shp_dir = gis_dir / "shp_hourly"
    if (gdb_path.exists() or shp_dir.exists()) and not overwrite:
        raise FileExistsError("GIS outputs exist; use --overwrite")
    if shp_dir.exists():
        shutil.rmtree(shp_dir)
    shp_dir.mkdir(parents=True)
    frame = pd.read_csv(prediction_csv, low_memory=False)
    frame["point_id"] = frame["point_id"].astype(str)
    geometries, spatial_wkt = _load_geometries(source_points)
    all_fields = [
        (name, name[:64])
        for name in frame.columns
        if name not in {"utm_x", "utm_y", "longitude", "latitude"}
    ]
    long_count = _create_layer(
        driver_name="OpenFileGDB",
        dataset_path=gdb_path,
        layer_name="AllPredictions_20240729",
        frame=frame,
        geometries=geometries,
        spatial_wkt=spatial_wkt,
        field_names=all_fields,
        recreate_dataset=True,
    )
    mapping = [
        (row["full_field"], row["shapefile_field"])
        for row in request["field_mapping"]
    ]
    hourly_counts: dict[str, int] = {}
    shapefile_counts: dict[str, int] = {}
    for hour in tqdm(
        range(6, 19),
        desc="Exporting hourly GIS layers",
        unit="hour",
        dynamic_ncols=True,
        file=sys.stderr,
    ):
        hour_frame = frame.loc[frame["hour"].astype(int) == hour].copy()
        name = f"Pred_{hour:02d}00"
        hourly_counts[name] = _create_layer(
            driver_name="OpenFileGDB",
            dataset_path=gdb_path,
            layer_name=name,
            frame=hour_frame,
            geometries=geometries,
            spatial_wkt=spatial_wkt,
            field_names=mapping,
            recreate_dataset=False,
        )
        shp_path = shp_dir / f"{name}.shp"
        shapefile_counts[name] = _create_layer(
            driver_name="ESRI Shapefile",
            dataset_path=shp_path,
            layer_name=name,
            frame=hour_frame,
            geometries=geometries,
            spatial_wkt=spatial_wkt,
            field_names=mapping,
            recreate_dataset=True,
        )
    with (gis_dir / "gis_field_dictionary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(request["field_mapping"][0])
        )
        writer.writeheader()
        writer.writerows(request["field_mapping"])
    return {
        "gdb_path": str(gdb_path),
        "long_feature_class_count": long_count,
        "hourly_feature_class_counts": hourly_counts,
        "shapefile_counts": shapefile_counts,
        "shapefile_count": len(shapefile_counts),
        "spatial_wkid": 32650,
    }


def _coordinates(source_path: Path, output_path: Path) -> dict[str, Any]:
    from osgeo import ogr, osr
    from tqdm import tqdm

    ogr.UseExceptions()
    dataset = ogr.Open(str(source_path), 0)
    if dataset is None:
        raise RuntimeError(f"Cannot open {source_path}")
    layer = dataset.GetLayer(0)
    definition = layer.GetLayerDefn()
    fields = [
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    ]
    id_field = next(
        (field for field in fields if field.lower() in {"id", "point_id"}),
        None,
    )
    if id_field is None:
        raise RuntimeError("No point ID field")
    source = layer.GetSpatialRef()
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["point_id", "longitude", "latitude", "utm_x", "utm_y"],
        )
        writer.writeheader()
        count = 0
        for feature in tqdm(
            layer,
            total=layer.GetFeatureCount(),
            desc="Exporting geometry coordinates",
            unit="point",
            dynamic_ncols=True,
            file=sys.stderr,
        ):
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                continue
            transformed = geometry.Clone()
            transformed.Transform(transform)
            writer.writerow(
                {
                    "point_id": str(feature.GetField(id_field)),
                    "longitude": float(transformed.GetX()),
                    "latitude": float(transformed.GetY()),
                    "utm_x": float(geometry.GetX()),
                    "utm_y": float(geometry.GetY()),
                }
            )
            count += 1
    return {"output_csv": str(output_path), "record_count": count}


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--export-request", action="store_true")
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--coordinates-output", type=Path)
    args = parser.parse_args()
    if args.export_request:
        request = json.loads(sys.stdin.read())
        print(json.dumps(_export(request), ensure_ascii=False))
        return 0
    if args.coordinates is not None:
        if args.coordinates_output is None:
            parser.error("--coordinates-output is required")
        print(
            json.dumps(
                _coordinates(args.coordinates, args.coordinates_output),
                ensure_ascii=False,
            )
        )
        return 0
    if args.probe is None:
        parser.error("--probe is required")
    request = json.loads(sys.stdin.read())
    result = _probe(args.probe, list(request["expected_point_ids"]))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
