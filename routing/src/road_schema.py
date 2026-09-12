"""Read-only road schema, CRS and feature loading helpers for Phase G1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


@dataclass
class RoadFeature:
    edge_id: str
    source_fid: int
    geometry: Any
    attrs: dict[str, Any]
    length_m: float
    start: tuple[float, float]
    end: tuple[float, float]
    bearing_deg: float
    envelope: tuple[float, float, float, float]
    inside_center: bool = False
    inside_center_500m: bool = False
    inside_center_1000m: bool = False
    walk_status: str = "uncertain"
    bike_status: str = "uncertain"
    shared_status: str = "uncertain"


def companion_file_qc(path: Path) -> list[dict[str, Any]]:
    required = [".shp", ".shx", ".dbf", ".prj"]
    optional = [".cpg", ".sbn", ".sbx", ".shp.xml"]
    rows: list[dict[str, Any]] = []
    for suffix in tqdm(
        required + optional,
        desc=f"Checking {path.stem} companions",
        unit="file",
        dynamic_ncols=True,
    ):
        candidate = path.with_suffix(suffix)
        rows.append(
            {
                "dataset": path.stem,
                "component": suffix,
                "required": suffix in required,
                "exists": candidate.exists(),
                "size_bytes": candidate.stat().st_size if candidate.exists() else None,
                "path": str(candidate),
            }
        )
    return rows


def describe_dataset(path: Path) -> dict[str, Any]:
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise FileNotFoundError(path)
    layer = dataset.GetLayer(0)
    definition = layer.GetLayerDefn()
    spatial_reference = layer.GetSpatialRef()
    authority = None
    if spatial_reference is not None:
        authority = spatial_reference.GetAuthorityCode(None) or spatial_reference.GetAuthorityCode(
            "PROJCS"
        )
    return {
        "path": str(path),
        "readable": True,
        "feature_count": int(layer.GetFeatureCount()),
        "geometry_type": ogr.GeometryTypeToName(definition.GetGeomType()),
        "field_count": int(definition.GetFieldCount()),
        "extent": list(layer.GetExtent()),
        "spatial_reference_name": (
            spatial_reference.GetName() if spatial_reference is not None else None
        ),
        "spatial_reference_wkt": (
            spatial_reference.ExportToWkt() if spatial_reference is not None else None
        ),
        "authority_code": int(authority) if authority else None,
    }


def field_inventory(path: Path) -> list[dict[str, Any]]:
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
    layer = dataset.GetLayer(0)
    definition = layer.GetLayerDefn()
    field_names = [
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    ]
    null_counts = {field: 0 for field in field_names}
    unique_values = {field: set() for field in field_names}
    examples: dict[str, list[str]] = {field: [] for field in field_names}
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Inventorying road fields",
        unit="road",
        dynamic_ncols=True,
    ):
        for field in field_names:
            value = feature.GetField(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                null_counts[field] += 1
                continue
            text = str(value)
            unique_values[field].add(text)
            if len(examples[field]) < 5 and text not in examples[field]:
                examples[field].append(text)
    rows: list[dict[str, Any]] = []
    important = {
        "fclass",
        "fclass_cn",
        "bridge",
        "tunnel",
        "name",
        "ref",
        "oneway",
        "layer",
        "z_level",
        "level",
        "access",
        "foot",
        "bicycle",
        "surface",
        "maxspeed",
    }
    for index in tqdm(
        range(definition.GetFieldCount()),
        desc="Summarizing road fields",
        unit="field",
        dynamic_ncols=True,
    ):
        field = definition.GetFieldDefn(index)
        name = field.GetName()
        rows.append(
            {
                "field_name": name,
                "field_type": field.GetTypeName(),
                "width": field.GetWidth(),
                "precision": field.GetPrecision(),
                "null_count": null_counts[name],
                "unique_value_count": len(unique_values[name]),
                "example_values": " | ".join(examples[name]),
                "important_field": name.lower() in important,
                "availability": "AVAILABLE",
            }
        )
    existing_lower = {name.lower() for name in field_names}
    for name in tqdm(
        sorted(important - existing_lower),
        desc="Recording unavailable fields",
        unit="field",
        dynamic_ncols=True,
    ):
        rows.append(
            {
                "field_name": name,
                "field_type": None,
                "width": None,
                "precision": None,
                "null_count": None,
                "unique_value_count": None,
                "example_values": None,
                "important_field": True,
                "availability": "NOT_AVAILABLE",
            }
        )
    return rows


def _bearing(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return 0.0
    return float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)


def load_projected_roads(
    path: Path,
    target_epsg: int,
    center_geometry: Any,
) -> tuple[list[RoadFeature], dict[str, Any]]:
    from osgeo import ogr, osr

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
    layer = dataset.GetLayer(0)
    definition = layer.GetLayerDefn()
    source = layer.GetSpatialRef()
    if source is None:
        raise RuntimeError("Road source has no spatial reference")
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(target_epsg)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transformation = osr.CoordinateTransformation(source, target)
    fields = [
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    ]
    center_500 = center_geometry.Buffer(500)
    center_1000 = center_geometry.Buffer(1000)
    roads: list[RoadFeature] = []
    failed = 0
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Projecting roads for audit",
        unit="road",
        dynamic_ncols=True,
    ):
        try:
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                failed += 1
                continue
            projected = geometry.Clone()
            projected.Transform(transformation)
            if projected.GetGeometryName().upper().startswith("MULTI"):
                first = projected.GetGeometryRef(0)
            else:
                first = projected
            point_count = first.GetPointCount()
            if point_count == 0:
                failed += 1
                continue
            start = tuple(map(float, first.GetPoint(0)[:2]))
            end = tuple(map(float, first.GetPoint(point_count - 1)[:2]))
            attrs = {field: feature.GetField(field) for field in fields}
            edge_id = str(attrs.get("osm_id") or attrs.get("OBJECTID") or feature.GetFID())
            envelope_raw = projected.GetEnvelope()
            road = RoadFeature(
                edge_id=edge_id,
                source_fid=int(feature.GetFID()),
                geometry=projected,
                attrs=attrs,
                length_m=float(projected.Length()),
                start=start,
                end=end,
                bearing_deg=_bearing(start, end),
                envelope=(
                    float(envelope_raw[0]),
                    float(envelope_raw[1]),
                    float(envelope_raw[2]),
                    float(envelope_raw[3]),
                ),
            )
            road.inside_center = bool(projected.Intersects(center_geometry))
            road.inside_center_500m = bool(projected.Intersects(center_500))
            road.inside_center_1000m = bool(projected.Intersects(center_1000))
            roads.append(road)
        except Exception:
            failed += 1
    return roads, {
        "source_count": int(layer.GetFeatureCount()),
        "loaded_count": len(roads),
        "failed_count": failed,
        "source_crs": source.GetName(),
        "source_authority": source.GetAuthorityCode(None),
        "target_epsg": target_epsg,
        "projection_copy_required": (
            source.GetAuthorityCode(None) != str(target_epsg)
            and source.GetAuthorityCode("PROJCS") != str(target_epsg)
        ),
    }


def load_union_geometry(path: Path, target_epsg: int) -> Any:
    from osgeo import ogr, osr

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
    layer = dataset.GetLayer(0)
    source = layer.GetSpatialRef()
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(target_epsg)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transformation = osr.CoordinateTransformation(source, target)
    union = None
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc=f"Reading {path.stem}",
        unit="feature",
        dynamic_ncols=True,
    ):
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue
        value = geometry.Clone()
        value.Transform(transformation)
        union = value if union is None else union.Union(value)
    if union is None:
        raise RuntimeError(f"No valid geometry in {path}")
    return union


def load_points(path: Path, target_epsg: int) -> list[dict[str, Any]]:
    from osgeo import ogr, osr

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
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
        raise RuntimeError("Thermal point ID field is unavailable")
    source = layer.GetSpatialRef()
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(target_epsg)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transformation = osr.CoordinateTransformation(source, target)
    points: list[dict[str, Any]] = []
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Loading thermal points",
        unit="point",
        dynamic_ncols=True,
    ):
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue
        value = geometry.Clone()
        value.Transform(transformation)
        points.append(
            {
                "point_id": str(feature.GetField(id_field)),
                "geometry": value,
                "x": float(value.GetX()),
                "y": float(value.GetY()),
            }
        )
    return points
