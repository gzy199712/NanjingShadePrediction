"""Validated, deterministic tools for the user route interface."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
GDAL_PROXY = ROOT / "routing" / "data" / "interface" / "gdal_proxy"
GDAL_PROXY.mkdir(parents=True, exist_ok=True)
os.environ["GDAL_PAM_PROXY_DIR"] = str(GDAL_PROXY)

from osgeo import ogr, osr

ogr.UseExceptions()
osr.UseExceptions()

from routing.src.route_engine import RouteEngine
from routing.src.route_tool_schema import tool_schemas


OBJECTIVES = ("shortest", "shade", "utci", "risk_aware")
CATEGORY_PRIORITY = {
    "transit": 0,
    "place": 1,
    "poi": 2,
    "road": 3,
    "named_feature": 4,
}


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _axis(srs: osr.SpatialReference) -> osr.SpatialReference:
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


class RouteToolInterface:
    """In-process JSON tools with no external requests or location logging."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.root = Path(project_root or ROOT)
        graph = self.root / "routing" / "data" / "graph"
        self.engine = RouteEngine(
            graph / "step27_route_graph.npz",
            graph / "step27_hourly_costs.npz",
            graph / "step27_graph_metadata.json",
        )
        self.aliases = pd.read_csv(
            self.root
            / "routing"
            / "data"
            / "gazetteer"
            / "built"
            / "gazetteer_aliases.csv"
        ).fillna("")
        self.aliases["normalized_name"] = self.aliases["normalized_name"].astype(str)
        self.schemas = tool_schemas()
        self.validators = {
            name: Draft202012Validator(value["input_schema"])
            for name, value in self.schemas["tools"].items()
        }
        source = osr.SpatialReference()
        source.ImportFromEPSG(4326)
        target = osr.SpatialReference()
        target.ImportFromEPSG(32650)
        self.to_projected = osr.CoordinateTransformation(_axis(source), _axis(target))
        self.output_dir = self.root / "routing" / "outputs" / "interface"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, tool: str, payload: dict[str, Any]) -> None:
        if tool not in self.validators:
            raise ValueError(f"Unknown tool: {tool}")
        errors = sorted(
            self.validators[tool].iter_errors(payload),
            key=lambda error: list(error.path),
        )
        if errors:
            messages = [
                f"{'.'.join(str(item) for item in error.path) or '$'}: {error.message}"
                for error in errors
            ]
            raise ValueError("Schema validation failed: " + " | ".join(messages))

    def coordinate_xy(self, value: dict[str, Any]) -> tuple[float, float]:
        if value["crs"] == "EPSG:32650":
            return float(value["x"]), float(value["y"])
        x, y, *_ = self.to_projected.TransformPoint(
            float(value["longitude"]), float(value["latitude"])
        )
        return float(x), float(y)

    def geocode_place(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.validate("geocode_place", payload)
        query = str(payload["query"]).strip()
        normalized = normalize_name(query)
        limit = int(payload.get("limit", 5))
        names = self.aliases["normalized_name"]
        exact = names.eq(normalized)
        prefix = names.str.startswith(normalized)
        contains = names.str.contains(normalized, regex=False)
        if exact.any():
            selected = self.aliases.loc[exact].copy()
            selected["match_score"] = 1.0
            selected["match_type"] = "exact"
        elif prefix.any() or contains.any():
            selected = self.aliases.loc[prefix | contains].copy()
            selected["match_score"] = np.where(
                selected["normalized_name"].str.startswith(normalized), 0.92, 0.85
            )
            selected["match_type"] = np.where(
                selected["normalized_name"].str.startswith(normalized), "prefix", "substring"
            )
        else:
            unique_names = names.drop_duplicates().tolist()
            scores = []
            for candidate in tqdm(
                unique_names,
                total=len(unique_names),
                desc="Fuzzy local gazetteer",
                unit="name",
                dynamic_ncols=True,
            ):
                scores.append(SequenceMatcher(None, normalized, candidate).ratio())
            top_names = pd.DataFrame(
                {"normalized_name": unique_names, "match_score": scores}
            ).nlargest(max(limit * 3, 15), "match_score")
            selected = self.aliases.merge(top_names, on="normalized_name", how="inner")
            selected["match_type"] = "fuzzy"
        selected["category_priority"] = selected["category"].map(
            CATEGORY_PRIORITY
        ).fillna(9)
        selected["name_length_delta"] = (
            selected["normalized_name"].str.len() - len(normalized)
        ).abs()
        selected = (
            selected.sort_values(
                [
                    "match_score",
                    "category_priority",
                    "name_length_delta",
                    "normalized_name",
                    "feature_id",
                ],
                ascending=[False, True, True, True, True],
                kind="mergesort",
            )
            .drop_duplicates("feature_id")
            .head(limit)
        )
        candidates = []
        for row in tqdm(
            selected.itertuples(index=False),
            total=len(selected),
            desc="Formatting geocoder candidates",
            unit="candidate",
            dynamic_ncols=True,
        ):
            candidates.append(
                {
                    "candidate_id": str(row.feature_id),
                    "display_name": str(row.display_name),
                    "category": str(row.category),
                    "subtype": str(row.subtype),
                    "match_type": str(row.match_type),
                    "match_score": float(row.match_score),
                    "coordinate": {
                        "crs": "EPSG:4326",
                        "longitude": float(row.lon),
                        "latitude": float(row.lat),
                    },
                    "projected_coordinate": {
                        "crs": "EPSG:32650",
                        "x": float(row.x),
                        "y": float(row.y),
                    },
                }
            )
        return {
            "query": query,
            "candidate_count": len(candidates),
            "ambiguous": len(candidates) > 1,
            "selection_required": len(candidates) > 1,
            "candidates": candidates,
            "source": "offline_nanjing_osm_gazetteer",
            "external_request_made": False,
        }

    def snap_origin_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.validate("snap_origin_destination", payload)
        origin = self.coordinate_xy(payload["origin"])
        destination = self.coordinate_xy(payload["destination"])
        origin_node, destination_node, origin_distance, destination_distance = (
            self.engine.snap_pair(origin, destination)
        )
        return {
            "origin_node": origin_node,
            "destination_node": destination_node,
            "origin_snap_distance_m": origin_distance,
            "destination_snap_distance_m": destination_distance,
            "origin_snapped": {
                "crs": "EPSG:32650",
                "x": float(self.engine.node_xy[origin_node, 0]),
                "y": float(self.engine.node_xy[origin_node, 1]),
            },
            "destination_snapped": {
                "crs": "EPSG:32650",
                "x": float(self.engine.node_xy[destination_node, 0]),
                "y": float(self.engine.node_xy[destination_node, 1]),
            },
        }

    def route(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.validate("route", payload)
        objective = str(payload.get("objective", "risk_aware"))
        max_detour = payload.get("max_detour_ratio", 0.20)
        if objective == "shortest":
            max_detour = None
        started = time.perf_counter()
        result = self.engine.route(
            self.coordinate_xy(payload["origin"]),
            self.coordinate_xy(payload["destination"]),
            int(payload["hour"]),
            mode=str(payload.get("mode", "walk")),
            objective=objective,
            algorithm=str(payload.get("algorithm", "astar")),
            max_detour_ratio=max_detour,
            uncertainty_weight=float(payload.get("uncertainty_weight", 1.0)),
        )
        result["route_id"] = uuid.uuid4().hex
        result["interface_runtime_seconds"] = time.perf_counter() - started
        result["max_detour_ratio_requested"] = max_detour
        result["privacy"] = {
            "exact_location_persisted": False,
            "external_request_made": False,
        }
        return result

    def compare_routes(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.validate("compare_routes", payload)
        objectives = payload.get("objectives", list(OBJECTIVES))
        results = []
        for objective in tqdm(
            objectives,
            total=len(objectives),
            desc="Comparing route objectives",
            unit="objective",
            dynamic_ncols=True,
        ):
            route_payload = {
                "origin": payload["origin"],
                "destination": payload["destination"],
                "hour": payload["hour"],
                "mode": payload.get("mode", "walk"),
                "objective": objective,
                "algorithm": payload.get("algorithm", "astar"),
                "max_detour_ratio": payload.get("max_detour_ratio", 0.20),
                "uncertainty_weight": payload.get("uncertainty_weight", 1.0),
            }
            results.append(self.route(route_payload))
        return {
            "route_count": len(results),
            "all_found": all(result.get("found", False) for result in results),
            "routes": results,
            "shared_topology_disclosed": True,
            "privacy": {
                "exact_location_persisted": False,
                "external_request_made": False,
            },
        }

    def summarize_route(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.validate("summarize_route", payload)
        result = payload["route_result"]
        language = payload.get("language", "zh-CN")
        if not result.get("found"):
            text = (
                "未找到满足连通性和吸附约束的路径。"
                if language == "zh-CN"
                else "No route satisfies the connectivity and snap constraints."
            )
        elif language == "zh-CN":
            text = (
                f"{result.get('objective')}路线约{result.get('distance_m', 0):.0f}米，"
                f"预计{result.get('estimated_duration_min', 0):.1f}分钟；"
                f"平均遮荫率{result.get('mean_shade', 0):.1%}，"
                f"平均UTCI {result.get('mean_utci', 0):.1f}°C，"
                f"绕行率{result.get('detour_ratio', 0):.1%}。"
                f"其中低可靠性兜底路段"
                f"{result.get('spatial_fallback_length_m', 0) + result.get('prior_imputed_length_m', 0):.0f}米。"
            )
        else:
            text = (
                f"The {result.get('objective')} route is about "
                f"{result.get('distance_m', 0):.0f} m and "
                f"{result.get('estimated_duration_min', 0):.1f} min, "
                f"with {result.get('mean_shade', 0):.1%} mean shade, "
                f"{result.get('mean_utci', 0):.1f} °C mean UTCI and "
                f"{result.get('detour_ratio', 0):.1%} detour."
            )
        return {
            "language": language,
            "summary": text,
            "uncertainty_disclosed": True,
            "fallback_disclosed": True,
        }

    def _route_xy(self, route_result: dict[str, Any]) -> np.ndarray:
        node_indices = np.asarray(route_result.get("node_indices", []), dtype=int)
        if len(node_indices) < 2:
            raise ValueError("route_result does not contain a drawable node sequence")
        return self.engine.node_xy[node_indices]

    def export_route_map(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.validate("export_route_map", payload)
        result = payload["route_result"]
        xy = self._route_xy(result)
        output_format = str(payload["format"])
        route_id = str(result.get("route_id") or uuid.uuid4().hex)
        stem = f"route_{route_id}_{result.get('objective', 'route')}"
        if output_format in {"png", "pdf"}:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            path = self.output_dir / f"{stem}.{output_format}"
            figure, axis = plt.subplots(figsize=(9, 7))
            axis.plot(xy[:, 0], xy[:, 1], color="#d62728", linewidth=2.2)
            axis.scatter(xy[0, 0], xy[0, 1], c="#2ca02c", s=55, label="Origin")
            axis.scatter(xy[-1, 0], xy[-1, 1], c="#1f77b4", s=55, label="Destination")
            axis.set_aspect("equal")
            axis.set_xlabel("Easting (m), EPSG:32650")
            axis.set_ylabel("Northing (m), EPSG:32650")
            axis.set_title(
                f"{result.get('objective', 'route')} | "
                f"{result.get('distance_m', 0):.0f} m | "
                f"UTCI {result.get('mean_utci', 0):.1f} °C"
            )
            axis.legend()
            axis.grid(alpha=0.2)
            figure.tight_layout()
            figure.savefig(path, dpi=300)
            plt.close(figure)
        else:
            gdb = self.output_dir / f"{stem}.gdb"
            driver = ogr.GetDriverByName("OpenFileGDB")
            if driver is None or driver.GetMetadataItem("DCAP_CREATE") != "YES":
                raise RuntimeError("Writable OpenFileGDB driver is unavailable")
            dataset = driver.CreateDataSource(str(gdb))
            spatial_reference = osr.SpatialReference()
            spatial_reference.ImportFromEPSG(32650)
            _axis(spatial_reference)
            layer = dataset.CreateLayer(
                "Route",
                srs=spatial_reference,
                geom_type=ogr.wkbLineString,
            )
            for field, field_type in tqdm(
                [
                    ("route_id", ogr.OFTString),
                    ("objective", ogr.OFTString),
                    ("distance_m", ogr.OFTReal),
                ],
                total=3,
                desc="Creating route export fields",
                unit="field",
                dynamic_ncols=True,
            ):
                layer.CreateField(ogr.FieldDefn(field, field_type))
            line = ogr.Geometry(ogr.wkbLineString)
            for x, y in tqdm(
                xy,
                total=len(xy),
                desc="Writing route export geometry",
                unit="node",
                dynamic_ncols=True,
            ):
                line.AddPoint_2D(float(x), float(y))
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("route_id", route_id)
            feature.SetField("objective", str(result.get("objective", "")))
            feature.SetField("distance_m", float(result.get("distance_m", 0)))
            feature.SetGeometry(line)
            layer.CreateFeature(feature)
            feature = None
            dataset = None
            path = gdb
        return {
            "format": output_format,
            "output_path": str(path.resolve()),
            "exists": path.exists(),
            "route_id": route_id,
            "attribution": "© OpenStreetMap contributors",
        }

    def call(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        methods = {
            "geocode_place": self.geocode_place,
            "snap_origin_destination": self.snap_origin_destination,
            "route": self.route,
            "compare_routes": self.compare_routes,
            "summarize_route": self.summarize_route,
            "export_route_map": self.export_route_map,
        }
        if tool not in methods:
            raise ValueError(f"Unknown tool: {tool}")
        return methods[tool](payload)


def request_fingerprint(tool: str, payload: dict[str, Any]) -> str:
    """Non-reversible audit identifier; never persists the original payload."""
    material = json.dumps(
        {"tool": tool, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]
