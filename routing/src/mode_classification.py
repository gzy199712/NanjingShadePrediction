"""Two-class thermal-comfort road screening for Phase G2 review."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from routing.src.road_schema import RoadFeature


WALK_LIKELY = {
    "footway",
    "pedestrian",
    "path",
    "living_street",
    "residential",
    "service",
    "unclassified",
    "track",
    "steps",
}
BIKE_LIKELY = {
    "cycleway",
    "path",
    "living_street",
    "residential",
    "service",
    "unclassified",
    "track",
    "tertiary",
    "tertiary_link",
    "secondary",
    "secondary_link",
}
MOTOR_LIKELY = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
}
MANUAL = {
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}


def _text(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def candidate_status(fclass: str, mode: str) -> str:
    if mode == "walk":
        if fclass in WALK_LIKELY:
            return "likely_allowed"
        if fclass in MOTOR_LIKELY:
            return "likely_prohibited"
        if fclass in MANUAL:
            return "manual_review"
        return "uncertain"
    if mode == "bike":
        if fclass in BIKE_LIKELY:
            return "likely_allowed"
        if fclass in MOTOR_LIKELY or fclass in {"steps", "footway"}:
            return "likely_prohibited"
        if fclass in MANUAL:
            return "manual_review"
        return "uncertain"
    walk = candidate_status(fclass, "walk")
    bike = candidate_status(fclass, "bike")
    if walk == bike == "likely_allowed":
        return "likely_allowed"
    if "manual_review" in {walk, bike}:
        return "manual_review"
    if walk == bike == "likely_prohibited":
        return "likely_prohibited"
    return "uncertain"


def assign_candidate_status(roads: list[RoadFeature]) -> None:
    for road in tqdm(
        roads,
        desc="Assigning non-binding mode candidates",
        unit="road",
        dynamic_ncols=True,
    ):
        fclass = _text(road.attrs.get("fclass"))
        road.walk_status = candidate_status(fclass, "walk")
        road.bike_status = candidate_status(fclass, "bike")
        road.shared_status = candidate_status(fclass, "shared")


def rule_template(roads: list[RoadFeature]) -> list[dict[str, Any]]:
    combinations: dict[tuple[str, str], dict[str, Any]] = {}
    for road in tqdm(
        roads,
        desc="Building mode rule template",
        unit="road",
        dynamic_ncols=True,
    ):
        fclass = _text(road.attrs.get("fclass")) or "NOT_AVAILABLE"
        fclass_cn = str(road.attrs.get("fclass_cn") or "")
        combinations[(fclass, fclass_cn)] = {
            "fclass": fclass,
            "fclass_cn": fclass_cn,
            "bridge_condition": "ANY; review T/F and layer",
            "tunnel_condition": "ANY; review T/F and layer",
            "walk_candidate": candidate_status(fclass, "walk"),
            "bike_candidate": candidate_status(fclass, "bike"),
            "shared_candidate": candidate_status(fclass, "shared"),
            "confidence": (
                "medium"
                if fclass in WALK_LIKELY | BIKE_LIKELY | MOTOR_LIKELY
                else "low"
            ),
            "rationale": (
                "Conservative class-based candidate only; access/foot/bicycle "
                "attributes are unavailable and bridge/tunnel legality needs review."
            ),
            "requires_manual_review": True,
        }
    return [
        combinations[key]
        for key in sorted(combinations, key=lambda value: (value[0], value[1]))
    ]


def mode_is_candidate(road: RoadFeature, mode: str) -> bool:
    status = {
        "walk": road.walk_status,
        "bike": road.bike_status,
        "shared": road.shared_status,
    }[mode]
    return status in {"likely_allowed", "manual_review", "uncertain"}


def load_mode_rules(path: Any) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    required = {"output_classes", "motor_vehicle_only_rules"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"Invalid mode rule file: {path}")
    return value


def read_road_attributes(path: Any) -> pd.DataFrame:
    from osgeo import ogr

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
    rows: list[dict[str, Any]] = []
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Reading mode classification attributes",
        unit="road",
        dynamic_ncols=True,
    ):
        values = {field: feature.GetField(field) for field in fields}
        values["edge_id"] = str(
            values.get("osm_id") or values.get("OBJECTID") or feature.GetFID()
        )
        rows.append(values)
    return pd.DataFrame(rows)


def classify_mode_preview(
    roads: pd.DataFrame,
    rules: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    motor_rules = rules["motor_vehicle_only_rules"]
    always_motor = set(motor_rules["always_by_fclass"])
    grade_separated_motor = set(motor_rules["when_grade_separated"])
    preserved_active = set(
        rules.get("thermal_comfort_relevant_rules", {}).get(
            "explicitly_preserve_even_when_grade_separated", []
        )
    )
    records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for row in tqdm(
        roads.to_dict("records"),
        desc="Previewing mode rule effects",
        unit="road",
        dynamic_ncols=True,
    ):
        fclass = str(row.get("fclass") or "unknown").strip().lower()
        bridge = _text(row.get("bridge")) in {"t", "true", "1", "yes"}
        tunnel = _text(row.get("tunnel")) in {"t", "true", "1", "yes"}
        layer = row.get("layer")
        try:
            nonzero_layer = layer is not None and float(layer) != 0
        except (TypeError, ValueError):
            nonzero_layer = False
            conflicts.append(
                {
                    "edge_id": row["edge_id"],
                    "conflict_type": "invalid_layer",
                    "fclass": fclass,
                    "bridge": row.get("bridge"),
                    "tunnel": row.get("tunnel"),
                    "detail": f"Invalid layer value {layer!r}; treated as zero.",
                }
            )
        grade_separated = bridge or tunnel or nonzero_layer
        if fclass in always_motor:
            mode_status = "motor_vehicle_only"
            classification_basis = "motor_vehicle_class"
        elif fclass in grade_separated_motor and grade_separated:
            mode_status = "motor_vehicle_only"
            classification_basis = "vehicular_class_and_grade_separated"
        else:
            mode_status = "thermal_comfort_relevant"
            classification_basis = (
                "active_travel_class_preserved"
                if grade_separated and fclass in preserved_active
                else "default_thermal_comfort_relevant"
            )
        records.append(
            {
                "edge_id": str(row["edge_id"]),
                "fclass": fclass,
                "fclass_cn": row.get("fclass_cn"),
                "name": row.get("name"),
                "ref": row.get("ref"),
                "oneway": row.get("oneway"),
                "maxspeed": row.get("maxspeed"),
                "layer": layer,
                "bridge": row.get("bridge"),
                "tunnel": row.get("tunnel"),
                "thermal_comfort_class_preview": mode_status,
                "mode_status_preview": mode_status,
                "thermal_comfort_relevant": (
                    mode_status == "thermal_comfort_relevant"
                ),
                "motor_vehicle_only": mode_status == "motor_vehicle_only",
                "classification_basis": classification_basis,
                "grade_separated": grade_separated,
                "structure_review": False,
                "manual_review": False,
                "formal_rule_applied": False,
            }
        )
    return pd.DataFrame(records), pd.DataFrame(conflicts)
