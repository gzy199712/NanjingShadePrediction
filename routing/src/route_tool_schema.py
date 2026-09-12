"""Approved JSON tool schemas for the Phase G8 route interface."""

from __future__ import annotations

from typing import Any


COORDINATE_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["crs", "longitude", "latitude"],
            "properties": {
                "crs": {"const": "EPSG:4326"},
                "longitude": {"type": "number", "minimum": 118.0, "maximum": 119.5},
                "latitude": {"type": "number", "minimum": 31.0, "maximum": 33.0},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["crs", "x", "y"],
            "properties": {
                "crs": {"const": "EPSG:32650"},
                "x": {"type": "number", "minimum": 500000, "maximum": 800000},
                "y": {"type": "number", "minimum": 3400000, "maximum": 3700000},
            },
        },
    ]
}


def tool_schemas() -> dict[str, Any]:
    route_properties = {
        "origin": COORDINATE_SCHEMA,
        "destination": COORDINATE_SCHEMA,
        "hour": {"type": "integer", "minimum": 6, "maximum": 18},
        "mode": {"enum": ["walk", "bike", "shared"]},
        "objective": {
            "enum": ["shortest", "shade", "utci", "risk_aware"]
        },
        "algorithm": {"enum": ["astar", "dijkstra"]},
        "max_detour_ratio": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
        },
        "uncertainty_weight": {
            "type": "number",
            "minimum": 0,
            "maximum": 3,
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": "phase_g8_approved_v1",
        "tools": {
            "geocode_place": {
                "status": "local_gazetteer_ready",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "minLength": 2},
                        "city": {"type": "string", "default": "南京市"},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                        },
                    },
                },
            },
            "snap_origin_destination": {
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["origin", "destination"],
                    "properties": {
                        "origin": COORDINATE_SCHEMA,
                        "destination": COORDINATE_SCHEMA,
                    },
                }
            },
            "route": {
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["origin", "destination", "hour"],
                    "properties": route_properties,
                }
            },
            "compare_routes": {
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["origin", "destination", "hour"],
                    "properties": {
                        **route_properties,
                        "objectives": {
                            "type": "array",
                            "items": {
                                "enum": [
                                    "shortest",
                                    "shade",
                                    "utci",
                                    "risk_aware",
                                ]
                            },
                            "minItems": 1,
                            "uniqueItems": True,
                            "default": [
                                "shortest",
                                "shade",
                                "utci",
                                "risk_aware",
                            ],
                        },
                    },
                }
            },
            "summarize_route": {
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["route_result"],
                    "properties": {
                        "route_result": {"type": "object"},
                        "language": {"enum": ["zh-CN", "en"], "default": "zh-CN"},
                    },
                }
            },
            "export_route_map": {
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["route_result", "format"],
                    "properties": {
                        "route_result": {"type": "object"},
                        "format": {"enum": ["png", "pdf", "file_geodatabase"]},
                        "include_uncertainty": {
                            "type": "boolean",
                            "default": True,
                        },
                        "include_fallback": {
                            "type": "boolean",
                            "default": True,
                        },
                    },
                }
            },
        },
    }
