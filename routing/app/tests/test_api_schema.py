"""API schema and boundary tests."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TURN_RESTRICTIONS = ROOT / "routing/configs/approved_grade_turn_restrictions.csv"


def approved_turn_restriction_count() -> int:
    with TURN_RESTRICTIONS.open(encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["tool_count"] == 6
    assert data["forbidden_turn_count"] == approved_turn_restriction_count()
    assert data["external_services_enabled"] is False
    assert data["supported_hours"] == list(range(6, 19))


def test_config(client):
    data = client.get("/api/config").json()
    assert data["default_hour"] == 14
    assert data["default_max_detour_ratio"] == 0.20
    assert data["external_tiles_enabled"] is False


def test_hour_boundaries(client, od):
    origin, destination = od
    for hour in (6, 18):
        response = client.post(
            "/api/route",
            json={"origin": origin, "destination": destination, "hour": hour},
        )
        assert response.status_code == 200
    for hour in (5, 19):
        response = client.post(
            "/api/route",
            json={"origin": origin, "destination": destination, "hour": hour},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_HOUR"


def test_invalid_mode_objective_and_weights(client, od):
    origin, destination = od
    cases = [
        ({"mode": "car"}, "INVALID_MODE"),
        ({"objective": "best"}, "INVALID_OBJECTIVE"),
        ({"max_detour_ratio": 1.2}, "INVALID_SCHEMA"),
        ({"uncertainty_weight": 3.1}, "INVALID_SCHEMA"),
    ]
    for change, code in cases:
        payload = {"origin": origin, "destination": destination, "hour": 14, **change}
        response = client.post("/api/route", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == code


def test_openapi_has_all_endpoints(client):
    paths = client.get("/api/openapi.json").json()["paths"]
    required = {
        "/api/health", "/api/config", "/api/geocode", "/api/snap",
        "/api/route", "/api/compare", "/api/summarize", "/api/export",
        "/api/visual-layers",
    }
    assert required.issubset(paths)


def test_visual_layers_are_local_hourly_and_display_only(client):
    response = client.post(
        "/api/visual-layers",
        json={
            "bbox": [665500, 3548500, 666500, 3549500],
            "hour": 14,
            "max_buildings": 100,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["hour"] == 14
    assert 0 < data["building_count"] <= 100
    assert data["building_render_mode"] == "all_footprints_raster"
    assert data["building_raster_feature_count"] >= data["building_count"]
    assert data["building_width"] > 0 and data["building_height"] > 0
    assert len(data["building_png_base64"]) > 100
    assert data["shadow_width"] > 0 and data["shadow_height"] > 0
    assert len(data["shadow_png_base64"]) > 100
    assert data["display_only"] is True
    assert data["external_request_made"] is False
