"""Frozen route-tool integration tests."""

import math


def payload(od, objective="shortest", algorithm="astar"):
    origin, destination = od
    return {
        "origin": origin,
        "destination": destination,
        "hour": 14,
        "mode": "walk",
        "objective": objective,
        "algorithm": algorithm,
        "max_detour_ratio": 0.20,
        "uncertainty_weight": 0.0,
    }


def test_snap_within_formal_limit(client, od):
    origin, destination = od
    response = client.post(
        "/api/snap",
        json={"origin": origin, "destination": destination, "mode": "walk"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["origin_snap_distance_m"] <= 100
    assert data["destination_snap_distance_m"] <= 100
    assert data["same_component"] is True


def test_three_objective_comparison(client, od):
    response = client.post("/api/compare", json=payload(od))
    assert response.status_code == 200
    data = response.json()
    assert data["route_count"] == 3
    assert data["all_found"] is True
    assert {route["objective"] for route in data["routes"]} == {
        "shortest", "shade", "utci"
    }
    required = {
        "distance_m", "estimated_duration_min", "detour_ratio", "mean_shade",
        "shade_exposure", "mean_tmrt", "max_tmrt", "mean_utci", "max_utci",
        "uncertainty_mean", "uncertainty_max", "direct_length_m",
        "interpolated_length_m", "low_reliability_length_m", "fallback_length_m",
        "prior_imputed_length_m", "center_inside_length_m",
        "center_outside_length_m", "route_found", "warning_codes", "geometry",
        "reliability_score", "reliability_grade", "uncertainty_percentile",
    }
    for route in data["routes"]:
        assert required.issubset(route)
        assert route["geometry"]["type"] == "LineString"
        assert len(route["geometry"]["coordinates"]) >= 2
        for key in ("distance_m", "mean_shade", "mean_utci", "uncertainty_mean"):
            assert math.isfinite(route[key])


def test_astar_matches_dijkstra(client, od):
    for objective in ("shortest", "shade", "utci"):
        astar = client.post("/api/route", json=payload(od, objective, "astar")).json()
        dijkstra = client.post("/api/route", json=payload(od, objective, "dijkstra")).json()
        assert abs(astar["total_cost"] - dijkstra["total_cost"]) <= 1e-6
        assert astar["segment_ids"] == dijkstra["segment_ids"]


def test_summary_discloses_scope_and_fallback(client, od):
    route = client.post("/api/route", json=payload(od)).json()
    response = client.post(
        "/api/summarize", json={"route_id": route["route_id"], "language": "zh-CN"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["fallback_disclosed"] is True
    assert data["uncertainty_disclosed"] is True
    assert "2024-08-04" in data["scenario_disclosure"]
    assert "2024-07-29" in data["scenario_disclosure"]
