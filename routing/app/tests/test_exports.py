"""Application export tests."""

from pathlib import Path


def test_all_export_formats(client, od):
    origin, destination = od
    route = client.post(
        "/api/route",
        json={
            "origin": origin, "destination": destination, "hour": 14,
                "mode": "walk", "objective": "shortest", "algorithm": "astar",
                "max_detour_ratio": 0.20, "uncertainty_weight": 0.0,
        },
    ).json()
    for output_format in ("geojson", "csv", "png", "pdf", "file_geodatabase"):
        response = client.post(
            "/api/export",
            json={"route_id": route["route_id"], "format": output_format},
        )
        assert response.status_code == 200
        data = response.json()
        path = Path(data["local_path"])
        assert path.exists()
        assert route["route_id"] in path.name
        assert data["contains_location_in_filename"] is False
