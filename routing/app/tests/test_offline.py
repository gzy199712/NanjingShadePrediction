"""Offline-operation checks."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_health_geocode_route_and_map_work_without_external_services(client, od):
    assert client.get("/api/health").json()["external_services_enabled"] is False
    assert client.post("/api/geocode", json={"query": "玄武湖"}).status_code == 200
    origin, destination = od
    route = client.post(
        "/api/route",
        json={"origin": origin, "destination": destination, "hour": 14},
    )
    assert route.status_code == 200
    assert route.json()["geometry"]["coordinates"]
    assert client.get("/").status_code == 200
    assert client.get("/assets/local_context.json").status_code == 200


def test_no_runtime_remote_urls_in_application_source():
    sources = [
        *list((ROOT / "routing/app/backend").rglob("*.py")),
        *list((ROOT / "routing/app").rglob("*.js")),
        *list((ROOT / "routing/app").rglob("*.html")),
        *list((ROOT / "routing/app").rglob("*.css")),
    ]
    content = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "nomin" + "atim" not in content.lower()
    assert "google" + "apis" not in content.lower()
    assert "tile." + "openstreetmap" not in content.lower()
    assert "https://" not in content.lower()
