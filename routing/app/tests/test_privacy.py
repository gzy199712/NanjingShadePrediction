"""Privacy and local-only source-policy tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_privacy_config(client):
    data = client.get("/api/config").json()["privacy"]
    assert data["exact_location_persisted"] is False
    assert data["request_body_logged"] is False
    assert data["search_history_logged"] is False
    assert data["external_requests_enabled"] is False


def test_no_external_frontend_dependencies():
    frontend = ROOT / "routing/app/frontend"
    content = "\n".join(
        path.read_text(encoding="utf-8")
        for path in frontend.rglob("*")
        if path.is_file() and path.suffix in {".html", ".js", ".css"}
    ).lower()
    assert "https://" not in content
    assert "http://" not in content
    assert "cdn" not in content


def test_privacy_middleware_does_not_read_body():
    source = (ROOT / "routing/app/backend/privacy.py").read_text(encoding="utf-8")
    assert "request.body" not in source
    assert "longitude" not in source
    assert "latitude" not in source


def test_service_binds_localhost_only():
    start = (ROOT / "routing/scripts/start_local_route_app.bat").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "routing/app/backend/run_local_server.py").read_text(
        encoding="utf-8"
    )
    assert 'host="127.0.0.1"' in runner
    assert 'host="0.0.0.0"' not in runner
    assert "New-NetFirewallRule" not in start
