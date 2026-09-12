"""Shared local API test client."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing.app.backend.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as value:
        yield value


@pytest.fixture(scope="session")
def od(client):
    def selected(query: str) -> dict:
        response = client.post("/api/geocode", json={"query": query, "limit": 5})
        assert response.status_code == 200
        candidates = response.json()["candidates"]
        assert candidates
        item = candidates[0]
        return {
            "crs": "EPSG:4326",
            "longitude": item["longitude"],
            "latitude": item["latitude"],
        }

    return selected("新街口"), selected("鼓楼")

