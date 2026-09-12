"""Offline local gazetteer tests."""

import pytest


@pytest.mark.parametrize(
    "query",
    ["新街口", "鼓楼", "夫子庙", "玄武湖", "南京站", "Nanjing站", "  新街口  "],
)
def test_known_and_normalized_queries(client, query):
    response = client.post("/api/geocode", json={"query": query, "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["external_request_made"] is False
    assert data["candidate_count"] >= 1
    assert all("candidate_id" in item for item in data["candidates"])


def test_empty_query_rejected(client):
    response = client.post("/api/geocode", json={"query": ""})
    assert response.status_code == 422


def test_nonexistent_place_not_silently_selected(client):
    response = client.post("/api/geocode", json={"query": "绝对不存在的测试地名XYZ", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_count"] == 0


def test_ambiguity_requires_selection(client):
    data = client.post("/api/geocode", json={"query": "鼓楼", "limit": 10}).json()
    assert data["candidate_count"] > 1
    assert data["ambiguity_flag"] is True
    assert data["selection_required"] is True

