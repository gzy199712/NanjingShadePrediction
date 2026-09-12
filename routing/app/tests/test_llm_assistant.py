"""Tests for the bounded, local-only Step31 assistant."""

from __future__ import annotations

import json

import pytest

from routing.app.backend import llm_adapter


def model_reply(**overrides) -> dict:
    value = {
        "intent": "compare_routes",
        "origin_query": "新街口",
        "destination_query": "鼓楼",
        "hour": 14,
        "mode": "walk",
        "objective": None,
        "max_detour_ratio": None,
        "uncertainty_weight": None,
    }
    value.update(overrides)
    return {"message": {"content": json.dumps(value, ensure_ascii=False)}}


def test_parse_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setattr(llm_adapter, "_post", lambda *args, **kwargs: model_reply())
    result = llm_adapter.parse_travel_request("下午2点从新街口步行到鼓楼")
    assert result["ready_for_geocoding"] is True
    assert result["explicit_confirmation_required"] is True
    assert result["missing_fields"] == []


def test_parse_reports_missing_fields(monkeypatch):
    monkeypatch.setattr(
        llm_adapter,
        "_post",
        lambda *args, **kwargs: model_reply(destination_query=None, hour=None),
    )
    result = llm_adapter.parse_travel_request("从新街口出发")
    assert result["ready_for_geocoding"] is False
    assert result["missing_fields"] == ["destination_query", "hour"]


def test_invalid_model_schema_is_rejected(monkeypatch):
    monkeypatch.setattr(
        llm_adapter,
        "_post",
        lambda *args, **kwargs: {"message": {"content": '{"hour": 25}'}},
    )
    with pytest.raises(llm_adapter.LocalLLMError):
        llm_adapter.parse_travel_request("晚上11点出发")


def test_llm_payload_contains_no_coordinates(monkeypatch):
    captured = {}

    def fake_post(path, payload, timeout=300):
        captured.update(payload)
        return model_reply()

    monkeypatch.setattr(llm_adapter, "_post", fake_post)
    llm_adapter.parse_travel_request("新街口到鼓楼")
    serialized = json.dumps(captured, ensure_ascii=False).lower()
    assert "longitude" not in serialized
    assert "latitude" not in serialized
    assert "segment_id" not in serialized


def test_explanation_only_sends_approved_metrics(monkeypatch):
    captured = {}

    def fake_post(path, payload, timeout=300):
        captured.update(payload)
        return {"message": {"content": "这是本地汇总解释。"}}

    monkeypatch.setattr(llm_adapter, "_post", fake_post)
    route = {
        "objective": "shade",
        "distance_m": 1000,
        "estimated_duration_min": 12,
        "detour_ratio": 0.1,
        "mean_shade": 0.7,
        "mean_tmrt": 42,
        "mean_utci": 36,
        "uncertainty_mean": 0.2,
        "warning_codes": [],
        "geometry": {"coordinates": [[1, 2]]},
        "segment_ids": ["secret"],
    }
    assert llm_adapter.explain_route_tradeoffs([route], "解释") == "这是本地汇总解释。"
    serialized = json.dumps(captured, ensure_ascii=False)
    assert "geometry" not in serialized
    assert "segment_ids" not in serialized


def test_non_local_model_url_is_rejected(monkeypatch):
    monkeypatch.setattr(llm_adapter, "OLLAMA_BASE_URL", "https://example.com")
    with pytest.raises(llm_adapter.LocalLLMError):
        llm_adapter._assert_local_url()
