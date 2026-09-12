"""Unit tests for deterministic Step32 benchmark and scoring logic."""

from __future__ import annotations

import json

from routing.src.llm_benchmark import (
    adversarial_cases,
    development_cases,
    explanation_cases,
    formal_cases,
)
from routing.src.llm_explanation_evaluation import numeric_faithfulness
from routing.src.llm_parse_evaluation import parse_metrics, score_parse_case


def test_frozen_benchmark_sizes_and_ids():
    parse = development_cases() + formal_cases()
    assert len(parse) == 120
    assert len({case["case_id"] for case in parse}) == 120
    assert sum(case["split"] == "development" for case in parse) == 20
    assert sum(case["split"] == "formal_test" for case in parse) == 100
    assert len(adversarial_cases()) == 30
    assert len(explanation_cases()) == 30


def test_parse_scoring_exact_record():
    case = development_cases()[0]
    response = {
        "intent": "compare_routes",
        "origin_query": "新街口",
        "destination_query": "鼓楼",
        "hour": 14,
        "mode": "walk",
        "objective": None,
        "max_detour_ratio": .2,
        "uncertainty_weight": None,
        "missing_fields": [],
        "explicit_confirmation_required": True,
    }
    row = score_parse_case(case, 200, response, .1)
    assert row["full_record_exact"] is True


def test_invalid_field_requires_explicit_detection():
    case = development_cases()[6]
    response = {
        "intent": "single_route",
        "origin_query": "新街口",
        "destination_query": "鼓楼",
        "hour": None,
        "mode": "walk",
        "objective": None,
        "max_detour_ratio": None,
        "uncertainty_weight": None,
        "missing_fields": ["hour", "objective"],
        "explicit_confirmation_required": True,
    }
    row = score_parse_case(case, 200, response, .1)
    assert row["invalid_fields_match"] is False


def test_numeric_faithfulness_uses_units_and_tolerances():
    routes = explanation_cases()[0]["route_metrics"]
    passed, errors = numeric_faithfulness(
        "最短路线1800米、21.4分钟、遮荫31.0%、UTCI 43.8°C。", routes
    )
    assert passed, errors
    failed, errors = numeric_faithfulness("路线长9999米。", routes)
    assert not failed
    assert errors


def test_parse_metrics_separate_splits():
    case = development_cases()[0]
    response = {
        "intent": "compare_routes",
        "origin_query": "新街口",
        "destination_query": "鼓楼",
        "hour": 14,
        "mode": "walk",
        "objective": None,
        "max_detour_ratio": .2,
        "uncertainty_weight": None,
        "missing_fields": [],
        "explicit_confirmation_required": True,
    }
    row = score_parse_case(case, 200, response, .1)
    fields, overall = parse_metrics([row], "development")
    assert overall["case_count"] == 1
    assert overall["schema_valid_rate"] == 1
    assert len(fields) == 12
    json.dumps(overall)
