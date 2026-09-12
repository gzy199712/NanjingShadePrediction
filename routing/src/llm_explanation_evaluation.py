"""Deterministic, non-LLM scoring of Step32 route explanations."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from routing.src.llm_parse_evaluation import api_json, load_existing, write_rows


FIELDS = [
    "case_id", "category", "status", "latency_ms", "numeric_pass",
    "ranking_pass", "objective_pass", "uncertainty_pass", "scope_pass",
    "unsupported_route_claim", "unsupported_generalization",
    "uncertainty_misrepresentation", "automatic_flags", "llm_explanation",
    "route_metrics_json",
]

LABELS = {
    "shortest": "最短",
    "shade": "遮荫",
    "utci": "UTCI",
    "risk_aware": "风险",
}


def near_any(value: float, candidates: list[float], tolerance: float) -> bool:
    return any(math.isclose(value, item, abs_tol=tolerance, rel_tol=0) for item in candidates)


def numeric_faithfulness(text: str, routes: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors = []
    found = 0
    distance_values = [float(route["distance_m"]) for route in routes]
    time_values = [float(route["estimated_duration_min"]) for route in routes]
    temperature_values = [
        float(route[key]) for route in routes for key in ("mean_tmrt", "mean_utci")
    ]
    percent_values = [
        float(route[key]) * 100
        for route in routes
        for key in ("detour_ratio", "mean_shade")
    ]
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(公里|km|千米|米|m)\b", text, re.I):
        found += 1
        value = float(match.group(1)) * (1000 if match.group(2).lower() in {"公里", "km", "千米"} else 1)
        if not near_any(value, distance_values, 1.0):
            errors.append(f"distance:{match.group(0)}")
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(分钟|min)\b", text, re.I):
        found += 1
        if not near_any(float(match.group(1)), time_values, .1):
            errors.append(f"time:{match.group(0)}")
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:°C|℃|摄氏度)", text, re.I):
        found += 1
        if not near_any(float(match.group(1)), temperature_values, .1):
            errors.append(f"temperature:{match.group(0)}")
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        found += 1
        if not near_any(float(match.group(1)), percent_values, .1):
            errors.append(f"percentage:{match.group(0)}")
    if found == 0:
        errors.append("no_verifiable_numeric_reference")
    return not errors, errors


def ranking_faithfulness(text: str, routes: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors = []
    expected = {
        "最短": min(routes, key=lambda route: route["distance_m"])["objective"],
        "遮荫最高": max(routes, key=lambda route: route["mean_shade"])["objective"],
        "UTCI最低": min(routes, key=lambda route: route["mean_utci"])["objective"],
        "最可靠": min(routes, key=lambda route: route["uncertainty_mean"])["objective"],
    }
    explicit_claim_patterns = {
        "最短": r"(shortest|最短距离路线|最短路线).{0,12}(?:是|为)?最短|最短的是.{0,12}(shortest|shade|utci|risk)",
        "遮荫最高": r"(shortest|shade|utci|risk).{0,12}遮荫(?:率)?最高|遮荫(?:率)?最高的是.{0,12}(shortest|shade|utci|risk)",
        "UTCI最低": r"(shortest|shade|utci|risk).{0,12}UTCI最低|UTCI最低的是.{0,12}(shortest|shade|utci|risk)",
        "最可靠": r"(shortest|shade|utci|risk).{0,12}(?:最可靠|不确定性最低)|(?:最可靠|不确定性最低)的是.{0,12}(shortest|shade|utci|risk)",
    }
    lower = text.lower()
    for claim, pattern in explicit_claim_patterns.items():
        match = re.search(pattern, lower, re.I)
        if not match:
            continue
        captured = next((value for value in match.groups() if value), "")
        normalized = (
            "risk_aware" if captured.startswith("risk")
            else "shortest" if captured.startswith("shortest") or captured == "最短距离路线" or captured == "最短路线"
            else captured
        )
        if normalized and normalized != expected[claim]:
            errors.append(f"{claim}:{normalized}!={expected[claim]}")
    return not errors, errors


def score_explanation(case: dict[str, Any], text: str, elapsed: float, success: bool) -> dict[str, Any]:
    routes = case["route_metrics"]
    numeric_pass, numeric_errors = numeric_faithfulness(text, routes) if success else (False, ["model_failure"])
    ranking_pass, ranking_errors = ranking_faithfulness(text, routes) if success else (False, ["model_failure"])
    lower = text.lower()
    objective_checks = [
        "shortest" in lower and ("长度" in text or "距离" in text),
        "shade" in lower and "遮荫" in text,
        "utci" in lower and "UTCI" in text,
        ("risk" in lower or "风险" in text) and "不确定性" in text and "UTCI" in text,
    ]
    objective_pass = success and all(objective_checks)
    uncertainty_bad = any(term in text for term in ("95%置信区间", "统计显著", "保证区间", "实际误差范围"))
    uncertainty_pass = success and "集成预测不确定性" in text and not uncertainty_bad
    scope_bad = any(term in text for term in ("任意日期都适用", "任意季节都适用", "已完成跨城市验证", "实测实时天气"))
    scope_pass = success and "2024年7月29日" in text and ("06:00" in text or "6:00" in text) and "18:00" in text and not scope_bad
    unsupported_route = any(term in text for term in ("保证最凉", "保证最佳", "实时实测", "绝对安全"))
    flags = numeric_errors + ranking_errors
    if not objective_pass:
        flags.append("objective_definition_missing_or_incorrect")
    if not uncertainty_pass:
        flags.append("uncertainty_term_missing_or_misrepresented")
    if not scope_pass:
        flags.append("scope_disclosure_missing_or_incorrect")
    passed = (
        numeric_pass and ranking_pass and objective_pass and uncertainty_pass
        and scope_pass and not unsupported_route and not scope_bad
    )
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "status": "PASS" if passed else "FAIL",
        "latency_ms": round(elapsed * 1000, 3),
        "numeric_pass": numeric_pass,
        "ranking_pass": ranking_pass,
        "objective_pass": objective_pass,
        "uncertainty_pass": uncertainty_pass,
        "scope_pass": scope_pass,
        "unsupported_route_claim": unsupported_route,
        "unsupported_generalization": scope_bad,
        "uncertainty_misrepresentation": uncertainty_bad,
        "automatic_flags": json.dumps(flags, ensure_ascii=False),
        "llm_explanation": text,
        "route_metrics_json": json.dumps(routes, ensure_ascii=False, sort_keys=True),
    }


def run_explanations(
    cases: list[dict[str, Any]], output_path: Path, *, resume: bool
) -> list[dict[str, Any]]:
    rows, completed = load_existing(output_path) if resume else ([], set())
    for case in tqdm(
        cases,
        total=len(cases),
        desc="Step32 explanation benchmark",
        unit="case",
        dynamic_ncols=True,
    ):
        if case["case_id"] in completed:
            continue
        try:
            status, response, elapsed = api_json(
                "/api/assistant/explain",
                {"routes": case["route_metrics"], "question": case["question"]},
            )
            text = str(response.get("explanation", ""))
            row = score_explanation(case, text, elapsed, status == 200)
        except Exception as exc:
            row = score_explanation(case, f"MODEL_ERROR:{type(exc).__name__}", 0, False)
        rows.append(row)
        write_rows(output_path, rows, FIELDS)
    return rows


def explanation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = lambda value: str(value).lower() in {"true", "1"}
    count = len(rows) or 1
    return {
        "case_count": len(rows),
        "numeric_faithfulness_rate": sum(truth(row["numeric_pass"]) for row in rows) / count,
        "route_ranking_faithfulness_rate": sum(truth(row["ranking_pass"]) for row in rows) / count,
        "objective_faithfulness_rate": sum(truth(row["objective_pass"]) for row in rows) / count,
        "scope_faithfulness_rate": sum(truth(row["scope_pass"]) for row in rows) / count,
        "unsupported_route_claim_count": sum(truth(row["unsupported_route_claim"]) for row in rows),
        "unsupported_model_generalization_count": sum(truth(row["unsupported_generalization"]) for row in rows),
        "uncertainty_misrepresentation_count": sum(truth(row["uncertainty_misrepresentation"]) for row in rows),
    }


def write_manual_review(rows: list[dict[str, Any]], output_path: Path, sample_size: int = 20) -> None:
    rng = random.Random(32)
    selected = rng.sample(rows, min(sample_size, len(rows)))
    manual = [
        {
            "case_id": row["case_id"],
            "route_metrics_json": row["route_metrics_json"],
            "llm_explanation": row["llm_explanation"],
            "automatic_flags": row["automatic_flags"],
            "numeric_pass": row["numeric_pass"],
            "ranking_pass": row["ranking_pass"],
            "scope_pass": row["scope_pass"],
            "reviewer_status": "",
            "reviewer_comment": "",
        }
        for row in selected
    ]
    write_rows(
        output_path,
        manual,
        [
            "case_id", "route_metrics_json", "llm_explanation",
            "automatic_flags", "numeric_pass", "ranking_pass", "scope_pass",
            "reviewer_status", "reviewer_comment",
        ],
    )
