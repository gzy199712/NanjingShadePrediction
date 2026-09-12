"""Step32 parsing and repeatability evaluation against the frozen Step31 API."""

from __future__ import annotations

import csv
import json
import math
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm


API_BASE = "http://127.0.0.1:8765"
FLOAT_TOLERANCE = 1e-6


def api_json(
    path: str, payload: dict[str, Any] | None = None, timeout: int = 360
) -> tuple[int, dict[str, Any], float]:
    request = urllib.request.Request(
        f"{API_BASE}{path}",
        data=(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        ),
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                response.status,
                json.loads(response.read().decode("utf-8")),
                time.perf_counter() - started,
            )
    except urllib.error.HTTPError as exc:
        try:
            data = json.loads(exc.read().decode("utf-8"))
        except Exception:
            data = {"error": {"code": "INTERNAL_EVALUATION_ERROR"}}
        return exc.code, data, time.perf_counter() - started


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def exact(expected: Any, actual: Any) -> bool:
    return normalize_text(expected) == normalize_text(actual)


def float_match(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    try:
        return math.isclose(
            float(expected), float(actual), abs_tol=FLOAT_TOLERANCE, rel_tol=0
        )
    except (TypeError, ValueError):
        return False


def load_existing(path: Path, key: str = "case_id") -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows, {row[key] for row in rows}


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


PARSE_RESULT_FIELDS = [
    "case_id", "split", "category", "status", "error_category", "latency_ms",
    "schema_valid", "origin_match", "destination_match", "hour_match",
    "mode_match", "objective_match", "max_detour_match", "uncertainty_match",
    "compare_all_match", "missing_fields_match", "invalid_fields_match",
    "ambiguity_action_match", "refusal_match", "full_record_exact",
    "expected_missing_fields", "actual_missing_fields", "expected_invalid_fields",
    "actual_invalid_fields", "expected_ambiguity_action",
    "explicit_confirmation_required", "raw_response_json",
]


def score_parse_case(case: dict[str, Any], status: int, result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    success = status == 200 and "error" not in result
    actual_missing = result.get("missing_fields", []) if success else []
    actual_invalid = result.get("invalid_fields", []) if success else []
    expected_invalid = case["expected_invalid_fields"]
    expected_compare = bool(case["expected_compare_all"])
    ambiguity_expected = case["expected_ambiguity_action"] == "require_candidate_selection"
    confirmation = bool(result.get("explicit_confirmation_required", False)) if success else False
    values = {
        "origin_match": success and exact(case["expected_origin_text"], result.get("origin_query")),
        "destination_match": success and exact(case["expected_destination_text"], result.get("destination_query")),
        "hour_match": success and case["expected_hour"] == result.get("hour"),
        "mode_match": success and case["expected_mode"] == result.get("mode"),
        "objective_match": success and case["expected_objective"] == result.get("objective"),
        "max_detour_match": success and float_match(case["expected_max_detour_ratio"], result.get("max_detour_ratio")),
        "uncertainty_match": success and float_match(case["expected_uncertainty_weight"], result.get("uncertainty_weight")),
        "compare_all_match": success and ((result.get("intent") == "compare_routes") == expected_compare),
        "missing_fields_match": success and set(case["expected_missing_fields"]) == set(actual_missing),
        "invalid_fields_match": success and set(expected_invalid) == set(actual_invalid),
        "ambiguity_action_match": (not ambiguity_expected) or (success and confirmation),
        "refusal_match": (not case["expected_refusal"]) or (not success) or bool(result.get("refusal", False)),
    }
    full_exact = success and all(values.values())
    error_code = result.get("error", {}).get("code", "") if not success else ""
    if not success and error_code in {"INTERNAL_TOOL_ERROR", "MODEL_TIMEOUT"}:
        error_category = "MODEL_TIMEOUT" if elapsed >= 295 else "MODEL_UNAVAILABLE"
    elif not success:
        error_category = "MODEL_SCHEMA_VIOLATION"
    else:
        error_category = ""
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "category": case["category"],
        "status": "PASS" if full_exact else "FAIL",
        "error_category": error_category,
        "latency_ms": round(elapsed * 1000, 3),
        "schema_valid": success,
        **values,
        "full_record_exact": full_exact,
        "expected_missing_fields": json.dumps(case["expected_missing_fields"], ensure_ascii=False),
        "actual_missing_fields": json.dumps(actual_missing, ensure_ascii=False),
        "expected_invalid_fields": json.dumps(expected_invalid, ensure_ascii=False),
        "actual_invalid_fields": json.dumps(actual_invalid, ensure_ascii=False),
        "expected_ambiguity_action": case["expected_ambiguity_action"],
        "explicit_confirmation_required": confirmation,
        "raw_response_json": json.dumps(result, ensure_ascii=False, sort_keys=True),
    }


def run_parse_benchmark(
    cases: list[dict[str, Any]], output_path: Path, *, resume: bool
) -> list[dict[str, Any]]:
    rows, completed = load_existing(output_path) if resume else ([], set())
    for case in tqdm(
        cases,
        total=len(cases),
        desc="Step32 parse benchmark",
        unit="case",
        dynamic_ncols=True,
    ):
        if case["case_id"] in completed:
            continue
        try:
            status, result, elapsed = api_json(
                "/api/assistant/parse", {"text": case["user_text"]}
            )
            row = score_parse_case(case, status, result, elapsed)
        except Exception as exc:
            row = score_parse_case(
                case,
                599,
                {"error": {"code": type(exc).__name__, "message": str(exc)}},
                0,
            )
            row["error_category"] = "INTERNAL_EVALUATION_ERROR"
        rows.append(row)
        write_rows(output_path, rows, PARSE_RESULT_FIELDS)
    return rows


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def parse_metrics(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [row for row in rows if row["split"] == split]
    fields = [
        "origin_match", "destination_match", "hour_match", "mode_match",
        "objective_match", "max_detour_match", "uncertainty_match",
        "compare_all_match", "missing_fields_match", "invalid_fields_match",
        "ambiguity_action_match", "refusal_match",
    ]
    field_rows = []
    for field in fields:
        correct = sum(bool_value(row[field]) for row in selected)
        field_rows.append({
            "split": split,
            "field": field.removesuffix("_match"),
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected) if selected else 0,
        })
    schema_valid = sum(bool_value(row["schema_valid"]) for row in selected)
    full = sum(bool_value(row["full_record_exact"]) for row in selected)
    missing_tp = missing_fp = missing_fn = 0
    invalid_expected = invalid_detected = 0
    ambiguity_expected = ambiguity_detected = silent_ambiguity = 0
    timeouts = malformed = 0
    for row in selected:
        expected_missing = set(json.loads(row["expected_missing_fields"]))
        actual_missing = set(json.loads(row["actual_missing_fields"]))
        missing_tp += len(expected_missing & actual_missing)
        missing_fp += len(actual_missing - expected_missing)
        missing_fn += len(expected_missing - actual_missing)
        expected_invalid = set(json.loads(row["expected_invalid_fields"]))
        if expected_invalid:
            invalid_expected += 1
            invalid_detected += bool_value(row["invalid_fields_match"])
        if row["expected_ambiguity_action"] == "require_candidate_selection":
            ambiguity_expected += 1
            detected = bool_value(row["ambiguity_action_match"])
            ambiguity_detected += detected
            silent_ambiguity += not detected
        timeouts += row["error_category"] == "MODEL_TIMEOUT"
        malformed += row["error_category"] == "MODEL_SCHEMA_VIOLATION"
    precision = missing_tp / (missing_tp + missing_fp) if missing_tp + missing_fp else 1
    recall = missing_tp / (missing_tp + missing_fn) if missing_tp + missing_fn else 1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    total_field = sum(row["total"] for row in field_rows)
    overall = {
        "split": split,
        "case_count": len(selected),
        "schema_valid_rate": schema_valid / len(selected) if selected else 0,
        "full_record_exact_rate": full / len(selected) if selected else 0,
        "field_level_micro_accuracy": sum(row["correct"] for row in field_rows) / total_field if total_field else 0,
        "field_level_macro_accuracy": sum(row["accuracy"] for row in field_rows) / len(field_rows),
        "missing_field_precision": precision,
        "missing_field_recall": recall,
        "missing_field_f1": f1,
        "invalid_input_detection_rate": invalid_detected / invalid_expected if invalid_expected else 1,
        "ambiguity_detection_rate": ambiguity_detected / ambiguity_expected if ambiguity_expected else 1,
        "silent_ambiguity_selection_count": silent_ambiguity,
        "parse_failure_rate": (len(selected) - schema_valid) / len(selected) if selected else 0,
        "timeout_rate": timeouts / len(selected) if selected else 0,
        "malformed_json_rate": malformed / len(selected) if selected else 0,
    }
    return field_rows, overall


REPEAT_FIELDS = [
    "case_id", "repeat_index", "status", "latency_ms", "json_hash",
    "field_signature", "objective", "missing_fields", "origin_query",
    "destination_query", "output_length", "raw_response_json",
]


def run_repeatability(
    core_cases: list[dict[str, Any]], output_path: Path, *, resume: bool
) -> list[dict[str, Any]]:
    rows, completed = load_existing(output_path, key="run_id") if resume and output_path.exists() else ([], set())
    for case in tqdm(
        core_cases,
        total=len(core_cases),
        desc="Step32 repeatability cases",
        unit="case",
        dynamic_ncols=True,
    ):
        for repeat_index in range(1, 6):
            run_id = f"{case['case_id']}__{repeat_index}"
            if run_id in completed:
                continue
            started = time.perf_counter()
            try:
                status, result, elapsed = api_json(
                    "/api/assistant/parse", {"text": case["user_text"]}
                )
            except Exception as exc:
                status, result, elapsed = 599, {"error": str(exc)}, time.perf_counter() - started
            canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            import hashlib
            signature_keys = (
                "intent", "origin_query", "destination_query", "hour", "mode",
                "objective", "max_detour_ratio", "uncertainty_weight", "missing_fields",
            )
            signature = json.dumps({key: result.get(key) for key in signature_keys}, ensure_ascii=False, sort_keys=True)
            rows.append({
                "run_id": run_id,
                "case_id": case["case_id"],
                "repeat_index": repeat_index,
                "status": "PASS" if status == 200 else "FAIL",
                "latency_ms": round(elapsed * 1000, 3),
                "json_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "field_signature": signature,
                "objective": result.get("objective"),
                "missing_fields": json.dumps(result.get("missing_fields", []), ensure_ascii=False),
                "origin_query": result.get("origin_query"),
                "destination_query": result.get("destination_query"),
                "output_length": len(canonical),
                "raw_response_json": canonical,
            })
            write_rows(output_path, rows, ["run_id", *REPEAT_FIELDS])
    return rows


def repeatability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["case_id"], []).append(row)
    json_consistent = field_consistent = objective_consistent = missing_consistent = place_consistent = 0
    latency_ranges = []
    output_ranges = []
    for values in groups.values():
        json_consistent += len({row["json_hash"] for row in values}) == 1
        field_consistent += len({row["field_signature"] for row in values}) == 1
        objective_consistent += len({row["objective"] for row in values}) == 1
        missing_consistent += len({row["missing_fields"] for row in values}) == 1
        place_consistent += len({(row["origin_query"], row["destination_query"]) for row in values}) == 1
        latencies = [float(row["latency_ms"]) for row in values]
        lengths = [int(row["output_length"]) for row in values]
        latency_ranges.append(max(latencies) - min(latencies))
        output_ranges.append(max(lengths) - min(lengths))
    count = len(groups) or 1
    return {
        "core_case_count": len(groups),
        "actual_model_call_count": len(rows),
        "json_complete_consistency_rate": json_consistent / count,
        "field_consistency_rate": field_consistent / count,
        "objective_consistency_rate": objective_consistent / count,
        "missing_field_consistency_rate": missing_consistent / count,
        "place_text_consistency_rate": place_consistent / count,
        "mean_latency_range_ms": sum(latency_ranges) / count,
        "mean_output_length_range": sum(output_ranges) / count,
    }
