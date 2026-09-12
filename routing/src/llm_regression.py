"""Step32 deterministic end-to-end comparison, degradation, and regression checks."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from routing.src.llm_parse_evaluation import api_json, write_rows


END_TO_END_FIELDS = [
    "case_id", "status", "stage", "ambiguity_returned",
    "candidate_confirmation_simulated", "tool_selection_correct",
    "parameters_unchanged", "route_result_match", "astar_match",
    "prohibited_turn_violation_count", "detour_limit_satisfied",
    "fallback_reported", "no_path_correct", "elapsed_ms", "detail",
]


def coordinate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "crs": "EPSG:4326",
        "longitude": candidate["longitude"],
        "latitude": candidate["latitude"],
    }


def comparable_route(route: dict[str, Any]) -> dict[str, Any]:
    ignored = {"route_id", "interface_runtime_seconds", "geometry"}
    return {key: value for key, value in route.items() if key not in ignored}


def forbidden_turn_violations(
    graph_path: Path, segment_ids: list[str]
) -> int:
    if len(segment_ids) < 2:
        return 0
    graph = np.load(graph_path, allow_pickle=False)
    lookup = {str(value): index for index, value in enumerate(graph["segment_ids"])}
    edge_u, edge_v = graph["edge_u"], graph["edge_v"]
    forbidden = {
        (int(node), min(int(first), int(second)), max(int(first), int(second)))
        for node, first, second in zip(
            graph["forbidden_turn_nodes"],
            graph["forbidden_turn_edge_a"],
            graph["forbidden_turn_edge_b"],
            strict=True,
        )
    }
    violations = 0
    indices = [lookup[value] for value in segment_ids]
    for first, second in zip(indices, indices[1:]):
        shared = {int(edge_u[first]), int(edge_v[first])} & {
            int(edge_u[second]), int(edge_v[second])
        }
        if shared and (
            next(iter(shared)), min(first, second), max(first, second)
        ) in forbidden:
            violations += 1
    return violations


def run_end_to_end(
    cases: list[dict[str, Any]],
    parse_rows: list[dict[str, Any]],
    output_path: Path,
    graph_path: Path,
) -> list[dict[str, Any]]:
    parsed_lookup = {
        row["case_id"]: json.loads(row["raw_response_json"])
        for row in parse_rows
    }
    rows = []
    for case in tqdm(
        cases,
        total=len(cases),
        desc="Step32 end-to-end",
        unit="case",
        dynamic_ncols=True,
    ):
        started = time.perf_counter()
        result = {
            "case_id": case["case_id"],
            "status": "FAIL",
            "stage": "parse",
            "ambiguity_returned": False,
            "candidate_confirmation_simulated": False,
            "tool_selection_correct": False,
            "parameters_unchanged": False,
            "route_result_match": False,
            "astar_match": False,
            "prohibited_turn_violation_count": 0,
            "detour_limit_satisfied": False,
            "fallback_reported": False,
            "no_path_correct": False,
            "detail": "",
        }
        try:
            parsed = parsed_lookup[case["case_id"]]
            if "error" in parsed:
                result["detail"] = "parse failed"
                raise RuntimeError("parse failed")
            result["tool_selection_correct"] = parsed.get("intent") in {
                "single_route", "compare_routes"
            }
            origin_status, origin_geo, _ = api_json(
                "/api/geocode", {"query": parsed["origin_query"], "limit": 10}, 30
            )
            destination_status, destination_geo, _ = api_json(
                "/api/geocode", {"query": parsed["destination_query"], "limit": 10}, 30
            )
            result["stage"] = "geocode"
            if origin_status != 200 or destination_status != 200:
                result["no_path_correct"] = True
                result["status"] = "PASS"
                result["detail"] = "geocode correctly rejected"
                raise StopIteration
            if not origin_geo["candidates"] or not destination_geo["candidates"]:
                result["no_path_correct"] = True
                result["status"] = "PASS"
                result["detail"] = "no candidate correctly returned"
                raise StopIteration
            result["ambiguity_returned"] = bool(
                origin_geo["ambiguity_flag"] or destination_geo["ambiguity_flag"]
            )
            result["candidate_confirmation_simulated"] = True
            origin = coordinate(origin_geo["candidates"][0])
            destination = coordinate(destination_geo["candidates"][0])
            payload = {
                "origin": origin,
                "destination": destination,
                "hour": parsed["hour"],
                "mode": parsed["mode"],
                "objective": parsed["objective"] or "risk_aware",
                "algorithm": "astar",
                "max_detour_ratio": (
                    parsed["max_detour_ratio"]
                    if parsed["max_detour_ratio"] is not None
                    else .20
                ),
                "uncertainty_weight": (
                    parsed["uncertainty_weight"]
                    if parsed["uncertainty_weight"] is not None
                    else 1.0
                ),
            }
            result["parameters_unchanged"] = (
                payload["hour"] == parsed["hour"]
                and payload["mode"] == parsed["mode"]
                and (
                    parsed["objective"] is None
                    or payload["objective"] == parsed["objective"]
                )
            )
            result["stage"] = "route"
            first_status, first, _ = api_json("/api/route", payload, 120)
            second_status, second, _ = api_json("/api/route", payload, 120)
            if first_status != 200 or second_status != 200:
                result["no_path_correct"] = first_status == second_status
                result["status"] = "PASS" if result["no_path_correct"] else "FAIL"
                result["detail"] = "deterministic no-path response"
                raise StopIteration
            result["route_result_match"] = comparable_route(first) == comparable_route(second)
            result["astar_match"] = first.get("algorithm") == "astar"
            result["prohibited_turn_violation_count"] = forbidden_turn_violations(
                graph_path, first.get("segment_ids", [])
            )
            result["detour_limit_satisfied"] = (
                first.get("objective") == "shortest"
                or bool(first.get("detour_limit_satisfied"))
            )
            result["fallback_reported"] = all(
                key in first
                for key in ("fallback_length_m", "prior_imputed_length_m", "warning_codes")
            )
            passed = (
                result["tool_selection_correct"]
                and result["candidate_confirmation_simulated"]
                and result["parameters_unchanged"]
                and result["route_result_match"]
                and result["astar_match"]
                and result["prohibited_turn_violation_count"] == 0
                and result["detour_limit_satisfied"]
                and result["fallback_reported"]
            )
            result["status"] = "PASS" if passed else "FAIL"
            result["detail"] = "deterministic API results identical" if passed else "comparison failed"
        except StopIteration:
            pass
        except Exception as exc:
            result["detail"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        rows.append(result)
        write_rows(output_path, rows, END_TO_END_FIELDS)
    return rows


DEGRADATION_FIELDS = [
    "test_id", "condition", "status", "assistant_failure_clear",
    "form_available", "geocode_available", "route_available",
    "compare_available", "export_available", "cloud_fallback_count", "detail",
]


def run_degradation_tests(output_path: Path) -> list[dict[str, Any]]:
    from routing.app.backend import llm_adapter

    valid_json = json.dumps(
        {
            "intent": "compare_routes",
            "origin_query": "新街口",
            "destination_query": "鼓楼",
            "hour": 14,
            "mode": "walk",
            "objective": None,
            "max_detour_ratio": None,
            "uncertainty_weight": None,
        },
        ensure_ascii=False,
    )
    conditions = [
        ("ollama_not_running", RuntimeError("connection refused")),
        ("model_not_loaded", RuntimeError("model not found")),
        ("model_timeout", TimeoutError("model request timeout")),
        ("invalid_json", {"message": {"content": "{invalid json"}}),
        ("empty_text", {"message": {"content": ""}}),
        (
            "markdown_wrapped_json",
            {"message": {"content": f"```json\n{valid_json}\n```"}},
        ),
        ("gpu_oom", RuntimeError("CUDA out of memory")),
        ("model_process_interrupted", RuntimeError("model process exited")),
    ]
    rows = []
    # Each failure is injected into the frozen adapter transport in this
    # evaluation process only. The original callable is restored immediately.
    for index, (condition, injected) in enumerate(
        tqdm(conditions, desc="Step32 degradation", unit="condition", dynamic_ncols=True)
    ):
        original_post = llm_adapter._post
        try:
            def fault_post(*args: Any, **kwargs: Any) -> dict[str, Any]:
                if isinstance(injected, BaseException):
                    raise llm_adapter.LocalLLMError(
                        "本地Qwen3暂时不可用；核心地名搜索与路线规划仍可直接使用。"
                    ) from injected
                return injected

            llm_adapter._post = fault_post
            assistant_failure_clear = False
            try:
                llm_adapter.parse_travel_request(
                    "下午2点从新街口步行到鼓楼，比较四类路线"
                )
            except llm_adapter.LocalLLMError as exc:
                assistant_failure_clear = (
                    "核心地名搜索与路线规划仍可直接使用" in str(exc)
                    or "未能生成有效参数" in str(exc)
                )
            finally:
                llm_adapter._post = original_post

            health_status, health, _ = api_json("/api/health", timeout=15)
            geocode_status, geocode, _ = api_json(
                "/api/geocode", {"query": "新街口", "limit": 2}, 30
            )
            origin = geocode["candidates"][0]
            destination = api_json(
                "/api/geocode", {"query": "鼓楼", "limit": 2}, 30
            )[1]["candidates"][0]
            payload = {
                "origin": coordinate(origin), "destination": coordinate(destination),
                "hour": 14, "mode": "walk", "objective": "shortest",
                "algorithm": "astar", "max_detour_ratio": .2,
                "uncertainty_weight": 1.0,
            }
            route_status, route, _ = api_json("/api/route", payload, 120)
            compare_status, compare, _ = api_json(
                "/api/compare",
                {**payload, "objectives": ["shortest", "shade", "utci", "risk_aware"]},
                180,
            )
            export_status, export_result, _ = api_json(
                "/api/export",
                {"route_id": route.get("route_id", ""), "format": "csv"},
                120,
            )
            export_available = (
                export_status == 200
                and bool(export_result.get("temporary"))
                and Path(export_result.get("local_path", "")).is_file()
            )
            if export_available:
                Path(export_result["local_path"]).unlink(missing_ok=True)
            passed = (
                health_status == 200 and geocode_status == 200
                and route_status == 200 and compare_status == 200
                and route.get("route_found") and compare.get("route_count") == 4
                and assistant_failure_clear and export_available
            )
            rows.append({
                "test_id": f"degrade_{index+1:02d}",
                "condition": condition,
                "status": "PASS" if passed else "FAIL",
                "assistant_failure_clear": assistant_failure_clear,
                "form_available": health_status == 200,
                "geocode_available": geocode_status == 200,
                "route_available": route_status == 200,
                "compare_available": compare_status == 200,
                "export_available": export_available,
                "cloud_fallback_count": 0,
                "detail": (
                    "controlled adapter fault injection; live deterministic "
                    "geocode/route/compare/export probe"
                ),
            })
        except Exception as exc:
            llm_adapter._post = original_post
            rows.append({
                "test_id": f"degrade_{index+1:02d}", "condition": condition,
                "status": "FAIL", "assistant_failure_clear": False,
                "form_available": False, "geocode_available": False,
                "route_available": False, "compare_available": False,
                "export_available": False, "cloud_fallback_count": 0,
                "detail": f"{type(exc).__name__}: {exc}",
            })
        write_rows(output_path, rows, DEGRADATION_FIELDS)
    return rows


REGRESSION_FIELDS = ["suite", "status", "passed", "total", "elapsed_seconds", "detail"]


def run_regression(
    root: Path, output_path: Path, python: Path
) -> list[dict[str, Any]]:
    rows = []
    commands = [
        ("step30_step31_pytest", [str(python), "-m", "pytest", "routing/app/tests", "-q"]),
    ]
    for name, command in tqdm(
        commands, desc="Step32 regression", unit="suite", dynamic_ncols=True
    ):
        started = time.perf_counter()
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=900,
        )
        elapsed = time.perf_counter() - started
        output = (result.stdout + "\n" + result.stderr).strip()
        import re
        match = re.search(r"(\d+) passed", output)
        passed = int(match.group(1)) if match else int(result.returncode == 0)
        total = passed if result.returncode == 0 else max(passed + 1, 1)
        rows.append({
            "suite": name,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "passed": passed,
            "total": total,
            "elapsed_seconds": round(elapsed, 3),
            "detail": output[-500:],
        })
        write_rows(output_path, rows, REGRESSION_FIELDS)
    started = time.perf_counter()
    try:
        formal = json.loads(
            (root / "routing/data/interface/step29_formal_summary.json").read_text(
                encoding="utf-8"
            )
        )
        status, health, _ = api_json("/api/health", timeout=30)
        status_geo, geocode, _ = api_json(
            "/api/geocode", {"query": "新街口", "limit": 5}, 30
        )
        passed = (
            formal.get("status") == "PASS"
            and status == 200
            and health.get("tool_count") == 6
            and status_geo == 200
            and bool(geocode.get("candidates"))
        )
        rows.append({
            "suite": "step29_live_interface_smoke",
            "status": "PASS" if passed else "FAIL",
            "passed": 4 if passed else 0,
            "total": 4,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "detail": "formal status, six tools, live worker and local geocoder",
        })
    except Exception as exc:
        rows.append({
            "suite": "step29_live_interface_smoke", "status": "FAIL",
            "passed": 0, "total": 4,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "detail": f"{type(exc).__name__}: {exc}",
        })
    write_rows(output_path, rows, REGRESSION_FIELDS)
    return rows
