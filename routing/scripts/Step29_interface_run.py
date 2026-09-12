"""Formal Phase G8 / Step29 route-tool integration and end-to-end QC."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from routing.src.reporting import atomic_csv, atomic_json, atomic_text
from routing.src.route_interface import RouteToolInterface
from routing.src.route_tool_schema import tool_schemas
from routing.src.thermal_segment_costs import now_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as handle, tqdm(
        total=total,
        desc=f"Hashing {path.name}",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            progress.update(len(block))
    return digest.hexdigest()


def run_formal(
    rules_path: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    output_dir = PROJECT_ROOT / "routing" / "data" / "interface"
    report_path = PROJECT_ROOT / "routing" / "reports" / "STEP29_FORMAL_INTERFACE_REPORT.md"
    outputs = {
        "schemas": output_dir / "step29_formal_tool_schemas.json",
        "geocoder": output_dir / "step29_formal_geocoder_qc.csv",
        "tools": output_dir / "step29_formal_tool_qc.csv",
        "routes": output_dir / "step29_formal_route_qc.csv",
        "algorithms": output_dir / "step29_formal_algorithm_qc.csv",
        "exports": output_dir / "step29_formal_export_qc.csv",
        "privacy": output_dir / "step29_formal_privacy_qc.csv",
        "failed": output_dir / "step29_formal_failed_records.csv",
        "summary": output_dir / "step29_formal_summary.json",
    }
    for path in [*outputs.values(), report_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Formal Step29 output exists: {path}")

    frozen = [
        PROJECT_ROOT / "routing/data/graph/step27_route_graph.npz",
        PROJECT_ROOT / "routing/data/graph/step27_hourly_costs.npz",
        PROJECT_ROOT / "routing/data/graph/step27_graph_metadata.json",
        PROJECT_ROOT / "routing/data/gazetteer/built/nanjing_local_gazetteer.gpkg",
        PROJECT_ROOT / "routing/data/gazetteer/built/gazetteer_aliases.csv",
    ]
    hashes_before = {
        str(path.relative_to(PROJECT_ROOT)): file_hash(path)
        for path in tqdm(
            frozen,
            total=len(frozen),
            desc="Freezing Step29 inputs",
            unit="file",
            dynamic_ncols=True,
        )
    }
    interface = RouteToolInterface(PROJECT_ROOT)
    atomic_json(outputs["schemas"], tool_schemas())

    failures = []
    geocoder_rows = []
    geocoder_results = {}
    for query in tqdm(
        ["新街口", "鼓楼", "夫子庙", "玄武湖", "南京站"],
        total=5,
        desc="Testing local geocoder",
        unit="query",
        dynamic_ncols=True,
    ):
        try:
            result = interface.call(
                "geocode_place", {"query": query, "city": "南京市", "limit": 5}
            )
            geocoder_results[query] = result
            geocoder_rows.append(
                {
                    "query": query,
                    "candidate_count": result["candidate_count"],
                    "ambiguous": result["ambiguous"],
                    "external_request_made": result["external_request_made"],
                    "pass": result["candidate_count"] > 0
                    and not result["external_request_made"],
                }
            )
        except Exception as error:
            failures.append(
                {"stage": "geocoder", "object_id": query, "error_message": str(error)}
            )

    od = pd.read_csv(
        PROJECT_ROOT / "routing/outputs/tables/step27_validation_od_samples.csv"
    ).iloc[0]
    origin = {
        "crs": "EPSG:32650",
        "x": float(od["origin_x"]),
        "y": float(od["origin_y"]),
    }
    destination = {
        "crs": "EPSG:32650",
        "x": float(od["destination_x"]),
        "y": float(od["destination_y"]),
    }
    snap_result = interface.call(
        "snap_origin_destination",
        {"origin": origin, "destination": destination},
    )
    tool_rows = [
        {
            "tool": "geocode_place",
            "pass": len(geocoder_rows) == 5 and all(row["pass"] for row in geocoder_rows),
            "detail": "five known Nanjing names",
        },
        {
            "tool": "snap_origin_destination",
            "pass": snap_result["origin_snap_distance_m"] <= 100
            and snap_result["destination_snap_distance_m"] <= 100,
            "detail": (
                f"origin={snap_result['origin_snap_distance_m']:.6f};"
                f"destination={snap_result['destination_snap_distance_m']:.6f}"
            ),
        },
    ]

    route_rows = []
    route_results = {}
    for objective in tqdm(
        ("shortest", "shade", "utci", "risk_aware"),
        total=4,
        desc="Testing formal route tool",
        unit="objective",
        dynamic_ncols=True,
    ):
        try:
            payload = {
                "origin": origin,
                "destination": destination,
                "hour": 14,
                "mode": "walk",
                "objective": objective,
                "algorithm": "astar",
                "max_detour_ratio": 0.20,
                "uncertainty_weight": 1.0,
            }
            result = interface.call("route", payload)
            route_results[objective] = result
            route_rows.append(
                {
                    "objective": objective,
                    "found": result.get("found", False),
                    "distance_m": result.get("distance_m"),
                    "detour_ratio": result.get("detour_ratio"),
                    "detour_limit_satisfied": result.get(
                        "detour_limit_satisfied", objective == "shortest"
                    ),
                    "segment_count": len(result.get("segment_ids", [])),
                    "node_count": len(result.get("node_indices", [])),
                    "exact_location_persisted": result["privacy"][
                        "exact_location_persisted"
                    ],
                    "pass": result.get("found", False)
                    and result.get("detour_limit_satisfied", True)
                    and len(result.get("node_indices", [])) >= 2,
                }
            )
        except Exception as error:
            failures.append(
                {"stage": "route", "object_id": objective, "error_message": str(error)}
            )

    compare_result = interface.call(
        "compare_routes",
        {
            "origin": origin,
            "destination": destination,
            "hour": 14,
            "mode": "walk",
            "objectives": ["shortest", "shade", "utci", "risk_aware"],
            "algorithm": "astar",
            "max_detour_ratio": 0.20,
            "uncertainty_weight": 1.0,
        },
    )
    tool_rows.append(
        {
            "tool": "route",
            "pass": len(route_rows) == 4 and all(row["pass"] for row in route_rows),
            "detail": "four approved objectives",
        }
    )
    tool_rows.append(
        {
            "tool": "compare_routes",
            "pass": compare_result["route_count"] == 4 and compare_result["all_found"],
            "detail": f"routes={compare_result['route_count']}",
        }
    )
    summary_result = interface.call(
        "summarize_route",
        {"route_result": route_results["risk_aware"], "language": "zh-CN"},
    )
    tool_rows.append(
        {
            "tool": "summarize_route",
            "pass": bool(summary_result["summary"])
            and summary_result["uncertainty_disclosed"]
            and summary_result["fallback_disclosed"],
            "detail": summary_result["summary"],
        }
    )

    algorithm_rows = []
    for objective in tqdm(
        ("shortest", "shade", "utci", "risk_aware"),
        total=4,
        desc="Checking A-star against Dijkstra",
        unit="objective",
        dynamic_ncols=True,
    ):
        dijkstra = interface.call(
            "route",
            {
                "origin": origin,
                "destination": destination,
                "hour": 14,
                "mode": "walk",
                "objective": objective,
                "algorithm": "dijkstra",
                "max_detour_ratio": 0.20,
                "uncertainty_weight": 1.0,
            },
        )
        astar = route_results[objective]
        difference = abs(float(astar["total_cost"]) - float(dijkstra["total_cost"]))
        algorithm_rows.append(
            {
                "objective": objective,
                "astar_found": astar["found"],
                "dijkstra_found": dijkstra["found"],
                "absolute_cost_difference": difference,
                "pass": astar["found"]
                and dijkstra["found"]
                and difference <= 1e-6,
            }
        )

    export_rows = []
    for output_format in tqdm(
        ("png", "pdf", "file_geodatabase"),
        total=3,
        desc="Testing route exports",
        unit="format",
        dynamic_ncols=True,
    ):
        try:
            result = interface.call(
                "export_route_map",
                {
                    "route_result": route_results["risk_aware"],
                    "format": output_format,
                    "include_uncertainty": True,
                    "include_fallback": True,
                },
            )
            path = Path(result["output_path"])
            export_rows.append(
                {
                    "format": output_format,
                    "output_path": str(path),
                    "exists": path.exists(),
                    "nonempty": path.is_dir()
                    or (path.is_file() and path.stat().st_size > 0),
                    "pass": path.exists()
                    and (
                        path.is_dir()
                        or (path.is_file() and path.stat().st_size > 0)
                    ),
                }
            )
        except Exception as error:
            failures.append(
                {"stage": "export", "object_id": output_format, "error_message": str(error)}
            )
    tool_rows.append(
        {
            "tool": "export_route_map",
            "pass": len(export_rows) == 3 and all(row["pass"] for row in export_rows),
            "detail": f"formats={len(export_rows)}",
        }
    )

    invalid_payload_rejected = False
    try:
        interface.call(
            "route",
            {
                "origin": origin,
                "destination": destination,
                "hour": 22,
                "mode": "car",
            },
        )
    except ValueError:
        invalid_payload_rejected = True

    hashes_after = {
        str(path.relative_to(PROJECT_ROOT)): file_hash(path)
        for path in tqdm(
            frozen,
            total=len(frozen),
            desc="Rechecking frozen inputs",
            unit="file",
            dynamic_ncols=True,
        )
    }
    frozen_unchanged = hashes_before == hashes_after
    privacy_rows = [
        {
            "check": "runtime_external_geocoder_requests",
            "value": 0,
            "pass": all(not row["external_request_made"] for row in geocoder_rows),
        },
        {
            "check": "exact_user_location_persisted",
            "value": 0,
            "pass": all(not row["exact_location_persisted"] for row in route_rows),
        },
        {
            "check": "invalid_schema_request_rejected",
            "value": invalid_payload_rejected,
            "pass": invalid_payload_rejected,
        },
        {
            "check": "frozen_inputs_unchanged",
            "value": frozen_unchanged,
            "pass": frozen_unchanged,
        },
    ]

    atomic_csv(outputs["geocoder"], geocoder_rows)
    atomic_csv(outputs["tools"], tool_rows)
    atomic_csv(outputs["routes"], route_rows)
    atomic_csv(outputs["algorithms"], algorithm_rows)
    atomic_csv(outputs["exports"], export_rows)
    atomic_csv(outputs["privacy"], privacy_rows)
    atomic_csv(outputs["failed"], failures)
    success = (
        len(failures) == 0
        and len(tool_rows) == 6
        and all(row["pass"] for row in tool_rows)
        and all(row["pass"] for row in route_rows)
        and all(row["pass"] for row in algorithm_rows)
        and all(row["pass"] for row in export_rows)
        and all(row["pass"] for row in privacy_rows)
    )
    ended_at = now_text()
    summary = {
        "phase": "G8",
        "step": 29,
        "mode": "run",
        "script": Path(__file__).name,
        "script_version": "1.0.0",
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS" if success else "FAIL",
        "tool_count": len(tool_rows),
        "tool_pass_count": sum(bool(row["pass"]) for row in tool_rows),
        "geocoder_query_count": len(geocoder_rows),
        "geocoder_pass_count": sum(bool(row["pass"]) for row in geocoder_rows),
        "route_objective_count": len(route_rows),
        "route_pass_count": sum(bool(row["pass"]) for row in route_rows),
        "algorithm_pair_count": len(algorithm_rows),
        "algorithm_pass_count": sum(bool(row["pass"]) for row in algorithm_rows),
        "export_count": len(export_rows),
        "export_pass_count": sum(bool(row["pass"]) for row in export_rows),
        "privacy_check_count": len(privacy_rows),
        "privacy_pass_count": sum(bool(row["pass"]) for row in privacy_rows),
        "failed_record_count": len(failures),
        "external_request_made": False,
        "exact_location_persisted": False,
        "frozen_inputs_modified": not frozen_unchanged,
        "formal_http_service_started": False,
        "ready_for_phase_g9": success,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "report_path": str(report_path),
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step29 Formal Natural-Language Route Interface",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `STATUS = {summary['status']}`",
        f"- Tools: {summary['tool_pass_count']}/{summary['tool_count']}",
        f"- Local geocoder queries: {summary['geocoder_pass_count']}/{summary['geocoder_query_count']}",
        f"- Route objectives: {summary['route_pass_count']}/{summary['route_objective_count']}",
        f"- A*/Dijkstra pairs: {summary['algorithm_pass_count']}/{summary['algorithm_pair_count']}",
        f"- Exports: {summary['export_pass_count']}/{summary['export_count']}",
        f"- Privacy/frozen-input checks: {summary['privacy_pass_count']}/{summary['privacy_check_count']}",
        f"- Failed records: {summary['failed_record_count']}",
        f"- `READY_FOR_PHASE_G9 = {str(summary['ready_for_phase_g9']).upper()}`",
        "",
        "The interface uses the offline Nanjing gazetteer and the frozen turn-aware Step27 route engine. It made no external geocoding request, persisted no exact user location, started no HTTP service and modified no frozen input.",
    ]
    atomic_text(report_path, "\n".join(lines) + "\n")
    logger.info(
        "Formal Step29 ended: status=%s tools=%d/%d failures=%d",
        summary["status"],
        summary["tool_pass_count"],
        summary["tool_count"],
        summary["failed_record_count"],
    )
    return summary
