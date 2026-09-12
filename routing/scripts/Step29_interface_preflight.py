"""Phase G8 / Step29 natural-language route interface preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jsonschema import Draft202012Validator
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.src.reporting import atomic_csv, atomic_json, atomic_text  # noqa: E402
from routing.src.route_engine import RouteEngine  # noqa: E402
from routing.src.route_tool_schema import tool_schemas  # noqa: E402
from routing.src.thermal_segment_costs import now_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase G8 interface preflight")
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT
        / "routing/configs/phase_g8_interface_draft.yaml",
    )
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_g_step29")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for handler in (
        logging.FileHandler(
            path, mode="w" if overwrite else "a", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Interface YAML root must be a mapping")
    return value


def example_requests(od: pd.Series) -> list[dict[str, Any]]:
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
    return [
        {
            "example_id": "valid_risk_route",
            "tool": "route",
            "expected_valid": True,
            "payload": {
                "origin": origin,
                "destination": destination,
                "hour": 14,
                "mode": "bike",
                "objective": "risk_aware",
                "algorithm": "astar",
                "max_detour_ratio": 0.20,
                "uncertainty_weight": 1.0,
            },
        },
        {
            "example_id": "valid_compare",
            "tool": "compare_routes",
            "expected_valid": True,
            "payload": {
                "origin": origin,
                "destination": destination,
                "hour": 14,
                "mode": "walk",
                "objectives": [
                    "shortest",
                    "shade",
                    "utci",
                    "risk_aware",
                ],
                "max_detour_ratio": 0.20,
            },
        },
        {
            "example_id": "invalid_hour",
            "tool": "route",
            "expected_valid": False,
            "payload": {
                "origin": origin,
                "destination": destination,
                "hour": 22,
            },
        },
        {
            "example_id": "invalid_mode",
            "tool": "route",
            "expected_valid": False,
            "payload": {
                "origin": origin,
                "destination": destination,
                "hour": 14,
                "mode": "car",
            },
        },
        {
            "example_id": "invalid_detour",
            "tool": "route",
            "expected_valid": False,
            "payload": {
                "origin": origin,
                "destination": destination,
                "hour": 14,
                "max_detour_ratio": -0.1,
            },
        },
    ]


def run_check(
    rules_path: Path, overwrite: bool, logger: logging.Logger
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    rules = load_yaml(rules_path)
    output_dir = PROJECT_ROOT / "routing/data/interface"
    report_path = (
        PROJECT_ROOT / "routing/reports/STEP29_INTERFACE_PREFLIGHT_REPORT.md"
    )
    outputs = {
        "schemas": output_dir / "step29_tool_schemas.json",
        "dependencies": output_dir / "step29_dependency_qc.csv",
        "examples": output_dir / "step29_example_request_qc.csv",
        "smoke": output_dir / "step29_route_tool_smoke_qc.csv",
        "failed": output_dir / "step29_failed_records.csv",
        "summary": output_dir / "step29_check_summary.json",
    }
    for path in [*outputs.values(), report_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Step29 check output exists: {path}")
    dependencies = []
    for name, purpose, required in tqdm(
        [
            ("numpy", "route arrays", True),
            ("pandas", "tables", True),
            ("jsonschema", "tool validation", True),
            ("osgeo", "coordinate transformation and GIS", True),
            ("pydantic", "optional typed models", False),
            ("fastapi", "optional HTTP service", False),
            ("uvicorn", "optional HTTP service", False),
            ("geopy", "optional external geocoder client", False),
            ("rapidfuzz", "optional local gazetteer matching", False),
        ],
        desc="Checking Phase G8 dependencies",
        unit="dependency",
        dynamic_ncols=True,
    ):
        available = importlib.util.find_spec(name) is not None
        dependencies.append(
            {
                "dependency": name,
                "purpose": purpose,
                "required_for_initial_tools": required,
                "available": available,
                "blocking": required and not available,
            }
        )
    formal_step28 = json.loads(
        (
            PROJECT_ROOT
            / "routing/data/evaluation/step28_formal_summary.json"
        ).read_text(encoding="utf-8")
    )
    blockers = []
    warnings = []
    if not formal_step28.get("ready_for_phase_g8"):
        blockers.append("Step28 is not ready for Phase G8")
    if any(row["blocking"] for row in dependencies):
        blockers.append("A required initial tool dependency is missing")
    schema_bundle = tool_schemas()
    Draft202012Validator.check_schema(
        schema_bundle["tools"]["route"]["input_schema"]
    )
    od_samples = pd.read_csv(
        PROJECT_ROOT
        / "routing/outputs/tables/step27_validation_od_samples.csv"
    )
    examples = example_requests(od_samples.iloc[0])
    example_rows = []
    for example in tqdm(
        examples,
        desc="Validating Phase G8 request examples",
        unit="example",
        dynamic_ncols=True,
    ):
        schema = schema_bundle["tools"][example["tool"]]["input_schema"]
        errors = sorted(
            Draft202012Validator(schema).iter_errors(example["payload"]),
            key=lambda error: list(error.path),
        )
        valid = not errors
        example_rows.append(
            {
                "example_id": example["example_id"],
                "tool": example["tool"],
                "expected_valid": example["expected_valid"],
                "actual_valid": valid,
                "expectation_pass": valid == example["expected_valid"],
                "validation_message": " | ".join(
                    error.message for error in errors
                ),
            }
        )
    graph_dir = PROJECT_ROOT / "routing/data/graph"
    engine = RouteEngine(
        graph_dir / "step27_route_graph.npz",
        graph_dir / "step27_hourly_costs.npz",
        graph_dir / "step27_graph_metadata.json",
    )
    od = od_samples.iloc[0]
    origin = (float(od["origin_x"]), float(od["origin_y"]))
    destination = (
        float(od["destination_x"]),
        float(od["destination_y"]),
    )
    smoke_rows = []
    for objective in tqdm(
        ("shortest", "shade", "utci", "risk_aware"),
        desc="Testing Phase G8 route tool",
        unit="objective",
        dynamic_ncols=True,
    ):
        result = engine.route(
            origin,
            destination,
            14,
            mode="walk",
            objective=objective,
            algorithm="astar",
            max_detour_ratio=(
                None if objective == "shortest" else 0.20
            ),
            uncertainty_weight=(
                1.0 if objective == "risk_aware" else 0.0
            ),
        )
        smoke_rows.append(
            {
                "objective": objective,
                "found": result.get("found", False),
                "distance_m": result.get("distance_m"),
                "detour_ratio": result.get("detour_ratio"),
                "detour_limit_satisfied": result.get(
                    "detour_limit_satisfied", objective == "shortest"
                ),
                "fallback_retry": result.get("fallback_retry"),
                "segment_count": len(result.get("segment_ids", [])),
            }
        )
    if not all(row["expectation_pass"] for row in example_rows):
        blockers.append("One or more request-schema examples failed")
    if not all(
        row["found"] and row["detour_limit_satisfied"]
        for row in smoke_rows
    ):
        blockers.append("One or more deterministic route-tool tests failed")
    geocoder_ready = (
        rules["geocoding"]["local_gazetteer_available"]
        or rules["geocoding"]["external_provider"] is not None
    )
    if not geocoder_ready:
        blockers.append(
            "No local gazetteer or approved external geocoding provider"
        )
    warnings.extend(
        [
            "walk, bike and shared currently use the same approved active-transport topology.",
            "FastAPI is absent, but it is not required for the proposed initial Python tool functions and CLI.",
            "The proposed 20% default detour limit and exact-location log policy require approval.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(outputs["schemas"], schema_bundle)
    atomic_csv(outputs["dependencies"], dependencies)
    atomic_csv(outputs["examples"], example_rows)
    atomic_csv(outputs["smoke"], smoke_rows)
    atomic_text(
        outputs["failed"],
        "category,object_id,error_type,error_message\n",
    )
    ended_at = now_text()
    pending = list(rules["approval"]["pending_decisions"])
    summary = {
        "phase": "G8",
        "step": 29,
        "mode": "check",
        "script": Path(__file__).name,
        "script_version": "1.0.0",
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "ready_for_interface_rule_review": len(
            [value for value in blockers if "Step28" in value]
        )
        == 0,
        "ready_for_formal_step29_run": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "pending_decisions": pending,
        "schema_tool_count": len(schema_bundle["tools"]),
        "schema_example_pass_count": sum(
            row["expectation_pass"] for row in example_rows
        ),
        "schema_example_count": len(example_rows),
        "route_smoke_pass_count": sum(
            row["found"] and row["detour_limit_satisfied"]
            for row in smoke_rows
        ),
        "route_smoke_count": len(smoke_rows),
        "geocoder_ready": geocoder_ready,
        "formal_service_started": False,
        "external_request_made": False,
        "frozen_routing_inputs_modified": False,
        "failed_record_count": 0,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "report_path": str(report_path),
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step29 Natural-Language Interface Preflight",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `READY_FOR_INTERFACE_RULE_REVIEW = {str(summary['ready_for_interface_rule_review']).upper()}`",
        "- `READY_FOR_FORMAL_STEP29_RUN = FALSE`",
        f"- Draft tools: {summary['schema_tool_count']}",
        f"- Schema examples passed: {summary['schema_example_pass_count']}/{summary['schema_example_count']}",
        f"- Deterministic route-tool tests passed: {summary['route_smoke_pass_count']}/{summary['route_smoke_count']}",
        f"- Geocoder ready: {geocoder_ready}",
        f"- Blocking decisions/issues: {len(blockers)}",
        "",
        "## Required decisions",
        "",
        "1. Approve or change the proposed default maximum detour ratio of 20%.",
        "2. Choose a local gazetteer or an external geocoding provider and define its key/privacy policy.",
        "3. Approve or change the proposal not to persist exact user locations.",
        "",
        "The initial delivery can use validated Python tool functions and a CLI; FastAPI is not required. No external geocoder was called, no service was started and no frozen route artifact was modified.",
        "",
        "## Blockers",
        "",
        *[f"- {value}" for value in blockers],
        "",
        "## Warnings",
        "",
        *[f"- {value}" for value in warnings],
    ]
    atomic_text(report_path, "\n".join(lines) + "\n")
    logger.info(
        "Step29 check complete: review=%s formal=%s blockers=%s",
        summary["ready_for_interface_rule_review"],
        summary["ready_for_formal_step29_run"],
        len(blockers),
    )
    return summary


def main() -> int:
    args = parse_args()
    logger = setup_logger(
        PROJECT_ROOT / "routing/logs/step29.log", args.overwrite
    )
    failure_path = (
        PROJECT_ROOT / "routing/data/interface/step29_failure.json"
    )
    try:
        if args.mode == "run":
            if not args.approved_by_user:
                raise PermissionError(
                    "Formal Step29 requires --approved-by-user."
                )
            from routing.scripts.Step29_interface_run import run_formal

            summary = run_formal(
                args.rules.resolve(), args.overwrite, logger
            )
            if failure_path.exists():
                failure_path.unlink()
            print(
                json.dumps(
                    {
                        "STATUS": summary["status"],
                        "READY_FOR_PHASE_G9": summary[
                            "ready_for_phase_g9"
                        ],
                        "failed_record_count": summary[
                            "failed_record_count"
                        ],
                        "report_path": summary["report_path"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if summary["ready_for_phase_g9"] else 2
        summary = run_check(args.rules.resolve(), args.overwrite, logger)
        if failure_path.exists():
            failure_path.unlink()
        print(
            json.dumps(
                {
                    "READY_FOR_INTERFACE_RULE_REVIEW": summary[
                        "ready_for_interface_rule_review"
                    ],
                    "READY_FOR_FORMAL_STEP29_RUN": summary[
                        "ready_for_formal_step29_run"
                    ],
                    "blocker_count": summary["blocker_count"],
                    "pending_decisions": summary["pending_decisions"],
                    "report_path": summary["report_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if summary["ready_for_interface_rule_review"] else 2
    except Exception as error:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.exception("Step29 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
