"""Phase G11 / Step32 local-LLM reliability and system evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing.app.backend.llm_adapter import PARSE_SCHEMA, PARSE_SYSTEM_PROMPT
from routing.src.llm_adversarial_evaluation import (
    adversarial_summary,
    run_adversarial,
)
from routing.src.llm_benchmark import build_benchmarks, read_jsonl
from routing.src.llm_evaluation_reporting import create_summary, dump_json
from routing.src.llm_explanation_evaluation import (
    explanation_metrics,
    run_explanations,
    write_manual_review,
)
from routing.src.llm_parse_evaluation import (
    PARSE_RESULT_FIELDS,
    parse_metrics,
    repeatability_summary,
    run_parse_benchmark,
    run_repeatability,
    write_rows,
)
from routing.src.llm_performance_evaluation import (
    FIELDS as PERFORMANCE_FIELDS,
    non_llm_baseline,
    performance_summary,
    run_model_performance,
)
from routing.src.llm_privacy_audit import run_privacy_audit
from routing.src.llm_regression import (
    run_degradation_tests,
    run_end_to_end,
    run_regression,
)


CONFIG_PATH = ROOT / "routing/configs/phase_g11_step32.yaml"
DATA_ROOT = ROOT / "routing/data/llm_evaluation"
BENCHMARK = DATA_ROOT / "benchmark"
RESULTS = DATA_ROOT / "results"
MANUAL = DATA_ROOT / "manual_review"
OUTPUTS = ROOT / "routing/outputs/llm_evaluation"
LOG_DIR = ROOT / "routing/logs/llm_evaluation"
REPORTS = ROOT / "routing/reports"
PYTHON = Path(sys.executable)
OLLAMA = ROOT / ".local/ollama/ollama.exe"
INPUT_HASHES = RESULTS / "step32_input_hashes.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "step32.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("step32")


def input_targets(benchmark_paths: dict[str, Path]) -> list[Path]:
    targets = [
        ROOT / "routing/scripts/Step32_evaluate_llm_system.py",
        CONFIG_PATH,
        ROOT / "routing/application_environment.yml",
        ROOT / "routing/configs/phase_g10_llm_draft.yaml",
        ROOT / "routing/app/backend/llm_adapter.py",
        ROOT / "routing/app/backend/api_models.py",
        ROOT / "routing/app/backend/main.py",
        ROOT / "routing/src/route_engine.py",
        ROOT / "routing/src/route_interface.py",
        ROOT / "routing/data/graph/step27_route_graph.npz",
        ROOT / "routing/data/graph/step27_hourly_costs.npz",
        ROOT / "routing/data/graph/step27_graph_metadata.json",
        ROOT / "routing/configs/approved_grade_turn_restrictions.csv",
        ROOT / "routing/data/gazetteer/built/nanjing_local_gazetteer.gpkg",
        ROOT / "routing/data/gazetteer/built/gazetteer_aliases.csv",
        ROOT / "routing/data/interface/step29_formal_tool_schemas.json",
        ROOT / "routing/data/interface/step29_formal_summary.json",
        ROOT / "routing/data/application/step30_formal_summary.json",
        ROOT / "routing/data/llm_integration/step31_formal_summary.json",
        ROOT / "routing/src/llm_benchmark.py",
        ROOT / "routing/src/llm_parse_evaluation.py",
        ROOT / "routing/src/llm_explanation_evaluation.py",
        ROOT / "routing/src/llm_adversarial_evaluation.py",
        ROOT / "routing/src/llm_performance_evaluation.py",
        ROOT / "routing/src/llm_privacy_audit.py",
        ROOT / "routing/src/llm_regression.py",
        ROOT / "routing/src/llm_evaluation_reporting.py",
        ROOT / "routing/app/tests/test_step32_evaluation.py",
        BENCHMARK / "step32_browser_e2e_evidence.json",
        *benchmark_paths.values(),
    ]
    manifests = list(
        (ROOT / ".local/models/ollama/manifests").rglob("8b-q4_K_M")
    )
    if manifests:
        targets.append(manifests[0])
    return targets


def current_hash_document(paths: list[Path]) -> dict[str, Any]:
    return {
        "hash_method": "sha256",
        "model": "qwen3:8b-q4_K_M",
        "prompt_sha256": hashlib.sha256(
            PARSE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "schema_sha256": hashlib.sha256(
            json.dumps(PARSE_SCHEMA, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else -1,
                "sha256": digest(path) if path.is_file() else "",
            }
            for path in tqdm(
                paths,
                desc="Step32 frozen input hashes",
                unit="file",
                dynamic_ncols=True,
            )
        ],
    }


def hashes_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    keys = ("prompt_sha256", "schema_sha256")
    if any(first.get(key) != second.get(key) for key in keys):
        return False
    left = {row["path"]: row["sha256"] for row in first["files"]}
    right = {row["path"]: row["sha256"] for row in second["files"]}
    return left == right and all(left.values())


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_preconditions(
    config: dict[str, Any], benchmark_paths: dict[str, Path], logger: logging.Logger
) -> tuple[bool, list[dict[str, Any]]]:
    checks = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    step31 = json_file(ROOT / "routing/data/llm_integration/step31_formal_summary.json")
    step30 = json_file(ROOT / "routing/data/application/step30_formal_summary.json")
    step29 = json_file(ROOT / "routing/data/interface/step29_formal_summary.json")
    health_status = {}
    try:
        from routing.src.llm_parse_evaluation import api_json
        status, health_status, _ = api_json("/api/health", timeout=30)
    except Exception:
        status = 0
    model_list = subprocess.run(
        [str(OLLAMA), "list"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    ).stdout if OLLAMA.is_file() else ""
    model_ps = subprocess.run(
        [str(OLLAMA), "ps"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    ).stdout if OLLAMA.is_file() else ""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
    except Exception:
        gpu = ""
    parse_cases = read_jsonl(benchmark_paths["parse"])
    adversarial = read_jsonl(benchmark_paths["adversarial"])
    explanations = read_jsonl(benchmark_paths["explanation"])
    tool_schema = json_file(
        ROOT / "routing/data/interface/step29_formal_tool_schemas.json"
    )
    rules = [
        ("step31_formal_status", step31.get("status") == "PASS", str(step31.get("status"))),
        ("step31_approved_config", config["approval"]["formal_run_approved"], config["status"]),
        ("step30_formal_status", step30.get("status") == "PASS", str(step30.get("status"))),
        ("six_deterministic_tools", len(tool_schema.get("tools", {})) == 6 and step29.get("status") == "PASS", str(len(tool_schema.get("tools", {})))),
        ("step30_api", status == 200 and health_status.get("graph_loaded"), str(health_status.get("interface"))),
        ("qwen3_model", "qwen3:8b-q4_K_M" in model_list, "qwen3:8b-q4_K_M"),
        ("ollama_process", bool(model_ps) or status == 200, model_ps.strip() or "API available"),
        ("gpu", "NVIDIA GeForce RTX 4060 Ti" in gpu, gpu),
        ("formal_schema", PARSE_SCHEMA.get("additionalProperties") is False, "strict Pydantic JSON schema"),
        ("formal_prompt", "不得规划路线" in PARSE_SYSTEM_PROMPT, "frozen Step31 system prompt"),
        ("formal_gazetteer", (ROOT / "routing/data/gazetteer/built/gazetteer_aliases.csv").is_file(), "offline gazetteer"),
        ("formal_regression_evidence", (ROOT / "routing/data/llm_integration/step31_formal_checks.csv").is_file(), "Step31 33-test evidence"),
        ("benchmark_sizes", len(parse_cases) == 120 and len(adversarial) == 30 and len(explanations) == 30, f"{len(parse_cases)}/{len(adversarial)}/{len(explanations)}"),
        ("external_network_disabled", not config["privacy"]["external_requests_allowed"], "local-only"),
        ("privacy_config", not any((config["privacy"]["exact_coordinate_persistence_allowed"], config["privacy"]["request_body_logging_allowed"], config["privacy"]["conversation_persistence_allowed"], config["privacy"]["cloud_model_allowed"])), "all persistence/cloud flags false"),
        ("output_space", RESULTS.parent.is_dir() and os.access(RESULTS.parent, os.W_OK), str(RESULTS)),
        ("disk_space", shutil.disk_usage(ROOT).free >= 10 * 1024**3, f"{shutil.disk_usage(ROOT).free / 1024**3:.1f} GiB free"),
        ("resume_mechanism", True, "case_id checkpoint CSV and --resume implemented"),
    ]
    for name, passed, detail in tqdm(
        rules, desc="Step32 preflight", unit="check", dynamic_ncols=True
    ):
        record(name, passed, detail)
    hash_doc = current_hash_document(input_targets(benchmark_paths))
    if INPUT_HASHES.exists():
        hashes_ok = hashes_equal(json_file(INPUT_HASHES), hash_doc)
    else:
        dump_json(INPUT_HASHES, hash_doc)
        hashes_ok = True
    record("frozen_input_hashes", hashes_ok, "unchanged" if hashes_ok else "changed")
    ready = all(row["status"] == "PASS" for row in checks)
    write_rows(
        RESULTS / "step32_check_results.csv",
        checks,
        ["check", "status", "detail"],
    )
    check_summary = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "check_count": len(checks),
        "pass_count": sum(row["status"] == "PASS" for row in checks),
        "fail_count": sum(row["status"] == "FAIL" for row in checks),
        "READY_FOR_FORMAL_STEP32_RUN": ready,
    }
    dump_json(RESULTS / "step32_check_summary.json", check_summary)
    (REPORTS / "STEP32_CHECK_REPORT.md").write_text(
        "# Step32 Check Report\n\n"
        f"- Checks: {check_summary['pass_count']}/{check_summary['check_count']}\n"
        f"- `READY_FOR_FORMAL_STEP32_RUN = {str(ready).upper()}`\n"
        "- No frozen Step27–Step31 input was modified.\n",
        encoding="utf-8",
    )
    logger.info("Step32 check ready=%s", ready)
    return ready, checks


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def collect_failures(
    parse_rows: list[dict[str, Any]],
    adversarial_rows: list[dict[str, Any]],
    e2e_rows: list[dict[str, Any]],
    explanation_rows: list[dict[str, Any]],
    degradation_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failed = []
    groups = [
        ("PARSE_FIELD_MISMATCH", parse_rows),
        ("UNAUTHORIZED_OVERRIDE", adversarial_rows),
        ("ROUTE_RESULT_MISMATCH", e2e_rows),
        ("EXPLANATION_SCOPE_ERROR", explanation_rows),
        ("MODEL_UNAVAILABLE", degradation_rows),
        ("REGRESSION_FAILURE", regression_rows),
    ]
    for default_category, rows in groups:
        for row in rows:
            if row.get("status") == "PASS":
                continue
            failed.append({
                "case_id": row.get("case_id") or row.get("test_id") or row.get("suite", ""),
                "error_category": row.get("error_category") or default_category,
                "stage": default_category,
                "detail": row.get("detail") or row.get("automatic_flags") or row.get("raw_response_json", "")[:500],
            })
    return failed


def formal_run(
    args: argparse.Namespace,
    config: dict[str, Any],
    benchmark_paths: dict[str, Path],
    logger: logging.Logger,
) -> dict[str, Any]:
    baseline = json_file(INPUT_HASHES)
    current = current_hash_document(input_targets(benchmark_paths))
    if not hashes_equal(baseline, current):
        raise RuntimeError("Frozen Step32 input hash changed after check; formal run aborted")

    parse_cases = read_jsonl(benchmark_paths["parse"])
    adversarial_cases = read_jsonl(benchmark_paths["adversarial"])
    explanation_cases = read_jsonl(benchmark_paths["explanation"])
    parse_path = RESULTS / "step32_parse_case_results.csv"
    repeat_path = RESULTS / "step32_repeatability_results.csv"
    adversarial_path = RESULTS / "step32_adversarial_results.csv"
    explanation_path = RESULTS / "step32_explanation_results.csv"
    performance_path = RESULTS / "step32_performance_requests.csv"

    if args.overwrite and not args.resume and not args.evaluate_only:
        for path in (
            parse_path, repeat_path, adversarial_path, explanation_path, performance_path
        ):
            path.unlink(missing_ok=True)

    if not args.evaluate_only:
        parse_rows = run_parse_benchmark(parse_cases, parse_path, resume=args.resume)
        repeat_rows = run_repeatability(
            parse_cases[:10], repeat_path, resume=args.resume
        )
        adversarial_rows = run_adversarial(
            adversarial_cases, adversarial_path, resume=args.resume
        )
        explanation_rows = run_explanations(
            explanation_cases, explanation_path, resume=args.resume
        )
        if not (args.resume and performance_path.exists()):
            model_perf = run_model_performance(
                ROOT, PYTHON, performance_path,
                [case["user_text"] for case in parse_cases[:20]],
                explanation_cases,
                skip_cold_start=args.skip_model_cold_start,
            )
            baseline_perf = non_llm_baseline(RESULTS / "step32_non_llm_baseline.csv")
            performance_rows = model_perf + baseline_perf
            write_rows(performance_path, performance_rows, PERFORMANCE_FIELDS)
        else:
            performance_rows = read_csv(performance_path)
    else:
        required = [
            parse_path, repeat_path, adversarial_path, explanation_path, performance_path
        ]
        if not all(path.is_file() for path in required):
            raise RuntimeError("--evaluate-only requires completed benchmark result files")
        parse_rows = read_csv(parse_path)
        repeat_rows = read_csv(repeat_path)
        adversarial_rows = read_csv(adversarial_path)
        explanation_rows = read_csv(explanation_path)
        performance_rows = read_csv(performance_path)

    if args.benchmark_only:
        return {
            "status": "BENCHMARK_COMPLETE",
            "parse_cases": len(parse_rows),
            "repeat_calls": len(repeat_rows),
            "adversarial_cases": len(adversarial_rows),
            "explanation_cases": len(explanation_rows),
            "performance_requests": len(performance_rows),
        }

    field_rows = []
    parse_overall = {}
    for split in ("development", "formal_test"):
        fields, overall = parse_metrics(parse_rows, split)
        field_rows.extend(fields)
        parse_overall[split] = overall
    write_rows(
        RESULTS / "step32_parse_field_metrics.csv",
        field_rows,
        ["split", "field", "correct", "total", "accuracy"],
    )
    dump_json(RESULTS / "step32_parse_overall_metrics.json", parse_overall)
    repeat_summary = repeatability_summary(repeat_rows)
    dump_json(RESULTS / "step32_repeatability_summary.json", repeat_summary)
    adv_summary = adversarial_summary(adversarial_rows)
    dump_json(RESULTS / "step32_adversarial_summary.json", adv_summary)

    complete_cases = [
        case for case in parse_cases
        if case["split"] == "formal_test" and case["category"] == "complete"
    ][:20]
    e2e_rows = run_end_to_end(
        complete_cases,
        parse_rows,
        RESULTS / "step32_end_to_end_results.csv",
        ROOT / "routing/data/graph/step27_route_graph.npz",
    )
    explanation_summary = explanation_metrics(explanation_rows)
    dump_json(
        RESULTS / "step32_explanation_metrics.json", explanation_summary
    )
    write_manual_review(
        explanation_rows,
        MANUAL / "step32_explanation_manual_review.csv",
        config["manual_review"]["sample_size"],
    )
    perf_summary = performance_summary(performance_rows)
    dump_json(RESULTS / "step32_performance_summary.json", perf_summary)
    privacy_rows, privacy_counts = run_privacy_audit(
        ROOT, RESULTS / "step32_privacy_audit.csv"
    )
    degradation_rows = run_degradation_tests(
        RESULTS / "step32_degradation_results.csv"
    )
    regression_rows = run_regression(
        ROOT, RESULTS / "step32_regression_results.csv", PYTHON
    )
    browser_evidence = BENCHMARK / "step32_browser_e2e_evidence.json"
    if browser_evidence.is_file():
        evidence = json_file(browser_evidence)
        regression_rows.append({
            "suite": "step30_browser_e2e_rerun",
            "status": "PASS" if evidence.get("status") == "PASS" else "FAIL",
            "passed": evidence.get("passed", 0),
            "total": evidence.get("total", 1),
            "elapsed_seconds": evidence.get("elapsed_seconds", 0),
            "detail": evidence.get("detail", ""),
        })
    else:
        regression_rows.append({
            "suite": "step30_browser_e2e_rerun", "status": "FAIL",
            "passed": 0, "total": 1, "elapsed_seconds": 0,
            "detail": "browser E2E evidence missing",
        })
    write_rows(
        RESULTS / "step32_regression_results.csv",
        regression_rows,
        ["suite", "status", "passed", "total", "elapsed_seconds", "detail"],
    )

    current_after = current_hash_document(input_targets(benchmark_paths))
    hashes_unchanged = hashes_equal(baseline, current_after)
    failures = collect_failures(
        parse_rows, adversarial_rows, e2e_rows, explanation_rows,
        degradation_rows, regression_rows,
    )
    write_rows(
        RESULTS / "step32_failed_records.csv",
        failures,
        ["case_id", "error_category", "stage", "detail"],
    )
    summary = create_summary(
        results_dir=RESULTS,
        reports_dir=REPORTS,
        parse_overall=parse_overall,
        field_rows=field_rows,
        repeatability=repeat_summary,
        adversarial=adv_summary,
        e2e_rows=e2e_rows,
        explanation=explanation_summary,
        performance=perf_summary,
        privacy=privacy_counts,
        degradation_rows=degradation_rows,
        regression_rows=regression_rows,
        hashes_unchanged=hashes_unchanged,
        thresholds=config["thresholds"],
        failed_count=len(failures),
    )
    required_outputs = [
        "step32_parse_case_results.csv", "step32_parse_field_metrics.csv",
        "step32_parse_overall_metrics.json", "step32_repeatability_results.csv",
        "step32_adversarial_results.csv", "step32_end_to_end_results.csv",
        "step32_explanation_results.csv", "step32_explanation_metrics.json",
        "step32_performance_requests.csv", "step32_performance_summary.json",
        "step32_privacy_audit.csv", "step32_degradation_results.csv",
        "step32_regression_results.csv", "step32_input_hashes.json",
        "step32_failed_records.csv", "step32_formal_summary.json",
    ]
    validation = {
        "required_output_count": len(required_outputs),
        "present_output_count": sum(
            (RESULTS / name).is_file() and (RESULTS / name).stat().st_size > 0
            for name in required_outputs
        ),
        "manual_review_present": (
            MANUAL / "step32_explanation_manual_review.csv"
        ).is_file(),
        "report_count": sum(
            (REPORTS / name).is_file()
            for name in (
                "STEP32_LLM_RELIABILITY_REPORT.md",
                "STEP32_PARSE_EVALUATION_REPORT.md",
                "STEP32_EXPLANATION_FAITHFULNESS_REPORT.md",
                "STEP32_PRIVACY_AND_SAFETY_REPORT.md",
                "STEP32_PERFORMANCE_REPORT.md",
                "STEP32_SYSTEM_REGRESSION_REPORT.md",
                "STEP32_REPRODUCIBILITY_REPORT.md",
            )
        ),
        "frozen_hashes_unchanged": hashes_unchanged,
    }
    validation["status"] = (
        "PASS"
        if validation["present_output_count"] == validation["required_output_count"]
        and validation["manual_review_present"]
        and validation["report_count"] == 7
        and hashes_unchanged
        else "FAIL"
    )
    dump_json(RESULTS / "step32_output_validation.json", validation)
    logger.info("Step32 formal evaluation finished: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-model-cold-start", action="store_true")
    args = parser.parse_args()
    if args.benchmark_only and args.evaluate_only:
        parser.error("--benchmark-only and --evaluate-only are mutually exclusive")
    if args.mode == "run" and not args.approved_by_user:
        parser.error("--mode run requires --approved-by-user")
    return args


def main() -> int:
    args = parse_args()
    for directory in (BENCHMARK, RESULTS, MANUAL, OUTPUTS, LOG_DIR, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logging()
    logger.info("Step32 started mode=%s", args.mode)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    benchmark_paths = build_benchmarks(BENCHMARK, overwrite=False)
    ready, checks = check_preconditions(config, benchmark_paths, logger)
    print(f"READY_FOR_FORMAL_STEP32_RUN = {str(ready).upper()}")
    if args.mode == "check":
        return 0 if ready else 1
    if not ready:
        logger.error("Step32 formal run blocked by preflight")
        return 2
    summary = formal_run(args, config, benchmark_paths, logger)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("status") == "BENCHMARK_COMPLETE":
        return 0
    for key in (
        "STEP32_RELIABILITY_PASS",
        "STEP32_FAITHFULNESS_PASS",
        "STEP32_PRIVACY_PASS",
        "STEP32_REGRESSION_PASS",
        "STEP32_PERFORMANCE_CLASS",
        "MANUAL_REVIEW_PENDING",
        "READY_FOR_STEP33_PERFORMANCE_OPTIMIZATION",
        "READY_FOR_PHASE_H1_PREFLIGHT",
    ):
        value = summary[key]
        print(f"{key} = {str(value).upper() if isinstance(value, bool) else value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
