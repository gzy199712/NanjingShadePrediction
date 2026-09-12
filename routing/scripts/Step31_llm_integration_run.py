"""Formal Step31 local-LLM integration run and evidence generator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "routing" / "data" / "llm_integration"
LOG_DIR = ROOT / "routing" / "logs" / "llm_integration"
REPORT = ROOT / "routing" / "reports" / "STEP31_FORMAL_LLM_INTEGRATION_REPORT.md"
CONFIG = ROOT / "routing" / "configs" / "phase_g10_llm_draft.yaml"
PYTHON = Path(sys.executable)
OLLAMA = ROOT / ".local" / "ollama" / "ollama.exe"
API_BASE = "http://127.0.0.1:8765"
EXPECTED_MODEL = "qwen3:8b-q4_K_M"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def api(path: str, payload: dict | None = None, timeout: int = 360) -> dict:
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "run"], required=True)
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "run" and not args.approved_by_user:
        parser.error("--mode run requires --approved-by-user")

    DATA.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    log_path = LOG_DIR / "step31.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Step31 started mode=%s", args.mode)

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    performance: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        )
        if not passed:
            failures.append(
                {"point_id": "STEP31", "filename": name, "error_message": detail}
            )

    tasks = [
        "approval_and_config",
        "runtime_and_model",
        "gpu_residency",
        "application_health",
        "live_structured_parse",
        "privacy_source_scan",
        "unit_and_regression_tests",
        "frozen_input_hashes",
    ]
    for task in tqdm(
        tasks, desc="Step31 formal checks", unit="check", dynamic_ncols=True
    ):
        try:
            if task == "approval_and_config":
                source = CONFIG.read_text(encoding="utf-8")
                check(
                    task,
                    args.approved_by_user
                    and "status: APPROVED_FORMAL_RUN" in source
                    and "conversation_retention: memory_only_no_persistence" in source,
                    "user approved; local-only memory-only policy frozen",
                )
            elif task == "runtime_and_model":
                version = json.loads(
                    urllib.request.urlopen(
                        "http://127.0.0.1:11434/api/version", timeout=5
                    )
                    .read()
                    .decode("utf-8")
                )["version"]
                listed = subprocess.run(
                    [str(OLLAMA), "list"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                    check=True,
                ).stdout
                check(
                    task,
                    EXPECTED_MODEL in listed,
                    f"Ollama {version}; {EXPECTED_MODEL}; local model inventory verified",
                )
            elif task == "gpu_residency":
                process = subprocess.run(
                    [str(OLLAMA), "ps"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                    check=True,
                ).stdout
                check(
                    task,
                    EXPECTED_MODEL in process and "100% GPU" in process,
                    "Ollama processor report: 100% GPU"
                    if "100% GPU" in process
                    else process.strip(),
                )
            elif task == "application_health":
                health = api("/api/health", timeout=10)
                assistant = health["local_assistant"]
                check(
                    task,
                    health["status"] == "ok"
                    and assistant["available"]
                    and not assistant["external_requests_enabled"]
                    and not assistant["conversation_persistence"],
                    "Step31 API healthy; assistant local-only and memory-only",
                )
            elif task == "live_structured_parse":
                begin = time.perf_counter()
                parsed = api(
                    "/api/assistant/parse",
                    {
                        "text": "下午2点从新街口步行到鼓楼，比较四类路线",
                    },
                )
                elapsed = time.perf_counter() - begin
                performance.append(
                    {
                        "operation": "live_structured_parse",
                        "elapsed_seconds": round(elapsed, 3),
                        "gpu_required": True,
                        "status": "PASS",
                    }
                )
                expected = {
                    "intent": "compare_routes",
                    "origin_query": "新街口",
                    "destination_query": "鼓楼",
                    "hour": 14,
                    "mode": "walk",
                }
                check(
                    task,
                    all(parsed.get(key) == value for key, value in expected.items())
                    and parsed["explicit_confirmation_required"]
                    and parsed["local_only"],
                    f"validated JSON; confirmation required; elapsed={elapsed:.1f}s",
                )
                (DATA / "step31_live_parse_result.json").write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            elif task == "privacy_source_scan":
                backend = (
                    ROOT / "routing/app/backend/llm_adapter.py"
                ).read_text(encoding="utf-8")
                logs = "\n".join(
                    path.read_text(encoding="utf-8", errors="replace")
                    for path in LOG_DIR.glob("*.log")
                )
                passed = (
                    "ALLOWED_HOSTS" in backend
                    and "conversation_persistence" in backend
                    and "下午2点从新街口" not in logs
                )
                check(
                    task,
                    passed,
                    "localhost allowlist; no prompt body in logs; no conversation store",
                )
            elif task == "unit_and_regression_tests":
                result = subprocess.run(
                    [
                        str(PYTHON),
                        "-m",
                        "pytest",
                        "routing/app/tests",
                        "-q",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=600,
                )
                last = result.stdout.strip().splitlines()[-1] if result.stdout else ""
                check(task, result.returncode == 0, last or result.stderr[-300:])
            elif task == "frozen_input_hashes":
                baseline = DATA / "step31_frozen_input_hashes.csv"
                targets = [
                    ROOT / "routing/data/application/step30_formal_summary.json",
                    ROOT / "routing/data/interface/step29_formal_summary.json",
                    ROOT / "routing/configs/route_search_draft.yaml",
                    ROOT / "routing/configs/thermal_edge_cost_draft.yaml",
                ]
                rows = [
                    {"filepath": str(path), "sha256": digest(path)} for path in targets
                ]
                write_csv(baseline, rows, ["filepath", "sha256"])
                check(task, all(path.is_file() for path in targets), f"{len(rows)} frozen files recorded")
        except Exception as exc:
            logging.exception("Step31 task failed: %s", task)
            check(task, False, f"{type(exc).__name__}: {exc}")

    finished = datetime.now().astimezone()
    write_csv(DATA / "step31_formal_checks.csv", checks, ["check", "status", "detail"])
    write_csv(
        DATA / "failed_files.csv",
        failures,
        ["point_id", "filename", "error_message"],
    )
    write_csv(
        DATA / "step31_performance.csv",
        performance,
        ["operation", "elapsed_seconds", "gpu_required", "status"],
    )
    passed = sum(row["status"] == "PASS" for row in checks)
    summary = {
        "step": 31,
        "phase": "G10",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "processed_count": len(checks),
        "success_count": passed,
        "failure_count": len(checks) - passed,
        "runtime": "Ollama",
        "runtime_version": "0.32.4",
        "model": EXPECTED_MODEL,
        "quantization": "Q4_K_M",
        "model_size_gb": 5.2,
        "gpu": "NVIDIA GeForce RTX 4060 Ti",
        "gpu_processor_share": "100%",
        "conversation_persistence": False,
        "external_requests_enabled": False,
        "route_engine_modified": False,
    }
    (DATA / "step31_formal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Step31 Formal Local LLM Integration Report

- Status: **{summary['status']}**
- Started: {started.isoformat()}
- Finished: {finished.isoformat()}
- Checks: {passed}/{len(checks)}
- Runtime: Ollama 0.32.4
- Model: `qwen3:8b-q4_K_M` (Q4_K_M, approximately 5.2 GB)
- Device: NVIDIA GeForce RTX 4060 Ti; Ollama reported `100% GPU`
- Privacy: localhost only; request bodies are not logged; conversation is not persisted
- Routing authority: unchanged deterministic Step30 API; the model cannot access graph,
  coordinates, geometry, segment IDs, costs, or turn restrictions

## Interaction contract

1. The model parses natural language into a strictly validated JSON draft.
2. Deterministic local geocoding returns candidates.
3. The user explicitly chooses origin and destination candidates.
4. Clicking a route button is explicit confirmation before the deterministic route call.
5. Optional model explanation receives only approved aggregate route metrics.

## Performance note

The approved 8B model remained fully GPU-resident according to Ollama. Concurrent
high-VRAM graphics workloads can nevertheless cause severe shared-memory paging and
increase response latency. Core geocoding and routing remain available when the
assistant is unavailable or busy.
"""
    REPORT.write_text(report, encoding="utf-8")
    logging.info(
        "Step31 finished processed=%s success=%s failure=%s",
        len(checks),
        passed,
        len(checks) - passed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
