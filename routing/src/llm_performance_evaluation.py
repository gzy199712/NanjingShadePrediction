"""Local hardware performance measurements for the frozen Step31 configuration."""

from __future__ import annotations

import concurrent.futures
import json
import math
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm

from routing.app.backend.llm_adapter import (
    OLLAMA_MODEL,
    PARSE_SCHEMA,
    PARSE_SYSTEM_PROMPT,
)
from routing.src.llm_parse_evaluation import api_json, write_rows


FIELDS = [
    "request_id", "operation", "concurrency", "cold_start", "status",
    "latency_ms", "load_ms", "prompt_tokens", "completion_tokens",
    "tokens_per_second", "gpu_utilization_percent", "gpu_memory_used_mib",
    "system_memory_used_mib", "timeout", "detail",
]


def gpu_stats() -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().split(",")
        return float(result[0]), float(result[1])
    except Exception:
        return None, None


def memory_used_mib() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().used / 1024 / 1024
    except Exception:
        return None


def direct_parse(text: str, timeout: int = 360) -> tuple[dict[str, Any], float]:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "format": PARSE_SCHEMA,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 40,
            "num_ctx": 2048,
            "num_predict": 180,
            "seed": 31,
        },
        "messages": [
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data, time.perf_counter() - started


def row_from_ollama(
    request_id: str, operation: str, concurrency: int, cold: bool,
    response: dict[str, Any] | None, elapsed: float, error: str = "",
) -> dict[str, Any]:
    gpu, gpu_memory = gpu_stats()
    completion = int((response or {}).get("eval_count", 0))
    eval_seconds = float((response or {}).get("eval_duration", 0)) / 1e9
    return {
        "request_id": request_id,
        "operation": operation,
        "concurrency": concurrency,
        "cold_start": cold,
        "status": "PASS" if response is not None else "FAIL",
        "latency_ms": round(elapsed * 1000, 3),
        "load_ms": round(float((response or {}).get("load_duration", 0)) / 1e6, 3),
        "prompt_tokens": int((response or {}).get("prompt_eval_count", 0)),
        "completion_tokens": completion,
        "tokens_per_second": round(completion / eval_seconds, 3) if eval_seconds > 0 else 0,
        "gpu_utilization_percent": gpu,
        "gpu_memory_used_mib": gpu_memory,
        "system_memory_used_mib": memory_used_mib(),
        "timeout": "timeout" in error.lower(),
        "detail": error or (response or {}).get("done_reason", ""),
    }


def cold_start_measurement(
    root: Path, python: Path, skip: bool
) -> list[dict[str, Any]]:
    if skip:
        return []
    ollama = root / ".local/ollama/ollama.exe"
    # A true cold start: stop only the project-local Ollama executable, then
    # restart it through the project manager before loading the frozen model.
    try:
        import psutil

        expected = ollama.resolve()
        for process in psutil.process_iter(["pid", "exe"]):
            try:
                executable = process.info.get("exe")
                if executable and Path(executable).resolve() == expected:
                    for child in process.children(recursive=True):
                        child.terminate()
                    process.terminate()
                    _, alive = psutil.wait_procs(
                        [*process.children(recursive=True), process], timeout=10
                    )
                    for remaining in alive:
                        remaining.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
    except Exception:
        subprocess.run(
            [str(ollama), "stop", OLLAMA_MODEL],
            capture_output=True, text=True, timeout=30, check=False,
        )
    time.sleep(1)
    server_started = time.perf_counter()
    start_result = subprocess.run(
        [str(python), str(root / "routing/scripts/manage_local_model.py"), "start"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    server_elapsed = time.perf_counter() - server_started
    rows = [
        row_from_ollama(
            "cold_server_001",
            "cold_ollama_server_start",
            1,
            True,
            {} if start_result.returncode == 0 else None,
            server_elapsed,
            "" if start_result.returncode == 0 else start_result.stderr[-300:],
        )
    ]
    started = time.perf_counter()
    try:
        response, _ = direct_parse(
            "下午2点从新街口步行到鼓楼，比较四类路线"
        )
        elapsed = time.perf_counter() - started
        rows.append(
            row_from_ollama(
                "cold_001", "cold_model_load_and_parse", 1, True, response, elapsed
            )
        )
    except Exception as exc:
        rows.append(
            row_from_ollama(
                "cold_001", "cold_model_load_and_parse", 1, True, None,
                time.perf_counter() - started, f"{type(exc).__name__}: {exc}"
            )
        )
    return rows


def run_model_performance(
    root: Path,
    python: Path,
    output_path: Path,
    parse_texts: list[str],
    explanation_cases: list[dict[str, Any]],
    *,
    skip_cold_start: bool,
) -> list[dict[str, Any]]:
    rows = cold_start_measurement(root, python, skip_cold_start)
    warm_texts = (parse_texts * 20)[:20]
    for index, text in enumerate(
        tqdm(warm_texts, desc="Step32 warm parse performance", unit="request", dynamic_ncols=True)
    ):
        started = time.perf_counter()
        try:
            response, elapsed = direct_parse(text)
            row = row_from_ollama(f"warm_{index+1:03d}", "warm_parse", 1, False, response, elapsed)
        except Exception as exc:
            row = row_from_ollama(f"warm_{index+1:03d}", "warm_parse", 1, False, None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}")
        rows.append(row)
        write_rows(output_path, rows, FIELDS)

    # Explanation latency is reused from the actual frozen endpoint output so
    # no model answer is cached or fabricated.
    for index, case in enumerate(explanation_cases[:10]):
        started = time.perf_counter()
        try:
            status, response, elapsed = api_json(
                "/api/assistant/explain",
                {"routes": case["route_metrics"], "question": case["question"]},
            )
            rows.append({
                **row_from_ollama(
                    f"explain_perf_{index+1:03d}", "explanation", 1, False,
                    {} if status == 200 else None, elapsed,
                    "" if status == 200 else json.dumps(response, ensure_ascii=False),
                ),
                "prompt_tokens": "",
                "completion_tokens": "",
                "tokens_per_second": "",
            })
        except Exception as exc:
            rows.append(row_from_ollama(
                f"explain_perf_{index+1:03d}", "explanation", 1, False,
                None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}",
            ))
        write_rows(output_path, rows, FIELDS)

    concurrency_text = "下午2点从新街口步行到鼓楼，比较四类路线"
    for concurrency in (1, 2):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(direct_parse, concurrency_text) for _ in range(concurrency)]
            for index, future in enumerate(futures):
                try:
                    response, individual_elapsed = future.result()
                    rows.append(row_from_ollama(
                        f"concurrency_{concurrency}_{index+1}",
                        "concurrency_parse", concurrency, False,
                        response, individual_elapsed,
                    ))
                except Exception as exc:
                    rows.append(row_from_ollama(
                        f"concurrency_{concurrency}_{index+1}",
                        "concurrency_parse", concurrency, False,
                        None, time.perf_counter() - started,
                        f"{type(exc).__name__}: {exc}",
                    ))
        write_rows(output_path, rows, FIELDS)
    return rows


def non_llm_baseline(output_path: Path) -> list[dict[str, Any]]:
    rows = []
    operations = []
    status, origin, elapsed = api_json("/api/geocode", {"query": "新街口", "limit": 5}, 30)
    operations.append(("geocode", status, elapsed))
    status2, destination, elapsed2 = api_json("/api/geocode", {"query": "鼓楼", "limit": 5}, 30)
    operations.append(("geocode", status2, elapsed2))
    origin_coord = {
        "crs": "EPSG:4326",
        "longitude": origin["candidates"][0]["longitude"],
        "latitude": origin["candidates"][0]["latitude"],
    }
    destination_coord = {
        "crs": "EPSG:4326",
        "longitude": destination["candidates"][0]["longitude"],
        "latitude": destination["candidates"][0]["latitude"],
    }
    route_payload = {
        "origin": origin_coord, "destination": destination_coord, "hour": 14,
        "mode": "walk", "objective": "risk_aware", "algorithm": "astar",
        "max_detour_ratio": .2, "uncertainty_weight": 1.0,
    }
    for operation, path, payload, timeout in [
        ("snap", "/api/snap", {"origin": origin_coord, "destination": destination_coord, "mode": "walk"}, 60),
        ("single_route", "/api/route", route_payload, 120),
        ("compare_routes", "/api/compare", {**route_payload, "objectives": ["shortest", "shade", "utci", "risk_aware"]}, 180),
    ]:
        code, data, duration = api_json(path, payload, timeout)
        operations.append((operation, code, duration))
    for index, (operation, code, duration) in enumerate(operations):
        rows.append({
            "request_id": f"baseline_{index+1:03d}", "operation": operation,
            "concurrency": 1, "cold_start": False,
            "status": "PASS" if code == 200 else "FAIL",
            "latency_ms": round(duration * 1000, 3), "load_ms": "",
            "prompt_tokens": "", "completion_tokens": "", "tokens_per_second": "",
            "gpu_utilization_percent": "", "gpu_memory_used_mib": "",
            "system_memory_used_mib": memory_used_mib(), "timeout": False,
            "detail": "deterministic non-LLM baseline",
        })
    write_rows(output_path, rows, FIELDS)
    return rows


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.inf
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def performance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    warm = [
        float(row["latency_ms"]) / 1000
        for row in rows
        if row["operation"] == "warm_parse" and row["status"] == "PASS"
    ]
    p95 = percentile(warm, .95)
    if p95 <= 10:
        classification = "interactive"
    elif p95 <= 30:
        classification = "acceptable_prototype"
    elif p95 <= 120:
        classification = "slow_research_prototype"
    else:
        classification = "non_interactive"
    return {
        "warm_parse_count": len(warm),
        "warm_parse_mean_seconds": statistics.mean(warm) if warm else None,
        "warm_parse_median_seconds": statistics.median(warm) if warm else None,
        "warm_parse_p95_seconds": p95 if math.isfinite(p95) else None,
        "interactive_performance_class": classification,
        "current_hardware_only": True,
        "model": OLLAMA_MODEL,
        "gpu": "NVIDIA GeForce RTX 4060 Ti",
    }
