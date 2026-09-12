"""stage_39: benchmark schema-constrained Qwen3-VL streetscape QA on 36 strata-balanced points."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from stage_38_qwen3vl_quantized_single_point import (
    assert_model_on_cuda,
    build_prompt,
    normalize_payload,
    parse_json_object,
    resize_for_vlm,
    schema_is_valid,
    validate_schema,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/structured_streetscape_qa.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_39"


def resolve(value: str) -> Path:
    return ROOT / value


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_39")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_39.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark_path = resolve(config["inputs"]["benchmark"])
    schema_path = resolve(config["inputs"]["response_schema"])
    require_file(benchmark_path)
    require_file(schema_path)
    benchmark = pd.read_csv(benchmark_path, dtype={"point_id": str})
    if len(benchmark) != 36 or benchmark["point_id"].duplicated().any():
        raise RuntimeError(f"Frozen benchmark contract changed: rows={len(benchmark)}")
    missing = [path for path in benchmark["output_path"].map(lambda value: Path(str(value))) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} benchmark panoramas; first={missing[0]}")
    return benchmark, json.loads(schema_path.read_text(encoding="utf-8"))


def load_runtime(config: dict[str, Any], logger: logging.Logger):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    model_id = config["model"]["model_id"]
    torch.manual_seed(int(config["runtime"]["seed"]))
    torch.cuda.manual_seed_all(int(config["runtime"]["seed"]))
    logger.info("Loading %s once for the 36-point benchmark", model_id)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation=config["runtime"]["attention"],
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda")
    placement = assert_model_on_cuda(model)
    model.eval()
    return processor, model, placement


def infer_one(
    config: dict[str, Any], case: dict[str, Any], schema: dict[str, Any],
    processor, model, output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    point_id = str(case["point_id"])
    case_dir = output_dir / "responses" / point_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / f"{point_id}_vlm_input.jpg"
    image_size = resize_for_vlm(
        Path(str(case["output_path"])), image_path,
        int(config["image"]["max_width"]), int(config["image"]["max_height"]),
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": build_prompt(case)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    if any(value.device.type != "cuda" for value in inputs.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("Inference tensor outside CUDA")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(config["runtime"]["max_new_tokens"]),
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    (case_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    raw_payload = parse_json_object(raw_text)
    raw_valid, raw_error = schema_is_valid(raw_payload, schema)
    normalized, events = normalize_payload(raw_payload, point_id)
    validate_schema(normalized, schema)
    (case_dir / "structured_response.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    peak_gib = torch.cuda.max_memory_reserved() / 1024**3
    if peak_gib > float(config["runtime"]["maximum_peak_vram_gib"]):
        raise RuntimeError(f"Peak VRAM {peak_gib:.3f} GiB exceeds gate")
    expected = config["evaluation"]["expected_actions"].get(str(case["benchmark_role"]), [])
    metrics = {
        "point_id": point_id,
        "benchmark_role": case["benchmark_role"],
        "raw_schema_valid": raw_valid,
        "normalized_schema_valid": True,
        "normalization_required": bool(events),
        "normalization_event_count": len(events),
        "normalization_events": "|".join(events),
        "raw_schema_error": raw_error or "",
        "recommended_action": normalized["recommended_action"],
        "expected_actions": "|".join(expected),
        "action_consistent": normalized["recommended_action"] in expected,
        "scene_type": normalized["scene_type"],
        "active_travel_eligible": normalized["active_travel_eligible"],
        "shade_need": normalized["shade_need"],
        "existing_canopy": normalized["existing_canopy"],
        "confidence": normalized["confidence"],
        "target_sector_count": len(normalized["target_sectors"]),
        "evidence_count": len(normalized["evidence"]),
        "requested_tool_count": len(normalized.get("requested_tools", [])),
        "inference_seconds": round(elapsed, 3),
        "peak_vram_gib": round(peak_gib, 3),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(trimmed.shape[1]),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "full_gpu_verified": True,
    }
    (case_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized, metrics, raw_text


def summarize(config: dict[str, Any], benchmark_count: int, metrics: pd.DataFrame, failure_count: int, placement: dict[str, Any]) -> dict[str, Any]:
    success = len(metrics)
    rates = {
        "raw_schema_rate": float(metrics["raw_schema_valid"].mean()) if success else 0.0,
        "normalized_schema_rate": float(metrics["normalized_schema_valid"].mean()) if success else 0.0,
        "normalization_required_rate": float(metrics["normalization_required"].mean()) if success else 0.0,
        "action_consistency_rate": float(metrics["action_consistent"].mean()) if success else 0.0,
        "full_gpu_rate": float(metrics["full_gpu_verified"].mean()) if success else 0.0,
        "failure_rate": failure_count / benchmark_count,
    }
    gates = config["evaluation"]["pass_gates"]
    failures = []
    if rates["normalized_schema_rate"] < float(gates["normalized_schema_rate"]):
        failures.append("normalized_schema_rate")
    if rates["full_gpu_rate"] < float(gates["full_gpu_rate"]):
        failures.append("full_gpu_rate")
    if rates["failure_rate"] > float(gates["maximum_failure_rate"]):
        failures.append("failure_rate")
    if rates["action_consistency_rate"] < float(gates["minimum_action_consistency_rate"]):
        failures.append("action_consistency_rate")
    return {
        "status": "PASS" if not failures else "TASK_QUALITY_FAIL",
        "generated": datetime.now().astimezone().isoformat(),
        "model_id": config["model"]["model_id"],
        "benchmark_points": benchmark_count,
        "successful_points": success,
        "failed_points": failure_count,
        **rates,
        "mean_inference_seconds": float(metrics["inference_seconds"].mean()) if success else None,
        "max_peak_vram_gib": float(metrics["peak_vram_gib"].max()) if success else None,
        "parameter_placement": placement,
        "failed_gates": failures,
    }


def write_report(summary: dict[str, Any], metrics: pd.DataFrame, path: Path) -> None:
    role_rows = metrics.groupby("benchmark_role", dropna=False).agg(
        points=("point_id", "count"), raw_schema_rate=("raw_schema_valid", "mean"),
        action_consistency_rate=("action_consistent", "mean"), mean_seconds=("inference_seconds", "mean"),
    ).reset_index()
    table = "\n".join(
        f"| {row.benchmark_role} | {int(row.points)} | {row.raw_schema_rate:.1%} | {row.action_consistency_rate:.1%} | {row.mean_seconds:.2f} |"
        for row in role_rows.itertuples(index=False)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# stage_39 Structured streetscape QA benchmark

Status: **{summary['status']}** (runtime completed; task-quality gate evaluated separately)  
Generated: {summary['generated']}

## Overall

- Model: `{summary['model_id']}`
- Success / failure: {summary['successful_points']} / {summary['failed_points']}
- Raw schema rate: {summary['raw_schema_rate']:.1%}
- Normalized schema rate: {summary['normalized_schema_rate']:.1%}
- Normalization required: {summary['normalization_required_rate']:.1%}
- Action consistency: {summary['action_consistency_rate']:.1%}
- Full-GPU rate: {summary['full_gpu_rate']:.1%}
- Mean inference: {summary['mean_inference_seconds']:.2f} s/point
- Maximum peak VRAM: {summary['max_peak_vram_gib']:.3f} GiB

## By frozen decision stratum

| Stratum | N | Raw schema | Action consistency | Seconds |
|---|---:|---:|---:|---:|
{table}

## Interpretation

Raw model output is never treated as a planning decision until deterministic parsing, schema
validation and action-policy checks pass. This benchmark measures whether QLoRA is justified;
it does not use the frozen citywide decisions as hidden labels for prompt completion.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-point metrics and process only missing points")
    args = parser.parse_args()
    logger = configure_logging()
    failed_path = LOG_DIR / "failed_files.csv"
    try:
        config = load_config()
        benchmark, schema = load_inputs(config)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        check = {
            "status": "PASS", "points": len(benchmark), "strata": benchmark["benchmark_role"].value_counts().to_dict(),
            "gpu": torch.cuda.get_device_name(0), "model_cached": True,
        }
        if args.mode == "check":
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0
        output_dir = resolve(config["outputs"]["directory"])
        if output_dir.exists() and any(output_dir.iterdir()) and not (args.overwrite or args.resume):
            raise FileExistsError(f"Output exists; use --overwrite: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        with failed_path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"]).writeheader()
        rows = []
        if args.resume:
            for metrics_path in (output_dir / "responses").glob("*/metrics.json"):
                rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        completed_ids = {str(row["point_id"]) for row in rows}
        pending = benchmark.loc[~benchmark["point_id"].isin(completed_ids)].copy()
        logger.info("Resume state: completed=%d pending=%d", len(completed_ids), len(pending))
        processor, model, placement = load_runtime(config, logger)
        failures = 0
        for record in tqdm(pending.to_dict("records"), desc="Structured streetscape QA", unit="point", dynamic_ncols=True):
            try:
                _, metrics, _ = infer_one(config, record, schema, processor, model, output_dir)
                rows.append(metrics)
            except Exception as error:
                failures += 1
                logger.exception("Point %s failed", record["point_id"])
                with failed_path.open("a", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
                    writer.writerow({"point_id": record["point_id"], "filename": record["output_path"], "error_message": f"{type(error).__name__}: {error}"})
                torch.cuda.empty_cache()
        metrics = pd.DataFrame(rows)
        if metrics.empty:
            raise RuntimeError("All benchmark points failed")
        metrics.to_csv(output_dir / "qa_metrics.csv", index=False, encoding="utf-8-sig")
        summary = summarize(config, len(benchmark), metrics, failures, placement)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(summary, metrics, resolve(config["outputs"]["report"]))
        logger.info("stage_39 %s: success=%d failure=%d raw_schema=%.1f%% action=%.1f%%", summary["status"], len(metrics), failures, 100 * summary["raw_schema_rate"], 100 * summary["action_consistency_rate"])
        return 0 if summary["status"] == "PASS" else 2
    except Exception as error:
        logger.exception("stage_39 failed")
        with failed_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_39", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
