"""stage_41c: screen active-travel evidence by heading without asking a 2B VLM for boxes."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
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

from stage_38_qwen3vl_quantized_single_point import assert_model_on_cuda, schema_is_valid, validate_schema


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/active_travel_heading_screen.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_41c"


def resolve(value: str) -> Path:
    return ROOT / value


def logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_41c")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_41c.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def parse(text: str) -> dict[str, Any]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize(payload: dict[str, Any], point_id: str, headings: list[int]) -> tuple[dict[str, Any], list[str]]:
    out = dict(payload)
    events = []
    out["schema_version"] = "1.0"
    out["point_id"] = str(point_id)
    scene_values = {"public_sidewalk", "shared_walk_cycle", "pedestrian_space", "motor_vehicle_only", "covered_infrastructure", "uncertain"}
    scene_alias = {"highway": "motor_vehicle_only", "motorway": "motor_vehicle_only", "urban street": "uncertain", "urban roadway": "uncertain"}
    scene = str(out.get("scene_type", "")).lower()
    out["scene_type"] = scene_alias.get(scene, scene if scene in scene_values else "uncertain")
    if scene != out["scene_type"]:
        events.append("scene_type:normalized")
    visible_values = {"yes", "no", "uncertain"}
    kind_values = {"sidewalk", "cycle_lane", "shared_path", "pedestrian_area", "crossing", "none", "uncertain"}
    raw_items = out.get("heading_assessments", []) if isinstance(out.get("heading_assessments"), list) else []
    by_heading = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            heading = int(item.get("heading"))
        except (TypeError, ValueError):
            continue
        if heading not in headings or heading in by_heading:
            continue
        visible = item.get("visible")
        if isinstance(visible, bool):
            visible = "yes" if visible else "no"
            events.append(f"heading_{heading}:boolean_visible")
        if visible not in visible_values:
            visible = "uncertain"
            events.append(f"heading_{heading}:unsafe_visible")
        kind = str(item.get("kind", "uncertain")).lower().replace(" ", "_")
        if kind not in kind_values:
            kind = "uncertain"
            events.append(f"heading_{heading}:unsafe_kind")
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        reason = str(item.get("reason", "No valid reason returned."))[:180] or "No valid reason returned."
        by_heading[heading] = {"heading": heading, "visible": visible, "kind": kind, "confidence": confidence, "reason": reason}
    for heading in headings:
        if heading not in by_heading:
            by_heading[heading] = {"heading": heading, "visible": "uncertain", "kind": "uncertain", "confidence": 0.0, "reason": "Heading assessment missing from model output."}
            events.append(f"heading_{heading}:missing_inserted")
    out["heading_assessments"] = [by_heading[heading] for heading in headings]
    request_more = out.get("request_more_views", False)
    out["request_more_views"] = request_more if isinstance(request_more, bool) else str(request_more).lower() in {"true", "yes", "1"}
    try:
        out["confidence"] = min(1.0, max(0.0, float(out.get("confidence", 0.0))))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    warnings = out.get("warnings", [])
    out["warnings"] = [str(value)[:200] for value in warnings[:1]] if isinstance(warnings, list) else []
    if events:
        out["warnings"].append("Deterministic protocol normalization applied.")
    return out, events


def prompt(case: dict[str, Any], headings: list[int]) -> str:
    context = {key: case.get(key) for key in ("point_id", "month", "SVF", "GVI", "fclass", "nearest_motor_only_m", "nearest_thermal_relevant_m")}
    context = {key: None if pd.isna(value) else value for key, value in context.items()}
    return f"""Independently inspect each labelled street-view direction for a visibly delineated sidewalk, cycle lane, shared path, pedestrian area or crossing. Camera pitch is 0 degrees. Context: {json.dumps(context, ensure_ascii=False, allow_nan=False)}.
Do not output boxes. Road asphalt used only by motor vehicles is not active-travel space. A planting strip alone is not a sidewalk. Do not copy one heading's answer to another.
Return compact JSON only: schema_version='1.0', point_id='{case['point_id']}', scene_type, heading_assessments, request_more_views, confidence, warnings.
heading_assessments must contain exactly headings {headings}; each item has heading, visible=yes/no/uncertain, kind, confidence, reason under 18 words. Stop after JSON."""


def load_runtime(cfg: dict[str, Any]):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    model_id = cfg["model"]["model_id"]
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation=cfg["runtime"]["attention"], low_cpu_mem_usage=True, local_files_only=True).to("cuda").eval()
    placement = assert_model_on_cuda(model)
    return processor, model, placement


def infer(case: dict[str, Any], cfg: dict[str, Any], processor, model, schema: dict[str, Any], output: Path) -> dict[str, Any]:
    headings = list(cfg["benchmark"]["headings"])
    content = []
    for heading in headings:
        content.extend([{"type": "text", "text": f"Heading {heading} degrees:"}, {"type": "image", "image": case[f"heading_{heading}_path"]}])
    content.append({"type": "text", "text": prompt(case, headings)})
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
    if any(value.device.type != "cuda" for value in inputs.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("Inference tensor outside CUDA")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=int(cfg["runtime"]["max_new_tokens"]), do_sample=False, use_cache=True)
    elapsed = time.perf_counter() - started
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    raw = parse(raw_text)
    raw_valid, raw_error = schema_is_valid(raw, schema)
    normalized, events = normalize(raw, str(case["point_id"]), headings)
    validate_schema(normalized, schema)
    peak = torch.cuda.max_memory_reserved() / 1024**3
    if peak > float(cfg["runtime"]["maximum_peak_vram_gib"]):
        raise RuntimeError(f"Peak VRAM {peak:.3f} GiB exceeds gate")
    point_dir = output / "responses" / str(case["point_id"])
    point_dir.mkdir(parents=True, exist_ok=True)
    (point_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    (point_dir / "heading_screen.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates = [item["heading"] for item in normalized["heading_assessments"] if item["visible"] == "yes"]
    uncertain = [item["heading"] for item in normalized["heading_assessments"] if item["visible"] == "uncertain"]
    metrics = {"point_id": str(case["point_id"]), "benchmark_role": case["benchmark_role"], "raw_schema_valid": raw_valid, "raw_schema_error": raw_error or "", "normalization_events": "|".join(events), "scene_type": normalized["scene_type"], "candidate_headings": "|".join(map(str, candidates)), "candidate_heading_count": len(candidates), "uncertain_heading_count": len(uncertain), "request_more_views": normalized["request_more_views"], "inference_seconds": round(elapsed, 3), "peak_vram_gib": round(peak, 3), "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(trimmed.shape[1]), "full_gpu_verified": True}
    (point_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_outputs(metrics: pd.DataFrame, cfg: dict[str, Any], output: Path, placement: dict[str, Any], scope: str, failures: int) -> dict[str, Any]:
    positive = metrics.loc[metrics.benchmark_role.eq("positive_walkable_control")]
    gates = cfg["gates"]
    checks = {
        "success_rate": len(metrics) / (3 if scope == "pilot" else int(cfg["benchmark"]["full_points"])),
        "normalized_schema_rate": 1.0 if len(metrics) else 0.0,
        "positive_control_has_candidate": bool(len(positive) and positive.candidate_heading_count.gt(0).all()),
        "mean_candidate_headings": float(metrics.candidate_heading_count.mean()) if len(metrics) else 4.0,
    }
    passed = checks["success_rate"] >= float(gates["pilot_success_rate"]) and checks["normalized_schema_rate"] >= float(gates["normalized_schema_rate"]) and checks["positive_control_has_candidate"] and checks["mean_candidate_headings"] <= float(gates["maximum_mean_candidate_headings"])
    summary = {"status": f"{scope.upper()}_PASS" if passed else f"{scope.upper()}_SELECTIVITY_FAIL", "generated": datetime.now().astimezone().isoformat(), "scope": scope, "successful_points": len(metrics), "failed_points": failures, "checks": checks, "mean_inference_seconds": float(metrics.inference_seconds.mean()) if len(metrics) else None, "max_peak_vram_gib": float(metrics.peak_vram_gib.max()) if len(metrics) else None, "placement": placement}
    metrics.to_csv(output / f"{scope}_metrics.csv", index=False, encoding="utf-8-sig")
    (output / f"{scope}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = resolve(cfg["outputs"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"""# stage_41c Active-travel heading screen

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Scope: {scope}, success: {len(metrics)}, failures: {failures}
- Mean candidate headings: {checks['mean_candidate_headings']:.2f}/4
- Positive control has candidate: {checks['positive_control_has_candidate']}
- Mean inference: {summary['mean_inference_seconds']:.2f} s/point
- Peak VRAM: {summary['max_peak_vram_gib']:.3f} GiB

This stage outputs heading clues only. It cannot create masks, promote planting locations or unlock generation.
""", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "pilot", "run"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    log = logger()
    failed_path = LOG_DIR / "failed_files.csv"
    try:
        cfg = config()
        benchmark = pd.read_csv(resolve(cfg["inputs"]["benchmark"]), dtype={"point_id": str})
        schema = json.loads(resolve(cfg["inputs"]["response_schema"]).read_text(encoding="utf-8"))
        path_columns = [f"heading_{heading}_path" for heading in cfg["benchmark"]["headings"]]
        missing = int((~benchmark[path_columns].apply(lambda row: all(Path(str(value)).is_file() for value in row), axis=1)).sum())
        check = {"status": "PASS" if len(benchmark) == int(cfg["benchmark"]["full_points"]) and missing == 0 and torch.cuda.is_available() else "FAIL", "points": len(benchmark), "missing_view_points": missing, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.mode == "check":
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0 if check["status"] == "PASS" else 2
        scope = "pilot" if args.mode == "pilot" else "full"
        selected = benchmark.groupby("benchmark_role", sort=True).head(int(cfg["benchmark"]["pilot_per_group"])) if scope == "pilot" else benchmark
        output = resolve(cfg["outputs"]["directory"])
        if output.exists() and any(output.iterdir()) and not (args.overwrite or args.resume):
            raise FileExistsError(f"Output exists; use --overwrite or --resume: {output}")
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        if args.resume:
            for path in (output / "responses").glob("*/metrics.json"):
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        done = {str(row["point_id"]) for row in rows}
        pending = selected.loc[~selected.point_id.isin(done)]
        processor, model, placement = load_runtime(cfg)
        errors = []
        for case in tqdm(pending.to_dict("records"), desc=f"Four-view heading screen ({scope})", unit="point", dynamic_ncols=True):
            try:
                rows.append(infer(case, cfg, processor, model, schema, output))
            except Exception as error:
                log.exception("Point %s failed", case["point_id"])
                errors.append({"point_id": case["point_id"], "filename": "four_views", "error_message": f"{type(error).__name__}: {error}"})
        pd.DataFrame(errors, columns=["point_id", "filename", "error_message"]).to_csv(failed_path, index=False, encoding="utf-8-sig")
        metrics = pd.DataFrame([row for row in rows if str(row["point_id"]) in set(selected.point_id)])
        summary = write_outputs(metrics, cfg, output, placement, scope, len(errors))
        log.info("stage_41c %s", summary)
        return 0 if summary["status"].endswith("PASS") else 2
    except Exception as error:
        log.exception("stage_41c failed")
        with failed_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_41c", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

