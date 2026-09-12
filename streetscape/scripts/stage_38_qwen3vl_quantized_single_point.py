"""stage_38: run one schema-constrained Qwen3-VL streetscape inference on CUDA only."""

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
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/qwen3vl_single_point.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_38"


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_38")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_38.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def resolve(value: str) -> Path:
    return ROOT / value


def load_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def cuda_contract() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free, _ = torch.cuda.mem_get_info()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gib": round(total, 3),
        "free_vram_gib_before": round(free / 1024**3, 3),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }


def select_case(config: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(resolve(config["inputs"]["benchmark"]), dtype={"point_id": str})
    point_id = str(config["benchmark"]["point_id"])
    selected = frame.loc[frame["point_id"] == point_id]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one benchmark record for point {point_id}; got {len(selected)}")
    row = selected.iloc[0].to_dict()
    if row.get("benchmark_role") != config["benchmark"]["expected_role"]:
        raise RuntimeError("Benchmark role changed after stage_37")
    panorama = Path(str(row["output_path"]))
    require_file(panorama)
    row["panorama_path"] = str(panorama)
    return row


def resize_for_vlm(source: Path, target: Path, max_width: int, max_height: int) -> tuple[int, int]:
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="JPEG", quality=94, subsampling=0)
        return image.size


def compact_context(case: dict[str, Any]) -> str:
    fields = ("point_id", "benchmark_role", "month", "SVF", "GVI", "shade_pred_mean", "tmrt_pred_mean", "utci_pred_mean")
    values = {}
    for key in fields:
        value = case.get(key)
        values[key] = None if pd.isna(value) else value
    return json.dumps(values, ensure_ascii=False, allow_nan=False)


def build_prompt(case: dict[str, Any]) -> str:
    return f"""You are an evidence-grounded urban streetscape shade planning assistant.
Inspect the north-aligned 360-degree panorama. North is at the left/right seam; east, south and west are at 1/4, 1/2 and 3/4 width.
Structured measurements: {compact_context(case)}

Return one JSON object only. Do not use markdown. Use exactly these keys:
schema_version='1.0'; point_id='{case['point_id']}'; scene_type; active_travel_eligible; shade_need;
existing_canopy; recommended_action; target_sectors; confidence; evidence; requested_tools; warnings.
Allowed sectors are integers 0..23 clockwise from north, 15 degrees each.
Do not invent ownership, engineering feasibility or exact thermal improvement. If evidence is insufficient, request tools or abstain.
Tree-first is preferred only for visible, usable walking/cycling space with shade need. Existing adequate shade should be maintained.
Evidence entries must contain source, claim, and optional sector.
Write English only. Use at most 6 target sectors, 1-3 evidence entries, 0-3 requested tools and 0-2 warnings.
Keep every claim under 20 words. Output a compact single-line JSON object and stop immediately after the closing brace."""


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        if "schema_version" not in cleaned or "point_id" not in cleaned:
            raise
        payload: dict[str, Any] = {}
        for segment in cleaned.split(";"):
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            payload[key] = value
        sectors = payload.get("target_sectors", "")
        payload["target_sectors"] = [int(value) for value in re.findall(r"\d+", sectors) if 0 <= int(value) <= 23]
        try:
            payload["confidence"] = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            payload["confidence"] = 0.0
        evidence_text = str(payload.get("evidence", "")).strip()
        sector_match = re.search(r"sector\s*(\d+)", evidence_text, flags=re.IGNORECASE)
        evidence = {"source": "panorama", "claim": evidence_text[:500] or "Key-value output contained no evidence claim."}
        if sector_match and 0 <= int(sector_match.group(1)) <= 23:
            evidence["sector"] = int(sector_match.group(1))
        payload["evidence"] = [evidence]
        payload["requested_tools"] = [] if str(payload.get("requested_tools", "")).lower() in {"", "none", "[]"} else [payload["requested_tools"]]
        payload["warnings"] = [] if str(payload.get("warnings", "")).lower() in {"", "none", "[]"} else [payload["warnings"]]
        return payload


def validate_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as error:
        raise RuntimeError("jsonschema is required; run --mode install first") from error
    jsonschema.validate(instance=payload, schema=schema)


def schema_is_valid(payload: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        validate_schema(payload, schema)
        return True, None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def normalize_payload(payload: dict[str, Any], point_id: str) -> tuple[dict[str, Any], list[str]]:
    """Deterministically normalize syntax only; never invent missing visual evidence."""
    normalized = dict(payload)
    events: list[str] = []
    allowed = {
        "scene_type": {"public_sidewalk", "shared_walk_cycle", "pedestrian_space", "motor_vehicle_only", "covered_infrastructure", "other", "uncertain"},
        "active_travel_eligible": {"yes", "no", "uncertain"},
        "shade_need": {"none", "low", "moderate", "high", "uncertain"},
        "existing_canopy": {"adequate", "continuous", "discontinuous", "sparse", "leaf_off", "none", "uncertain"},
        "recommended_action": {"maintain", "tree_first", "shade_facility_supplement", "planner_review", "abstain"},
    }
    normalized["schema_version"] = "1.0"
    normalized["point_id"] = str(point_id)
    if isinstance(normalized.get("active_travel_eligible"), bool):
        normalized["active_travel_eligible"] = "yes" if normalized["active_travel_eligible"] else "no"
        events.append("active_travel_eligible:boolean_to_enum")
    action_text = str(normalized.get("recommended_action", "")).lower()
    action_aliases = {
        "maintain existing canopy": "maintain",
        "maintain existing shade": "maintain",
        "maintain": "maintain",
        "tree planting": "tree_first",
        "tree-first": "tree_first",
        "planner review": "planner_review",
    }
    if action_text in action_aliases:
        normalized["recommended_action"] = action_aliases[action_text]
        if action_text != normalized["recommended_action"]:
            events.append("recommended_action:alias_to_enum")
    for field, values in allowed.items():
        field_value = normalized.get(field)
        if not isinstance(field_value, str) or field_value not in values:
            normalized[field] = "uncertain" if field != "recommended_action" else "planner_review"
            events.append(f"{field}:unsafe_value_to_uncertain")
    sectors = normalized.get("target_sectors", [])
    if not isinstance(sectors, list):
        sectors = []
        events.append("target_sectors:non_list_to_empty")
    normalized["target_sectors"] = sorted({int(value) for value in sectors if isinstance(value, (int, float)) and 0 <= int(value) <= 23})[:6]
    try:
        normalized["confidence"] = min(1.0, max(0.0, float(normalized.get("confidence", 0.0))))
    except (TypeError, ValueError):
        normalized["confidence"] = 0.0
        events.append("confidence:invalid_to_zero")
    allowed_sources = {"panorama", "direction_view", "semantic", "depth", "svf", "thermal", "road_context", "retrieved_case", "tool"}
    evidence = []
    for item in normalized.get("evidence", []) if isinstance(normalized.get("evidence"), list) else []:
        if not isinstance(item, dict) or item.get("source") not in allowed_sources or not str(item.get("claim", "")).strip():
            continue
        clean = {"source": item["source"], "claim": str(item["claim"])[:500]}
        if isinstance(item.get("sector"), int) and 0 <= item["sector"] <= 23:
            clean["sector"] = item["sector"]
        evidence.append(clean)
        if len(evidence) == 3:
            break
    if not evidence:
        evidence = [{"source": "panorama", "claim": "Model output contained no schema-valid visual evidence."}]
        events.append("evidence:empty_to_explicit_warning")
    elif len(normalized.get("evidence", [])) > len(evidence):
        events.append("evidence:truncated_to_three")
    normalized["evidence"] = evidence
    allowed_tools = {"measure_svf", "locate_walkable_area", "detect_existing_canopy", "estimate_depth_ground", "retrieve_similar_cases", "propose_tree_layout", "render_counterfactual", "evaluate_counterfactual"}
    tools = normalized.get("requested_tools", [])
    normalized["requested_tools"] = list(dict.fromkeys(value for value in tools if value in allowed_tools))[:3] if isinstance(tools, list) else []
    if any(event.startswith(("shade_need:", "existing_canopy:")) for event in events):
        for tool in ("measure_svf", "detect_existing_canopy"):
            if tool not in normalized["requested_tools"] and len(normalized["requested_tools"]) < 3:
                normalized["requested_tools"].append(tool)
    warnings = normalized.get("warnings", [])
    warnings = [str(value)[:300] for value in warnings[:1]] if isinstance(warnings, list) else []
    if events:
        warnings.append("Deterministic schema normalization applied; uncertain fields require tool verification.")
    normalized["warnings"] = warnings[:2]
    return normalized, events


def assert_model_on_cuda(model: torch.nn.Module) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for parameter in tqdm(model.parameters(), desc="Verifying model placement", unit="tensor", dynamic_ncols=True):
        key = parameter.device.type
        counts[key] = counts.get(key, 0) + 1
    forbidden = [key for key in counts if key != "cuda"]
    device_map = getattr(model, "hf_device_map", None)
    if forbidden:
        raise RuntimeError(f"CPU/disk model tensors detected: {counts}")
    if isinstance(device_map, dict) and any(str(value) in {"cpu", "disk"} for value in device_map.values()):
        raise RuntimeError(f"CPU/disk offload detected in hf_device_map: {device_map}")
    return {"parameter_devices": counts, "hf_device_map": device_map}


def run_inference(config: dict[str, Any], case: dict[str, Any], output_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    model_id = config["models"]["baseline"]["model_id"]
    image_path = output_dir / f"{case['point_id']}_vlm_input.jpg"
    image_size = resize_for_vlm(
        Path(case["panorama_path"]), image_path,
        int(config["benchmark"]["image_max_width"]), int(config["benchmark"]["image_max_height"]),
    )
    torch.manual_seed(int(config["runtime"]["seed"]))
    torch.cuda.manual_seed_all(int(config["runtime"]["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    logger.info("Loading %s in BF16 directly on CUDA", model_id)
    started = time.perf_counter()
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
    load_seconds = time.perf_counter() - started

    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": build_prompt(case)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    if any(tensor.device.type != "cuda" for tensor in inputs.values() if isinstance(tensor, torch.Tensor)):
        raise RuntimeError("One or more inference tensors are not on CUDA")
    inference_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(config["runtime"]["max_new_tokens"]),
            do_sample=False,
            use_cache=True,
        )
    inference_seconds = time.perf_counter() - inference_started
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    peak_gib = torch.cuda.max_memory_reserved() / 1024**3
    if peak_gib > float(config["runtime"]["maximum_peak_vram_gib"]):
        raise RuntimeError(f"Peak VRAM {peak_gib:.3f} GiB exceeds frozen gate")
    (output_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    raw_payload = parse_json_object(raw_text)
    schema = json.loads(resolve(config["inputs"]["response_schema"]).read_text(encoding="utf-8"))
    raw_valid, raw_error = schema_is_valid(raw_payload, schema)
    payload, normalization_events = normalize_payload(raw_payload, str(case["point_id"]))
    validate_schema(payload, schema)
    return {
        "payload": payload,
        "raw_text": raw_text,
        "metrics": {
            "model_id": model_id,
            "precision": "bf16",
            "point_id": str(case["point_id"]),
            "image_size": list(image_size),
            "load_seconds": round(load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "peak_vram_gib": round(peak_gib, 3),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(trimmed.shape[1]),
            "schema_valid": True,
            "raw_schema_valid": raw_valid,
            "raw_schema_error": raw_error,
            "normalization_events": normalization_events,
            "full_gpu_verified": True,
            **placement,
        },
    }


def write_report(config: dict[str, Any], result: dict[str, Any], report: Path) -> None:
    metrics = result["metrics"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"""# stage_38 Qwen3-VL single-point benchmark

Status: **PASS**  
Generated: {datetime.now().astimezone().isoformat()}

## Verified baseline

- Model: `{metrics['model_id']}`
- Precision: {metrics['precision']}
- GPU-only placement: {metrics['full_gpu_verified']}
- Peak reserved VRAM: {metrics['peak_vram_gib']:.3f} GiB
- Load / inference: {metrics['load_seconds']:.3f} s / {metrics['inference_seconds']:.3f} s
- Input / output tokens: {metrics['input_tokens']} / {metrics['output_tokens']}
- Raw JSON Schema valid: {metrics['raw_schema_valid']}
- Deterministically normalized JSON Schema valid: {metrics['schema_valid']}
- Normalization events: {', '.join(metrics['normalization_events']) if metrics['normalization_events'] else 'none'}

## 4B gate

`{config['models']['primary_candidate']['model_id']}` remains blocked on this Windows 8 GiB host until a
verified runtime can keep every FP8 tensor on CUDA. It is not counted as completed.

## Next gate

stage_39 expands the frozen structured QA to the 36-point stratified benchmark and measures schema,
abstention, action consistency and evidence-grounding reliability before any QLoRA decision.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "install", "run"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = configure_logging()
    failed = LOG_DIR / "failed_files.csv"
    try:
        config = load_config()
        require_file(resolve(config["inputs"]["benchmark"]))
        require_file(resolve(config["inputs"]["response_schema"]))
        hardware = cuda_contract()
        case = select_case(config)
        if args.mode == "install":
            raise RuntimeError("Use the project environment installer; this script never mutates environments")
        if args.mode == "check":
            print(json.dumps({"status": "PASS", "hardware": hardware, "case": case, "model": config["models"]}, ensure_ascii=False, indent=2))
            return 0
        output_dir = resolve(config["outputs"]["directory"])
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = run_inference(config, case, output_dir, logger)
        (output_dir / "structured_response.json").write_text(json.dumps(result["payload"], ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "raw_response.txt").write_text(result["raw_text"], encoding="utf-8")
        (output_dir / "runtime_metrics.json").write_text(json.dumps(result["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(config, result, resolve(config["outputs"]["report"]))
        with failed.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"]).writeheader()
        logger.info("stage_38 PASS: peak_vram=%.3f GiB", result["metrics"]["peak_vram_gib"])
        return 0
    except Exception as error:
        logger.exception("stage_38 failed")
        with failed.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_38", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
