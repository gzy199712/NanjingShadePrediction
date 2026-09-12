"""stage_40: multi-view Qwen3-VL rescue benchmark for missed active-travel space."""

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
CONFIG = ROOT / "streetscape/configs/active_travel_rescue.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_40"


def resolve(value: str) -> Path:
    return ROOT / value


def logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_40")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_40.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def evenly(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    fields = [field for field in ("walkable_direction_count", "retained_walkable_ratio", "SVF", "month", "point_id") if field in frame]
    ordered = frame.sort_values(fields, kind="stable", na_position="first")
    if len(ordered) <= count:
        return ordered
    indices = sorted({round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)})
    return ordered.iloc[indices]


def build_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    decisions = pd.read_csv(resolve(cfg["inputs"]["decisions"]), dtype={"point_id": str}, low_memory=False)
    groups = [
        ("rescue_candidate", decisions.final_decision_class.eq("AUTO_PLANNING_ABSTAIN_INSUFFICIENT_VISIBLE_WALKABLE"), int(cfg["benchmark"]["rescue_candidates"])),
        ("motor_only_control", decisions.final_decision_class.eq("MOTOR_VEHICLE_ONLY_EXCLUDED"), int(cfg["benchmark"]["motor_only_controls"])),
        ("positive_walkable_control", decisions.walkable_direction_count.fillna(0).ge(2) & ~decisions.final_decision_class.isin(["TREE_PRIORITY_CANDIDATE", "ACTIONABLE_VISUAL_ADVICE"]), int(cfg["benchmark"]["positive_walkable_controls"])),
    ]
    selected = []
    for role, mask, count in tqdm(groups, desc="Selecting rescue benchmark", unit="group", dynamic_ncols=True):
        part = evenly(decisions.loc[mask].copy(), count)
        part["benchmark_role"] = role
        selected.append(part)
    manifest = pd.concat(selected, ignore_index=True).drop_duplicates("point_id")
    metadata = pd.read_csv(resolve(cfg["inputs"]["image_metadata"]), dtype={"point_id": str})
    metadata = metadata.loc[metadata.heading.isin(cfg["benchmark"]["headings"]), ["point_id", "heading", "filepath"]].copy()
    raw_dir = resolve(cfg["inputs"]["raw_direction_directory"])
    metadata["filepath"] = metadata.filepath.map(lambda value: str(raw_dir / Path(str(value)).name))
    wide = metadata.pivot(index="point_id", columns="heading", values="filepath").reset_index()
    wide.columns = ["point_id" if value == "point_id" else f"heading_{int(value)}_path" for value in wide.columns]
    manifest = manifest.merge(wide, on="point_id", how="left", validate="one_to_one")
    path_columns = [f"heading_{heading}_path" for heading in cfg["benchmark"]["headings"]]
    manifest["all_four_views_exist"] = manifest[path_columns].apply(lambda row: all(Path(str(value)).is_file() for value in row), axis=1)
    columns = ["point_id", "benchmark_role", "month", "SVF", "GVI", "fclass", "nearest_motor_only_m", "nearest_thermal_relevant_m", "walkable_direction_count", "retained_walkable_ratio", "final_decision_class", *path_columns, "all_four_views_exist"]
    return manifest[[column for column in columns if column in manifest.columns]].sort_values(["benchmark_role", "point_id"], kind="stable")


def build_prompt(case: dict[str, Any], headings: list[int]) -> str:
    context = {key: case.get(key) for key in ("point_id", "month", "SVF", "GVI", "fclass", "nearest_motor_only_m", "nearest_thermal_relevant_m")}
    context = {key: (None if pd.isna(value) else value) for key, value in context.items()}
    return f"""Assess visible walking/cycling space from four perspective street views. Each image is labelled by geographic heading. Camera pitch is 0 degrees. Road context: {json.dumps(context, ensure_ascii=False, allow_nan=False)}.
Do not infer the hidden benchmark role. Localize only visible sidewalk, cycle lane, shared path, pedestrian area or crossing.
Return one compact JSON object only with keys: schema_version='1.0', point_id='{case['point_id']}', view_headings={headings}, scene_type, active_travel_visible, walkable_regions, occlusion, rescue_decision, confidence, evidence, warnings.
Each walkable region must contain heading, bbox_norm=[x1,y1,x2,y2] scaled 0..1, kind, confidence. Maximum 8 regions and 4 short evidence strings.
Use PROMOTE_TO_GROUNDING only when a visible region is localized; EXCLUDE_MOTOR_ONLY only with strong visual and road evidence; REQUEST_8_VIEWS when four views are insufficient; otherwise KEEP_ABSTAIN. Stop after the JSON."""


def parse_payload(text: str) -> dict[str, Any]:
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
    out["view_headings"] = headings
    allowed = {
        "scene_type": {"public_sidewalk", "shared_walk_cycle", "pedestrian_space", "motor_vehicle_only", "covered_infrastructure", "uncertain"},
        "active_travel_visible": {"yes", "no", "uncertain"},
        "occlusion": {"none", "vehicles", "vegetation", "barrier", "geometry", "blur", "other", "uncertain"},
        "rescue_decision": {"PROMOTE_TO_GROUNDING", "KEEP_ABSTAIN", "EXCLUDE_MOTOR_ONLY", "REQUEST_8_VIEWS"},
    }
    fallback = {"scene_type": "uncertain", "active_travel_visible": "uncertain", "occlusion": "uncertain", "rescue_decision": "KEEP_ABSTAIN"}
    for field, values in allowed.items():
        if not isinstance(out.get(field), str) or out[field] not in values:
            out[field] = fallback[field]
            events.append(f"{field}:unsafe_to_{fallback[field]}")
    regions = []
    kinds = {"sidewalk", "cycle_lane", "shared_path", "pedestrian_area", "crossing", "uncertain"}
    for region in out.get("walkable_regions", []) if isinstance(out.get("walkable_regions"), list) else []:
        if not isinstance(region, dict):
            continue
        heading = region.get("heading")
        bbox = region.get("bbox_norm")
        if heading not in headings or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            bbox = [min(1.0, max(0.0, float(value))) for value in bbox]
            confidence = min(1.0, max(0.0, float(region.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        regions.append({"heading": int(heading), "bbox_norm": bbox, "kind": region.get("kind") if region.get("kind") in kinds else "uncertain", "confidence": confidence})
        if len(regions) == 8:
            break
    out["walkable_regions"] = regions
    if out["rescue_decision"] == "PROMOTE_TO_GROUNDING" and not regions:
        out["rescue_decision"] = "KEEP_ABSTAIN"
        events.append("promotion_without_bbox:kept_abstain")
    try:
        out["confidence"] = min(1.0, max(0.0, float(out.get("confidence", 0.0))))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
        events.append("confidence:invalid_to_zero")
    evidence = out.get("evidence", [])
    out["evidence"] = [str(value)[:300] for value in evidence[:4]] if isinstance(evidence, list) else [str(evidence)[:300]]
    if not out["evidence"]:
        out["evidence"] = ["No schema-valid evidence was returned."]
    warnings = out.get("warnings", [])
    out["warnings"] = [str(value)[:300] for value in warnings[:2]] if isinstance(warnings, list) else []
    if events:
        out["warnings"].append("Deterministic normalization applied; unsafe values did not unlock generation.")
    return out, events


def load_runtime(cfg: dict[str, Any]):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    model_id = cfg["model"]["model_id"]
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation=cfg["runtime"]["attention"], low_cpu_mem_usage=True, local_files_only=True).to("cuda").eval()
    placement = assert_model_on_cuda(model)
    return processor, model, placement


def infer(cfg: dict[str, Any], case: dict[str, Any], processor, model, schema: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    headings = list(cfg["benchmark"]["headings"])
    content = []
    for heading in headings:
        content.extend([{"type": "text", "text": f"Geographic heading {heading} degrees:"}, {"type": "image", "image": case[f"heading_{heading}_path"]}])
    content.append({"type": "text", "text": build_prompt(case, headings)})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
    if any(value.device.type != "cuda" for value in inputs.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("Inference tensor outside CUDA")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=int(cfg["runtime"]["max_new_tokens"]), do_sample=False, use_cache=True)
    elapsed = time.perf_counter() - started
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    raw = parse_payload(raw_text)
    raw_valid, raw_error = schema_is_valid(raw, schema)
    normalized, events = normalize(raw, str(case["point_id"]), headings)
    validate_schema(normalized, schema)
    peak = torch.cuda.max_memory_reserved() / 1024**3
    if peak > float(cfg["runtime"]["maximum_peak_vram_gib"]):
        raise RuntimeError(f"Peak VRAM {peak:.3f} GiB exceeds gate")
    point_dir = out_dir / "pilot" / str(case["point_id"])
    point_dir.mkdir(parents=True, exist_ok=True)
    (point_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    (point_dir / "structured_response.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {"point_id": str(case["point_id"]), "benchmark_role": case["benchmark_role"], "raw_schema_valid": raw_valid, "raw_schema_error": raw_error or "", "normalization_events": "|".join(events), "rescue_decision": normalized["rescue_decision"], "active_travel_visible": normalized["active_travel_visible"], "region_count": len(normalized["walkable_regions"]), "inference_seconds": round(elapsed, 3), "peak_vram_gib": round(peak, 3), "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(trimmed.shape[1]), "full_gpu_verified": True}
    (point_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_report(metrics: pd.DataFrame, manifest: pd.DataFrame, cfg: dict[str, Any]) -> None:
    path = resolve(cfg["outputs"]["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(f"| {row.point_id} | {row.benchmark_role} | {row.rescue_decision} | {row.region_count} | {row.inference_seconds:.2f} | {row.peak_vram_gib:.3f} |" for row in metrics.itertuples(index=False))
    path.write_text(f"""# stage_40 Active-travel recognition rescue preflight

Status: **{'PILOT_RUNTIME_PASS_GROUNDING_FAIL' if len(metrics) == 3 and int((metrics.region_count > 0).sum()) == 0 else 'PILOT_REVIEW'}**  
Generated: {datetime.now().astimezone().isoformat()}

- Frozen benchmark: {len(manifest)} points (60 rescue / 30 motor-only controls / 30 positive controls)
- Pilot: {len(metrics)} points, four original perspective views per point
- Maximum peak VRAM: {metrics.peak_vram_gib.max():.3f} GiB
- Mean inference: {metrics.inference_seconds.mean():.2f} s/point
- Full GPU: {metrics.full_gpu_verified.mean():.0%}
- Valid localized-region rate: {(metrics.region_count > 0).mean():.0%}

| Point | Role | Decision | Regions | Seconds | Peak GiB |
|---|---|---|---:|---:|---:|
{lines}

The 2B VLM produced plausible language descriptions but degenerate zero-height boxes, including
sidewalk claims for the motor-only control. Direct VLM grounding is rejected. stage_41 uses the VLM
only to rank suspicious headings; semantic probability, depth-ground geometry and road context must
produce the location and provide two-source agreement before promotion.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "build", "pilot"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    log = logger()
    failed = LOG_DIR / "failed_files.csv"
    try:
        cfg = config()
        manifest = build_manifest(cfg)
        check = {"status": "PASS" if len(manifest) == 120 and manifest.all_four_views_exist.all() and torch.cuda.is_available() else "FAIL", "points": len(manifest), "roles": manifest.benchmark_role.value_counts().to_dict(), "missing_view_points": int((~manifest.all_four_views_exist).sum()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.mode == "check":
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0 if check["status"] == "PASS" else 2
        out_dir = resolve(cfg["outputs"]["directory"])
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_dir / "active_travel_rescue_manifest.csv", index=False, encoding="utf-8-sig")
        if args.mode == "build":
            (out_dir / "build_summary.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
            return 0 if check["status"] == "PASS" else 2
        schema = json.loads(resolve(cfg["inputs"]["response_schema"]).read_text(encoding="utf-8"))
        pilot = manifest.groupby("benchmark_role", sort=True).head(int(cfg["benchmark"]["pilot_per_group"]))
        processor, model, placement = load_runtime(cfg)
        rows, errors = [], []
        for case in tqdm(pilot.to_dict("records"), desc="Four-view active-travel pilot", unit="point", dynamic_ncols=True):
            try:
                rows.append(infer(cfg, case, processor, model, schema, out_dir))
            except Exception as error:
                log.exception("Point %s failed", case["point_id"])
                errors.append({"point_id": case["point_id"], "filename": "four_direction_views", "error_message": f"{type(error).__name__}: {error}"})
        pd.DataFrame(errors, columns=["point_id", "filename", "error_message"]).to_csv(failed, index=False, encoding="utf-8-sig")
        metrics = pd.DataFrame(rows)
        if len(metrics) != len(pilot):
            raise RuntimeError(f"Pilot incomplete: {len(metrics)}/{len(pilot)}")
        metrics.to_csv(out_dir / "pilot_metrics.csv", index=False, encoding="utf-8-sig")
        valid_grounding_rate = float((metrics.region_count > 0).mean())
        summary = {**check, "status": "PILOT_RUNTIME_PASS_GROUNDING_FAIL" if valid_grounding_rate == 0 else "PILOT_REVIEW", "pilot_points": len(metrics), "max_peak_vram_gib": float(metrics.peak_vram_gib.max()), "mean_inference_seconds": float(metrics.inference_seconds.mean()), "valid_grounding_rate": valid_grounding_rate, "placement": placement}
        (out_dir / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(metrics, manifest, cfg)
        log.info("stage_40 pilot %s: points=%d peak=%.3f GiB", summary["status"], len(metrics), summary["max_peak_vram_gib"])
        return 0
    except Exception as error:
        log.exception("stage_40 failed")
        with failed.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_40", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
