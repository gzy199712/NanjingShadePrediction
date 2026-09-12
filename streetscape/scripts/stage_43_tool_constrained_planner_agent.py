"""stage_43: multimodal RAG plus a deterministic tool-constrained planning controller."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetscape.scripts.stage_38_qwen3vl_quantized_single_point import (  # noqa: E402
    assert_model_on_cuda,
    normalize_payload,
    parse_json_object,
    resize_for_vlm,
    schema_is_valid,
    validate_schema,
)
from streetscape.scripts.stage_42b_qwen3vl_embedding_preflight import (  # noqa: E402
    OfficialQwen3VLEmbedder,
    assert_cuda,
)


CONFIG = ROOT / "streetscape/configs/tool_constrained_planner_agent.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_43"


def resolve(value: str) -> Path:
    return ROOT / value


def cached_snapshot(model_id: str) -> str:
    """Resolve an already-downloaded Hugging Face model without any network metadata request."""
    cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))
    revision = (cache / "refs" / "main").read_text(encoding="utf-8").strip()
    snapshot = cache / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    return str(snapshot)


def make_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_43")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_43.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def clean_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def select_pilot(benchmark: pd.DataFrame, role_counts: dict[str, int]) -> pd.DataFrame:
    selections = []
    for role, count in role_counts.items():
        members = benchmark.loc[benchmark.benchmark_role.eq(role)].sort_values("point_id", kind="stable")
        if len(members) < int(count):
            raise RuntimeError(f"Pilot role {role} has {len(members)} rows; needs {count}")
        selections.append(members.head(int(count)))
    return pd.concat(selections, ignore_index=True)


def tool_context(row: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "point_id", "month", "SVF", "GVI", "fclass", "motor_vehicle_only_excluded",
        "thermal_comfort_relevant", "retained_walkable_ratio", "walkable_direction_count",
        "tree_evidence_direction_count", "existing_tree_evidence", "semantic_sky_ratio",
        "semantic_vegetation_ratio", "shade_pred_mean", "tmrt_pred_mean", "utci_pred_mean",
        "march_phenology_uncertain", "optimization_eligible_final",
    ]
    return {field: clean_value(row.get(field)) for field in fields}


def retrieve_balanced(
    query: np.ndarray, case_embeddings: np.ndarray, case_index: pd.DataFrame,
    point_id: str, per_class: int,
) -> list[dict[str, Any]]:
    scores = case_embeddings @ query
    rows = []
    for label in ("STRONG_WALKABLE_POSITIVE", "MOTOR_ONLY_NEGATIVE", "AMBIGUOUS_WALKABLE_REVIEW"):
        candidates = np.flatnonzero(case_index.case_label.to_numpy() == label)
        candidates = np.array([idx for idx in candidates if str(case_index.iloc[idx].point_id) != point_id], dtype=int)
        ordered = candidates[np.argsort(-scores[candidates])[:per_class]]
        for idx in ordered:
            record = case_index.iloc[int(idx)]
            rows.append({
                "case_id": str(record.case_id),
                "case_label": str(record.case_label),
                "similarity": round(float(scores[idx]), 6),
                "case_text": str(record.case_text),
            })
    return rows


def tool_calls(context: dict[str, Any], retrieved: list[dict[str, Any]]) -> list[dict[str, str]]:
    motor = bool_value(context.get("motor_vehicle_only_excluded"))
    thermal = bool_value(context.get("thermal_comfort_relevant"))
    svf = context.get("SVF")
    walkable = context.get("walkable_direction_count")
    month = context.get("month")
    return [
        {"tool": "road_access_gate", "status": "STOP" if motor or not thermal else "PASS", "result": f"motor_excluded={motor}; thermal_relevant={thermal}"},
        {"tool": "measure_svf", "status": "PASS" if svf is not None else "UNCERTAIN", "result": f"SVF={svf}"},
        {"tool": "inspect_semantic_depth", "status": "PASS" if walkable is not None and float(walkable) > 0 else "UNCERTAIN", "result": f"walkable_directions={walkable}; retained_ratio={context.get('retained_walkable_ratio')}"},
        {"tool": "inspect_phenology", "status": "UNCERTAIN" if month in {11, 12, 1, 2} or bool_value(context.get("march_phenology_uncertain")) else "PASS", "result": f"month={month}; march_uncertain={context.get('march_phenology_uncertain')}"},
        {"tool": "retrieve_similar_cases", "status": "PASS", "result": "balanced_positive_negative_abstention=" + "|".join(item["case_id"] for item in retrieved)},
    ]


def prompt(point_id: str, context: dict[str, Any], retrieved: list[dict[str, Any]]) -> str:
    compact_retrieval = [
        {"case_id": item["case_id"], "label": item["case_label"], "similarity": item["similarity"], "summary": item["case_text"][:260]}
        for item in retrieved
    ]
    return f"""You are a tool-constrained urban shade planning reasoner. Inspect the north-aligned panorama.
North is at the left/right seam. Tool measurements: {json.dumps(context, ensure_ascii=False, allow_nan=False)}
Retrieved local counterexamples: {json.dumps(compact_retrieval, ensure_ascii=False, allow_nan=False)}

Propose one JSON object only using the planner-agent v1 fields: schema_version, point_id, scene_type,
active_travel_eligible, shade_need, existing_canopy, recommended_action, target_sectors, confidence,
evidence, requested_tools, warnings. point_id must be '{point_id}'. Use retrieved_case as evidence only
when it supports analogy, never as ground truth. Motor-only, already shaded, covered, winter leaf-off and
unlocalized walkable space must not receive an intervention. Do not invent geometry or thermal improvement.
Target sectors are provisional. Use English, at most 3 evidence entries, and stop after the JSON object."""


def policy_controller(context: dict[str, Any], proposal: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    motor = bool_value(context.get("motor_vehicle_only_excluded"))
    thermal = bool_value(context.get("thermal_comfort_relevant"))
    svf = context.get("SVF")
    month = context.get("month")
    phenology = bool_value(context.get("march_phenology_uncertain")) or month in set(policy["winter_months"])
    walkable = context.get("walkable_direction_count")
    tree_directions = context.get("tree_evidence_direction_count")
    existing_tree = bool_value(context.get("existing_tree_evidence"))
    optimization = bool_value(context.get("optimization_eligible_final"))
    if motor or not thermal:
        state, action, reason, next_tool = "EXCLUDED", "abstain", "road_access_hard_gate", "none"
    elif svf is not None and float(svf) <= float(policy["low_svf_no_intervention_maximum"]):
        state, action, reason, next_tool = "NO_INTERVENTION", "maintain", "svf_indicates_existing_enclosure_or_shade", "none"
    elif phenology:
        state, action, reason, next_tool = "PHENOLOGY_REVIEW", "planner_review", "leaf_off_or_phenology_uncertain", "detect_existing_canopy"
    elif walkable is None or float(walkable) < float(policy["minimum_walkable_direction_count"]):
        state, action, reason, next_tool = "NEEDS_LOCALIZATION", "abstain", "walkable_space_not_geometrically_localized", "locate_walkable_area"
    elif existing_tree and tree_directions is not None and float(tree_directions) >= float(policy["existing_tree_direction_minimum"]):
        state, action, reason, next_tool = "EXISTING_TREE_REVIEW", "maintain", "existing_tree_row_or_growth_requires_lifecycle_review", "planner_review"
    elif optimization:
        state, action, reason, next_tool = "READY_FOR_LAYOUT", "tree_first", "two_family_evidence_supports_layout_preflight", "propose_tree_layout"
    else:
        state, action, reason, next_tool = "PLANNER_REVIEW", "planner_review", "evidence_does_not_satisfy_automatic_layout_gate", "planner_review"
    return {"state": state, "final_action": action, "stop_reason": reason, "generation_unlocked": False, "next_tool": next_tool}


def write_report(summary: dict[str, Any], results: pd.DataFrame, path: Path) -> None:
    table = "\n".join(
        f"| {row.point_id} | {row.benchmark_role} | {row.agent_action} | {row.policy_state} | {row.final_action} | {row.stop_reason} |"
        for row in results.itertuples(index=False)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# stage_43 Tool-constrained planner Agent

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Pilot points: {summary['pilot_points']}
- Successful / failed: {summary['successful_points']} / {summary['failed_points']}
- Proposal schema rate: {summary['proposal_schema_rate']:.1%}
- Trace schema rate: {summary['trace_schema_rate']:.1%}
- Hard-gate safety rate: {summary['hard_gate_safety_rate']:.1%}
- Tool-trace rate: {summary['tool_trace_rate']:.1%}
- Maximum peak VRAM: {summary['max_peak_vram_gib']:.3f} GiB
- Mean reasoner inference: {summary['mean_inference_seconds']:.2f} s/point

| Point | Frozen benchmark role | Agent proposal | Controller state | Final action | Stop reason |
|---|---|---|---|---|---|
{table}

The model proposes and explains; the controller owns hard gates. Retrieved cases support analogy but
cannot override road access, SVF, phenology or localization tools. All target sectors remain provisional,
and generation stays locked until the later geometry-layout gate.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    log = make_logger()
    failed_path = LOG_DIR / "failed_files.csv"
    try:
        cfg = load_config()
        for path_value in cfg["inputs"].values():
            if not resolve(path_value).is_file():
                raise FileNotFoundError(resolve(path_value))
        benchmark = pd.read_csv(resolve(cfg["inputs"]["benchmark"]), dtype={"point_id": str})
        decisions = pd.read_csv(resolve(cfg["inputs"]["decisions"]), dtype={"point_id": str}, low_memory=False)
        pilot = select_pilot(benchmark, cfg["pilot"]["roles"])
        pilot = pilot.drop(columns=[column for column in decisions.columns if column in pilot.columns and column != "point_id"]).merge(decisions, on="point_id", how="left", validate="one_to_one")
        case_index = pd.read_csv(resolve(cfg["inputs"]["case_index"]), dtype={"point_id": str})
        case_embeddings = np.load(resolve(cfg["inputs"]["case_embeddings"])).astype(np.float32)
        proposal_schema = json.loads(resolve(cfg["inputs"]["proposal_schema"]).read_text(encoding="utf-8"))
        trace_schema = json.loads(resolve(cfg["inputs"]["trace_schema"]).read_text(encoding="utf-8"))
        # Resolve the local proposal schema explicitly so validation is fully offline and deterministic.
        trace_schema["properties"]["agent_proposal"] = proposal_schema
        check = {"status": "PASS", "pilot_points": len(pilot), "rag_cases": len(case_index), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if not torch.cuda.is_available() or len(pilot) != sum(cfg["pilot"]["roles"].values()) or len(case_index) != len(case_embeddings):
            check["status"] = "FAIL"
        if args.mode == "check":
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0 if check["status"] == "PASS" else 2
        if check["status"] != "PASS":
            raise RuntimeError(f"Preflight failed: {check}")
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        output = resolve(cfg["outputs"]["directory"])
        if output.exists() and any(output.iterdir()) and not (args.overwrite or args.resume):
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
        output.mkdir(parents=True, exist_ok=True)
        response_dir = output / "responses"
        response_dir.mkdir(exist_ok=True)
        existing_rows = []
        existing_results = output / "agent_pilot_results.csv"
        if args.resume and existing_results.is_file():
            existing_rows = pd.read_csv(existing_results, dtype={"point_id": str}).to_dict("records")
        completed_ids = {str(row["point_id"]) for row in existing_rows}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log.info("Phase 1/2: embedding %d multimodal agent queries", len(pilot))
        embedder = OfficialQwen3VLEmbedder(cfg["models"]["embedder"], cfg["models"]["retrieval_instruction"], args.offline)
        integrity = {name: len(embedder.loading_info.get(name, [])) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        if any(integrity.values()):
            raise RuntimeError(f"Embedding checkpoint integrity failed: {integrity}")
        embedding_placement = assert_cuda(embedder.model)
        query_vectors, retrieval_by_point = [], {}
        for row in tqdm(pilot.to_dict("records"), desc="Retrieving balanced cases", unit="point", dynamic_ncols=True):
            context = tool_context(row)
            query = embedder.encode(json.dumps(context, ensure_ascii=False, allow_nan=False), row["output_path"], int(cfg["models"]["embedding_dimension"]))
            query_vectors.append(query)
            retrieval_by_point[str(row["point_id"])] = retrieve_balanced(query, case_embeddings, case_index, str(row["point_id"]), int(cfg["pilot"]["retrieved_per_class"]))
        embedding_peak = torch.cuda.max_memory_reserved() / 1024**3
        del embedder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log.info("Phase 2/2: loading reasoner after releasing embedding model")
        reasoner_source = cached_snapshot(cfg["models"]["reasoner"]) if args.offline else cfg["models"]["reasoner"]
        processor = AutoProcessor.from_pretrained(reasoner_source, local_files_only=args.offline)
        reasoner = Qwen3VLForConditionalGeneration.from_pretrained(
            reasoner_source, dtype=torch.bfloat16, attn_implementation=cfg["runtime"]["attention"], low_cpu_mem_usage=True, local_files_only=args.offline
        ).to("cuda").eval()
        reasoner_placement = assert_model_on_cuda(reasoner)
        rows, failures = list(existing_rows), []
        pending_records = [row for row in pilot.to_dict("records") if str(row["point_id"]) not in completed_ids]
        for row in tqdm(pending_records, desc="Running planning Agent", unit="point", dynamic_ncols=True):
            point_id = str(row["point_id"])
            try:
                case_dir = response_dir / point_id
                case_dir.mkdir(parents=True, exist_ok=True)
                image_path = case_dir / f"{point_id}_agent_input.jpg"
                resize_for_vlm(Path(row["output_path"]), image_path, 1536, 512)
                context = tool_context(row)
                retrieved = retrieval_by_point[point_id]
                messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt(point_id, context, retrieved)}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
                if any(value.device.type != "cuda" for value in inputs.values() if isinstance(value, torch.Tensor)):
                    raise RuntimeError("Inference tensor outside CUDA")
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = reasoner.generate(**inputs, max_new_tokens=int(cfg["runtime"]["max_new_tokens"]), do_sample=False, use_cache=True)
                inference_seconds = time.perf_counter() - started
                trimmed = generated[:, inputs["input_ids"].shape[1]:]
                raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                raw_payload = parse_json_object(raw_text)
                raw_valid, _ = schema_is_valid(raw_payload, proposal_schema)
                proposal, normalization_events = normalize_payload(raw_payload, point_id)
                validate_schema(proposal, proposal_schema)
                policy = policy_controller(context, proposal, cfg["policy"])
                calls = tool_calls(context, retrieved)
                trace = {
                    "schema_version": "2.0", "point_id": point_id, "tool_calls": calls,
                    "retrieved_cases": [{key: item[key] for key in ("case_id", "case_label", "similarity")} for item in retrieved],
                    "agent_proposal": proposal, "policy_decision": policy,
                    "warnings": ["Target sectors remain provisional until spherical geometry localization."] + (["Agent proposal required deterministic schema normalization."] if normalization_events else []),
                }
                jsonschema.validate(trace, trace_schema)
                (case_dir / "raw_agent_response.txt").write_text(raw_text, encoding="utf-8")
                (case_dir / "agent_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
                peak = torch.cuda.max_memory_reserved() / 1024**3
                rows.append({
                    "point_id": point_id, "benchmark_role": row["benchmark_role"], "agent_action": proposal["recommended_action"],
                    "policy_state": policy["state"], "final_action": policy["final_action"], "stop_reason": policy["stop_reason"],
                    "next_tool": policy["next_tool"], "generation_unlocked": False, "raw_schema_valid": raw_valid,
                    "proposal_schema_valid": True, "trace_schema_valid": True, "tool_trace_complete": len(calls) >= 3,
                    "hard_gate_safe": not (bool_value(context.get("motor_vehicle_only_excluded")) and policy["state"] != "EXCLUDED"),
                    "normalization_required": bool(normalization_events), "inference_seconds": inference_seconds, "peak_vram_gib": peak,
                })
            except Exception as error:
                log.exception("Point %s failed", point_id)
                failures.append({"point_id": point_id, "filename": row["output_path"], "error_message": f"{type(error).__name__}: {error}"})
                torch.cuda.empty_cache()
        results = pd.DataFrame(rows)
        results.to_csv(output / "agent_pilot_results.csv", index=False, encoding="utf-8-sig")
        np.save(output / "query_embeddings.npy", np.stack(query_vectors).astype(np.float32))
        pd.DataFrame(failures, columns=["point_id", "filename", "error_message"]).to_csv(failed_path, index=False, encoding="utf-8-sig")
        total = len(pilot)
        metrics = {
            "proposal_schema_rate": float(results.proposal_schema_valid.mean()) if len(results) else 0.0,
            "trace_schema_rate": float(results.trace_schema_valid.mean()) if len(results) else 0.0,
            "hard_gate_safety_rate": float(results.hard_gate_safe.mean()) if len(results) else 0.0,
            "tool_trace_rate": float(results.tool_trace_complete.mean()) if len(results) else 0.0,
            "failure_rate": (total - len(results)) / total,
        }
        gates = cfg["quality_gates"]
        passed = metrics["proposal_schema_rate"] >= gates["normalized_schema_rate"] and metrics["trace_schema_rate"] >= gates["trace_schema_rate"] and metrics["hard_gate_safety_rate"] >= gates["hard_gate_safety_rate"] and metrics["tool_trace_rate"] >= gates["tool_trace_rate"] and metrics["failure_rate"] <= gates["maximum_failure_rate"]
        summary = {
            "status": "PILOT_PASS" if passed else "PILOT_FAIL", "generated": datetime.now().astimezone().isoformat(),
            "pilot_points": total, "successful_points": len(results), "failed_points": len(failures), **metrics,
            "embedding_checkpoint_integrity": integrity, "embedding_placement": embedding_placement,
            "reasoner_placement": reasoner_placement, "embedding_peak_vram_gib": embedding_peak,
            "max_peak_vram_gib": float(results.peak_vram_gib.max()) if len(results) else 0.0,
            "mean_inference_seconds": float(results.inference_seconds.mean()) if len(results) else 0.0,
            "generation_unlock_count": int(results.generation_unlocked.sum()) if len(results) else 0,
        }
        if max(summary["embedding_peak_vram_gib"], summary["max_peak_vram_gib"]) > float(cfg["runtime"]["maximum_peak_vram_gib"]):
            raise RuntimeError("Peak VRAM exceeded gate")
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(summary, results, resolve(cfg["outputs"]["report"]))
        log.info("stage_43 %s: %s", summary["status"], summary)
        return 0 if passed else 2
    except Exception as error:
        log.exception("stage_43 failed")
        with failed_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_43", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
