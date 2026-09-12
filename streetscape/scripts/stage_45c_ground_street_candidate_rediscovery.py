"""stage_45c: rediscover genuinely ground-level public tree-gap scenes citywide."""

from __future__ import annotations

import argparse, csv, json, logging, sys, time, traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from streetscape.scripts.stage_38_qwen3vl_quantized_single_point import assert_model_on_cuda  # noqa: E402
from streetscape.scripts.stage_43_tool_constrained_planner_agent import cached_snapshot  # noqa: E402

CONFIG = ROOT / "streetscape/configs/ground_street_candidate_rediscovery.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_45c"


def resolve(value): return ROOT / value


def get_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_45c"); log.handlers.clear(); log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_45c.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); log.addHandler(handler)
    return log


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): rows.append(json.loads(line))
    return pd.DataFrame(rows)


def build_pool(decisions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["screening"]
    work = decisions.copy()
    truth = lambda name: work[name].fillna(False).astype(bool)
    keep = (
        ~work["month"].isin(s["excluded_months"])
        & truth("thermal_comfort_relevant")
        & ~truth("motor_vehicle_only_excluded")
        & (pd.to_numeric(work["SVF"], errors="coerce") >= float(s["minimum_svf"]))
        & (pd.to_numeric(work["GVI"], errors="coerce") <= float(s["maximum_gvi"]))
        & (pd.to_numeric(work["walkable_direction_count"], errors="coerce").fillna(0) >= int(s["minimum_walkable_directions"]))
        & (pd.to_numeric(work["retained_walkable_ratio"], errors="coerce").fillna(0) >= float(s["minimum_retained_walkable_ratio"]))
    )
    work = work.loc[keep].copy()
    work["candidate_score"] = (
        work["SVF"].astype(float) * 2.0
        - work["GVI"].astype(float)
        + work["walkable_direction_count"].astype(float).clip(0, 12) / 24.0
        + work["retained_walkable_ratio"].astype(float).clip(0, 0.02) * 10.0
    )
    cell = float(s["spatial_cell_degrees"])
    work["spatial_cell"] = (work["x"].astype(float).div(cell).round().astype(int).astype(str) + "_" + work["y"].astype(float).div(cell).round().astype(int).astype(str))
    work = work.sort_values(["candidate_score", "point_id"], ascending=[False, True])
    work["cell_rank"] = work.groupby("spatial_cell").cumcount()
    return work.loc[work.cell_rank < int(s["maximum_per_spatial_cell"])].head(int(s["ranked_pool_size"])).copy()


def classification_prompt(labels: list[str]) -> str:
    return f"""Classify this north-aligned 360-degree street panorama for a tree-first shade-planning agent.
First identify whether it is an ordinary ground-level PUBLIC walking/cycling street. Then assess the planting state.
GROUND_WALKABLE_TREE_GAP requires visible public sidewalk/cycleway, ordinary rooted ground or planting strip, and a meaningful gap where a row of tall shade trees could continue.
GROUND_WALKABLE_YOUNG_TREE_ROW means a continuous row of young/small or leaf-sparse trees already exists: report growth potential and do not add trees.
GROUND_WALKABLE_EXISTING_CANOPY means mature canopy already serves the walking area.
GROUND_WALKABLE_NO_PLANTING_SPACE means public walking is visible but no credible ground planting corridor is visible.
Motorways, expressways, bridges/elevated decks, tunnels/covered roads, and construction/blocked scenes must never be treated as tree gaps.
Output exactly one token and nothing else: {', '.join(labels)}. If ambiguous output UNCERTAIN."""


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", choices=("check", "run"), default="check"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    log = get_logger(); failed_path = LOG_DIR / "failed_files.csv"
    try:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        decisions = pd.read_csv(resolve(cfg["inputs"]["citywide_decisions"]), low_memory=False, dtype={"point_id": str})
        metadata = read_jsonl(resolve(cfg["inputs"]["panorama_metadata"])); metadata["point_id"] = metadata["point_id"].astype(str)
        pool = build_pool(decisions, cfg).merge(metadata[["point_id", "output_path", "width", "height", "north_at_x0"]], on="point_id", validate="one_to_one")
        pool["panorama_exists"] = pool.output_path.map(lambda value: Path(str(value)).is_file())
        check = {"status": "PASS" if torch.cuda.is_available() and len(pool) >= 1 and pool.panorama_exists.all() else "FAIL", "ranked_pool_points": len(pool), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.mode == "check": print(json.dumps(check, ensure_ascii=False, indent=2)); return 0 if check["status"] == "PASS" else 2
        if check["status"] != "PASS": raise RuntimeError(check)
        out = resolve(cfg["outputs"]["directory"])
        if out.exists() and any(out.iterdir()) and not args.overwrite: raise FileExistsError(f"Output exists; use --overwrite: {out}")
        out.mkdir(parents=True, exist_ok=True)
        source = cached_snapshot(cfg["model"]["id"]); processor = AutoProcessor.from_pretrained(source, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True).to("cuda").eval()
        placement = assert_model_on_cuda(model); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); rows = []; failures = []; labels = cfg["labels"]
        for case in tqdm(pool.to_dict("records"), desc="Rediscovering ground tree gaps", unit="point", dynamic_ncols=True):
            point_id = str(case["point_id"])
            try:
                messages = [{"role": "user", "content": [{"type": "image", "image": str(case["output_path"])}, {"type": "text", "text": classification_prompt(labels)}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
                tic = time.perf_counter()
                with torch.inference_mode(): generated = model.generate(**inputs, max_new_tokens=int(cfg["runtime"]["max_new_tokens"]), do_sample=False, use_cache=True)
                raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().upper()
                matches = [label for label in labels if label in raw]; label = matches[0] if len(matches) == 1 else "UNCERTAIN"
                tree_directions = float(case["tree_evidence_direction_count"]) if pd.notna(case["tree_evidence_direction_count"]) else 0.0
                evidence_conflict = label == "GROUND_WALKABLE_TREE_GAP" and tree_directions >= float(cfg["screening"]["continuous_tree_evidence_direction_minimum"])
                fused_label = "EXISTING_TREE_ROW_EVIDENCE_CONFLICT" if evidence_conflict else label
                eligible = fused_label == "GROUND_WALKABLE_TREE_GAP"
                rows.append({"point_id": point_id, "candidate_rank": len(rows) + 1, "candidate_score": case["candidate_score"], "month": case["month"], "SVF": case["SVF"], "GVI": case["GVI"], "walkable_direction_count": case["walkable_direction_count"], "tree_evidence_direction_count": tree_directions, "retained_walkable_ratio": case["retained_walkable_ratio"], "existing_tree_evidence": case["existing_tree_evidence"], "output_path": case["output_path"], "raw_response": raw, "scene_label": fused_label, "vlm_scene_label": label, "cross_modal_evidence_conflict": evidence_conflict, "layout_eligible": eligible, "automatic_disposition": "PROCEED_TO_25D_LAYOUT" if eligible else "REPORT_ONLY_OR_STOP", "inference_seconds": time.perf_counter() - tic, "full_gpu_verified": True})
            except Exception as error:
                log.exception("Point %s failed", point_id); failures.append({"point_id": point_id, "filename": case["output_path"], "error_message": f"{type(error).__name__}: {error}"})
        result = pd.DataFrame(rows); result.to_csv(out / "candidate_scene_classification.csv", index=False, encoding="utf-8-sig")
        eligible = result.loc[result.layout_eligible].copy()
        eligible.assign(policy_state="READY_FOR_LAYOUT", next_tool="propose_tree_layout")[['point_id','policy_state','next_tool']].to_csv(out / "layout_agent_results.csv", index=False, encoding="utf-8-sig")
        eligible.assign(benchmark_role="GROUND_WALKABLE_TREE_GAP", width=2048, height=512, north_at_x0=True, panorama_exists=True)[['point_id','benchmark_role','output_path','width','height','north_at_x0','panorama_exists']].to_csv(out / "layout_benchmark_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(failures, columns=["point_id", "filename", "error_message"]).to_csv(failed_path, index=False, encoding="utf-8-sig")
        peak = torch.cuda.max_memory_reserved() / 1024**3
        passed = not failures and len(result) == len(pool) and len(eligible) >= int(cfg["quality_gates"]["minimum_tree_gap_candidates"]) and peak <= float(cfg["runtime"]["maximum_peak_vram_gib"])
        summary = {"status": "PASS" if passed else "NO_VALID_GROUND_TREE_GAP", "generated": datetime.now().astimezone().isoformat(), "runtime_seconds": time.perf_counter() - started, "ranked_pool_points": len(pool), "classified_points": len(result), "failure_count": len(failures), "scene_counts": result.scene_label.value_counts().to_dict(), "tree_gap_candidates": len(eligible), "full_gpu_verified": True, "parameter_placement": placement, "peak_vram_gib": peak}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        report = resolve(cfg["outputs"]["report"]); report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"""# stage_45c Ground-street candidate rediscovery

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Ranked citywide pool: {summary['ranked_pool_points']}
- Scene counts: {summary['scene_counts']}
- Ground public tree-gap candidates: {summary['tree_gap_candidates']}
- Full GPU: {summary['full_gpu_verified']}
- Peak reserved VRAM: {summary['peak_vram_gib']:.3f} GiB

This automatic gate separates real ground-level public tree gaps from young-tree rows, existing
canopy, non-plantable sidewalks, motor facilities, bridges, tunnels and construction scenes. Only
`GROUND_WALKABLE_TREE_GAP` is handed to the 2.5D layout stage; no manual review is required.
""", encoding="utf-8")
        log.info("stage_45c %s: %s", summary["status"], summary); return 0 if passed else 2
    except Exception as error:
        log.exception("stage_45c failed")
        with failed_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0: writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_45c", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__": raise SystemExit(main())
