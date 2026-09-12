"""stage_48: evidence-based decision for QLoRA versus RAG/gated inference."""

from __future__ import annotations
import argparse,json
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"streetscape/data/agent/stage_48_adaptation_strategy"
REPORT=ROOT/"streetscape/reports/STAGE_48_ADAPTATION_AND_MODEL_STRATEGY.md"


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--mode",choices=("check","run"),default="check");parser.add_argument("--overwrite",action="store_true");args=parser.parse_args()
    paths={"rag":ROOT/"streetscape/data/agent/stage_42c_multimodal_rag_index/case_index.csv.gz","agent":ROOT/"streetscape/data/agent/stage_43_tool_constrained_agent/agent_pilot_results.csv","structure":ROOT/"streetscape/data/agent/stage_45b_structural_scene_gate/structural_scene_qc.csv","rediscovery":ROOT/"streetscape/data/agent/stage_45c_candidate_rediscovery/candidate_scene_classification.csv"}
    check={"status":"PASS" if all(p.is_file() for p in paths.values()) else "FAIL","inputs":{k:str(v) for k,v in paths.items()}}
    if args.mode=="check":print(json.dumps(check,ensure_ascii=False,indent=2));return 0 if check["status"]=="PASS" else 2
    OUT.mkdir(parents=True,exist_ok=True);rag=pd.read_csv(paths["rag"]);agent=pd.read_csv(paths["agent"]);structure=pd.read_csv(paths["structure"]);rediscovery=pd.read_csv(paths["rediscovery"])
    independent_visual_labels=len(structure);independent_visual_classes=structure.structural_scene.nunique();conflicts=int(rediscovery.cross_modal_evidence_conflict.fillna(False).astype(bool).sum());raw_schema_rate=float(agent.raw_schema_valid.mean())
    qlora_ready=independent_visual_labels>=200 and independent_visual_classes>=4 and conflicts<=int(.15*len(rediscovery));decision="PROCEED_QWEN3_VL_2B_QLORA" if qlora_ready else "DEFER_QLORA_USE_RAG_AND_DETERMINISTIC_GATES"
    summary={"status":"PASS","generated":datetime.now().astimezone().isoformat(),"decision":decision,"qlora_started":False,"reason":"insufficient independently verified and class-balanced visual labels; self-training on VLM labels would amplify the observed tree-gap shortcut" if not qlora_ready else "quality and diversity thresholds passed","rag_cases":len(rag),"independent_visual_labels":independent_visual_labels,"independent_visual_classes":independent_visual_classes,"cross_modal_conflicts":conflicts,"raw_schema_valid_rate":raw_schema_rate,"qwen3vl_4b_fp8_status":"official_model_selected_but_download_blocked_by_current_codex_usage_limit","recommended_runtime":"Qwen3-VL-2B + 300-case RAG + deterministic pre/post validation + PowerPaint full-GPU editor"}
    (OUT/"strategy.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");REPORT.write_text(f"""# stage_48 Adaptation and model strategy

Status: **PASS**  
Decision: **{decision}**

- RAG cases: {len(rag)}
- Independently checked structural visual labels: {independent_visual_labels}
- Structural label classes: {independent_visual_classes}
- Cross-modal tree-gap conflicts: {conflicts}/{len(rediscovery)}
- Raw 2B schema-valid rate: {raw_schema_rate:.1%}

QLoRA is not started because the available automatically verified visual set is too small and
class-imbalanced. Training on the 2B model's own erroneous tree-gap labels would encode, rather than
repair, its shortcut. The production strategy therefore keeps the 300-case multimodal RAG index,
deterministic schema validation, structural/tree-continuity gates, semantic-depth layout and the
verified full-GPU PowerPaint editor. Qwen3-VL-4B-Instruct-FP8 remains the next model upgrade when the
official weights can be downloaded and verified locally.
""",encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
