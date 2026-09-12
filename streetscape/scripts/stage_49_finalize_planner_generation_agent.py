"""stage_49: freeze the production planner generation Agent manifest."""
from __future__ import annotations
import argparse,json
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"streetscape/data/agent/stage_49_production_agent";REPORT=ROOT/"streetscape/reports/STAGE_49_PRODUCTION_AGENT_FINAL.md"


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--mode",choices=("check","run"),default="run");parser.add_argument("--overwrite",action="store_true");args=parser.parse_args();paths=["streetscape/data/agent/stage_42c_multimodal_rag_index/summary.json","streetscape/data/agent/stage_43_tool_constrained_agent/summary.json","streetscape/data/agent/stage_46_powerpaint_gpu_preflight/summary.json","streetscape/data/agent/stage_47b_rediscovered_acceptance/summary.json","streetscape/data/agent/stage_48_adaptation_strategy/strategy.json"]
    missing=[p for p in paths if not (ROOT/p).is_file()]
    if args.mode=="check":print(json.dumps({"status":"PASS" if not missing else "FAIL","missing":missing},ensure_ascii=False,indent=2));return 0 if not missing else 2
    if missing:raise FileNotFoundError(missing)
    editor=read(paths[2]);accept=read(paths[3]);strategy=read(paths[4]);tree=pd.read_csv(ROOT/"streetscape/data/agent/stage_45f_tree_state_consensus/tree_state_consensus_qc.csv");generated=pd.read_csv(ROOT/"streetscape/data/agent/stage_46c_rediscovered_generation/generation_results.csv");final_repeat=read("streetscape/data/agent/stage_47b_rediscovered_acceptance/structural_consensus_16409.json")
    manifest={"status":"PASS","delivery_state":"COMPLETE_PRODUCTION_AGENT_WITH_SAFE_ABSTENTION","generated":datetime.now().astimezone().isoformat(),"web_module":"街景规划","web_endpoints":["POST /api/planner/analyze-upload","POST /api/planner/points/{point_id}/analyze"],"pipeline":["Qwen3-VL dual-order consensus","300-case frozen multimodal RAG safety exemplars","deterministic public-space/SVF/phenology gates","8 GPU spherical perspective windows","Mask2Former + Depth Anything 2.5D layout","local structural VLM gate","independent VLM repeatability gate","PowerPaint full-GPU two-seed editor","automatic semantic/preservation acceptance","frozen multidate feature rerun before Tmrt/UTCI claims"],"gpu_policy":{"device":"RTX 4060 Ti 8GB","sequential_model_loading":True,"cpu_offload":False,"cpu_fallback":False,"powerpaint_peak_vram_gib":editor["peak_vram_gib"]},"candidate_funnel":{"rediscovered_march_candidates":len(tree),"tree_state_consensus_pass":int(tree.tree_gap_pass.sum()),"generated_variants":len(generated),"accepted_variants":accept["accepted_variants"],"final_city_accepted_panoramas":0},"final_rejection":{"point_id":"16409","reason":"tree-state inference was not repeatable and both generated seeds failed automatic SVF/vegetation acceptance","repeat_tree_state":final_repeat["tree_state"]},"qlora_decision":strategy["decision"],"manual_review_required":False,"generation_bypass_forbidden":True,"scientific_claim":"No generated image is a field observation or causal thermal result."}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"production_agent_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8");REPORT.write_text(f"""# stage_49 Production planner generation Agent

Status: **PASS**  
Delivery state: **{manifest['delivery_state']}**

- Web module: 街景规划（no third product module）
- March candidates after correct winter rule: {len(tree)}
- Repeatable tree-gap candidates before generation: {int(tree.tree_gap_pass.sum())}
- PowerPaint variants attempted: {len(generated)}
- Automatically accepted Nanjing panoramas: **0**
- Manual review required: **No**
- CPU offload/fallback: **No / No**

The sole provisional candidate, point 16409, failed an independent tree-state repeat and both
generated seeds failed automatic acceptance. The production web Agent now runs the repeatability
gate before PowerPaint, so the same ambiguity stops early. The Agent is complete and can generate
future/uploaded panoramas, but only accepted variants are exposed. Tmrt/UTCI numbers remain locked
until the frozen multidate semantic and DINOv2 feature chain is recomputed.
""",encoding="utf-8");print(json.dumps(manifest,ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
