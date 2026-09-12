"""stage_45f: dual-order VLM tree-state consensus before March candidate generation."""
from __future__ import annotations
import argparse,json,sys,time
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from streetscape.scripts.stage_38_qwen3vl_quantized_single_point import assert_model_on_cuda
from streetscape.scripts.stage_43_tool_constrained_planner_agent import cached_snapshot
from routing.app.backend.planner_vlm_agent_worker import TREE,consensus


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--classification",type=Path,required=True);parser.add_argument("--structural-qc",type=Path,required=True);parser.add_argument("--output-directory",type=Path,required=True);parser.add_argument("--overwrite",action="store_true");args=parser.parse_args();classification=pd.read_csv(args.classification,dtype={"point_id":str});structural=pd.read_csv(args.structural_qc,dtype={"point_id":str});cases=classification.loc[classification.layout_eligible.astype(bool)].merge(structural.loc[structural.planting_support_pass.astype(bool),["point_id","selected_heading"]],on="point_id",validate="one_to_one")
    if not torch.cuda.is_available() or not len(cases):raise RuntimeError("CUDA and at least one structural candidate are required")
    out=args.output_directory.resolve();out.mkdir(parents=True,exist_ok=True);source=cached_snapshot("Qwen/Qwen3-VL-2B-Instruct");processor=AutoProcessor.from_pretrained(source,local_files_only=True);model=Qwen3VLForConditionalGeneration.from_pretrained(source,dtype=torch.bfloat16,attn_implementation="sdpa",low_cpu_mem_usage=True,local_files_only=True).to("cuda").eval();placement=assert_model_on_cuda(model);torch.cuda.reset_peak_memory_stats();started=time.perf_counter();rows=[]
    prompt="""Classify tree-first shade intervention. TREE_GAP_WITH_PLANTING_SPACE requires a visible public sidewalk/cycleway, credible rooted ground planting strip, and a real missing segment in a tree row. Bare or small trees already arranged continuously are CONTINUOUS_YOUNG_TREE_ROW and must not receive new trees. Output exactly one token: {labels}."""
    for row in tqdm(cases.to_dict("records"),desc="Checking tree-state consensus",unit="point",dynamic_ncols=True):
        label,raw=consensus(model,processor,Path(row["output_path"]),prompt,TREE);directions=float(row["tree_evidence_direction_count"]);month=int(row["month"]);passed=label=="TREE_GAP_WITH_PLANTING_SPACE" and directions<6 and month not in {11,12,1,2};rows.append({"point_id":row["point_id"],"month":month,"selected_heading":int(row["selected_heading"]),"tree_evidence_direction_count":directions,"forward_response":raw[0],"reverse_response":raw[1],"tree_state_consensus":label,"tree_gap_pass":passed,"final_generation_gate":"READY_FOR_IMAGE_EDITOR" if passed else "SAFE_ABSTENTION_TREE_STATE"})
    result=pd.DataFrame(rows);result.to_csv(out/"tree_state_consensus_qc.csv",index=False,encoding="utf-8-sig")
    gated=structural.merge(result[["point_id","tree_gap_pass","tree_state_consensus"]],on="point_id",validate="one_to_one");gated["planting_support_pass"]=gated["planting_support_pass"].astype(bool)&gated["tree_gap_pass"].astype(bool);gated["final_layout_status"]=gated["planting_support_pass"].map({True:"LAYOUT_READY_FOR_IMAGE_EDITOR",False:"LAYOUT_REJECTED_TREE_STATE_CONSENSUS"});gated.to_csv(out/"generation_structural_qc.csv",index=False,encoding="utf-8-sig")
    summary={"status":"PASS","generated":datetime.now().astimezone().isoformat(),"candidate_points":len(result),"tree_gap_pass_count":int(result.tree_gap_pass.sum()),"state_counts":result.tree_state_consensus.value_counts().to_dict(),"full_gpu_verified":True,"parameter_placement":placement,"peak_vram_gib":torch.cuda.max_memory_reserved()/1024**3,"runtime_seconds":time.perf_counter()-started};(out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
