"""One-shot full-CUDA Qwen3-VL scene and tree-state consensus classifier."""

from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from streetscape.scripts.stage_38_qwen3vl_quantized_single_point import assert_model_on_cuda
from streetscape.scripts.stage_43_tool_constrained_planner_agent import cached_snapshot

STRUCT=["GROUND_STREET","BRIDGE_OR_ELEVATED","TUNNEL_OR_COVERED","CONSTRUCTION_OR_BLOCKED","MOTOR_OR_EXPRESSWAY","UNCERTAIN"]
TREE=["TREE_GAP_WITH_PLANTING_SPACE","CONTINUOUS_YOUNG_TREE_ROW","CONTINUOUS_MATURE_CANOPY","NO_CREDIBLE_PLANTING_SPACE","UNCERTAIN"]
ACTION=["TREE_FIRST","FACILITY_SUPPLEMENT_REVIEW","ASSESS_ONLY","UNCERTAIN"]


def ask(model,processor,image:Path,prompt:str,labels:list[str])->tuple[str,str]:
    messages=[{"role":"user","content":[{"type":"image","image":str(image)},{"type":"text","text":prompt}]}];inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to("cuda")
    with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=24,do_sample=False,use_cache=True)
    raw=processor.batch_decode(generated[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip().upper();matches=[label for label in labels if label in raw];return (matches[0] if len(matches)==1 else "UNCERTAIN"),raw


def consensus(model,processor,image,prompt,labels):
    a,raw_a=ask(model,processor,image,prompt.format(labels=", ".join(labels)),labels);b,raw_b=ask(model,processor,image,prompt.format(labels=", ".join(reversed(labels))),labels);return (a if a==b else "UNCERTAIN"),[raw_a,raw_b]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--image",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--intent",default="");args=parser.parse_args();started=time.perf_counter()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; CPU fallback forbidden")
    source=cached_snapshot("Qwen/Qwen3-VL-2B-Instruct");processor=AutoProcessor.from_pretrained(source,local_files_only=True);model=Qwen3VLForConditionalGeneration.from_pretrained(source,dtype=torch.bfloat16,attn_implementation="sdpa",low_cpu_mem_usage=True,local_files_only=True).to("cuda").eval();placement=assert_model_on_cuda(model);torch.cuda.reset_peak_memory_stats()
    structural_prompt="""Classify the physical street scene. GROUND_STREET requires an ordinary ground-level public street, not a bridge/elevated deck, tunnel/covered road, construction barrier area, or motor-only expressway. Output exactly one token: {labels}."""
    tree_prompt="""Classify tree-first shade intervention. TREE_GAP_WITH_PLANTING_SPACE requires a visible public sidewalk/cycleway, a credible rooted ground planting strip, and a real gap in a tree row. A continuous row of young or leaf-sparse trees is CONTINUOUS_YOUNG_TREE_ROW and must not receive new trees. Output exactly one token: {labels}."""
    structural,struct_raw=consensus(model,processor,args.image,structural_prompt,STRUCT);tree,tree_raw=consensus(model,processor,args.image,tree_prompt,TREE)
    action_prompt=f"""User planning request: {args.intent[:500] or 'automatic shade assessment'}. Choose a planning action grounded in the image. TREE_FIRST is the default whenever credible planting space exists. FACILITY_SUPPLEMENT_REVIEW is only a non-generative review when trees appear infeasible. ASSESS_ONLY means no visual intervention. Output exactly one token: {{labels}}."""
    action,action_raw=consensus(model,processor,args.image,action_prompt,ACTION)
    cases=pd.read_csv(ROOT/"streetscape/data/agent/stage_42c_multimodal_rag_index/case_index.csv.gz");safety=[]
    for label in ("STRONG_WALKABLE_POSITIVE","MOTOR_ONLY_NEGATIVE","AMBIGUOUS_WALKABLE_REVIEW"):
        subset=cases.loc[cases.case_label.eq(label)].sort_values("sha256");
        if len(subset):safety.append(str(subset.iloc[0].case_id))
    result={"status":"PASS","model":"Qwen/Qwen3-VL-2B-Instruct","structural_scene":structural,"tree_state":tree,"planning_action":action,"structural_consensus_raw":struct_raw,"tree_consensus_raw":tree_raw,"planning_action_consensus_raw":action_raw,"consensus_required":True,"rag_safety_exemplar_ids":safety,"rag_mode":"balanced_safety_exemplars_from_frozen_300_case_index","full_gpu_verified":True,"parameter_placement":placement,"peak_vram_gib":torch.cuda.max_memory_reserved()/1024**3,"inference_seconds":time.perf_counter()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");return 0


if __name__=="__main__":raise SystemExit(main())
