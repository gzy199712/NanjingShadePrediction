"""stage_45b: reject elevated, covered and blocked layouts with a short-label VLM gate."""

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

CONFIG = ROOT / "streetscape/configs/structural_scene_gate.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_45b"


def resolve(value): return ROOT / value


def logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True); log=logging.getLogger("stage_45b"); log.handlers.clear(); log.setLevel(logging.INFO)
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR/"stage_45b.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)): handler.setFormatter(fmt);log.addHandler(handler)
    return log


def prompt(labels):
    return f"""Classify the physical support beneath the candidate walking/cycling area for planting a row of tall shade trees.
Use the panorama for global context and the perspective window for local context. Distinguish an ordinary ground-level street from a bridge/elevated deck, tunnel/covered road, or construction/barrier-blocked area.
Output exactly one token and nothing else: {', '.join(labels)}.
If the evidence is ambiguous output UNCERTAIN. Do not infer that a visible narrow walkway is plantable."""


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--mode",choices=("check","run"),default="check");parser.add_argument("--overwrite",action="store_true");parser.add_argument("--layout-directory",type=Path);parser.add_argument("--windows-directory",type=Path);parser.add_argument("--benchmark",type=Path);parser.add_argument("--output-directory",type=Path);parser.add_argument("--report",type=Path);args=parser.parse_args();log=logger();failed=LOG_DIR/"failed_files.csv"
    try:
        cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));layout_root=args.layout_directory.resolve() if args.layout_directory else resolve(cfg["inputs"]["layout_directory"]);window_root=args.windows_directory.resolve() if args.windows_directory else resolve(cfg["inputs"]["windows_directory"])
        benchmark_path=args.benchmark.resolve() if args.benchmark else resolve(cfg["inputs"]["benchmark"]);qc=pd.read_csv(layout_root/"layout_qc.csv",dtype={"point_id":str});benchmark=pd.read_csv(benchmark_path,dtype={"point_id":str})
        candidates=qc.loc[qc.status.eq("LAYOUT_READY")].merge(benchmark[["point_id","output_path"]],on="point_id",validate="one_to_one")
        check={"status":"PASS" if torch.cuda.is_available() and len(candidates)>=1 else "FAIL","candidate_points":len(candidates),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.mode=="check": print(json.dumps(check,ensure_ascii=False,indent=2));return 0 if check["status"]=="PASS" else 2
        if check["status"]!="PASS": raise RuntimeError(check)
        out=args.output_directory.resolve() if args.output_directory else resolve(cfg["outputs"]["directory"])
        if out.exists() and any(out.iterdir()) and not args.overwrite: raise FileExistsError(f"Output exists; use --overwrite: {out}")
        out.mkdir(parents=True,exist_ok=True);source=cached_snapshot(cfg["model"]["id"]);processor=AutoProcessor.from_pretrained(source,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(source,dtype=torch.bfloat16,attn_implementation="sdpa",low_cpu_mem_usage=True,local_files_only=True).to("cuda").eval();placement=assert_model_on_cuda(model)
        rows=[];failures=[];torch.cuda.reset_peak_memory_stats();started=time.perf_counter();allowed=set(cfg["labels"])
        for row in tqdm(candidates.to_dict("records"),desc="Classifying structural scenes",unit="point",dynamic_ncols=True):
            point_id=str(row["point_id"])
            try:
                heading=int(row["selected_heading"]);window=window_root/"points"/point_id/"windows"/f"heading_{heading:03d}.png"
                messages=[{"role":"user","content":[{"type":"image","image":str(row["output_path"])},{"type":"image","image":str(window)},{"type":"text","text":prompt(cfg["labels"])}]}]
                inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to("cuda")
                tic=time.perf_counter()
                with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=int(cfg["runtime"]["max_new_tokens"]),do_sample=False,use_cache=True)
                trimmed=generated[:,inputs["input_ids"].shape[1]:];raw=processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip().upper()
                matches=[label for label in cfg["labels"] if label in raw];label=matches[0] if len(matches)==1 else "UNCERTAIN"
                rows.append({"point_id":point_id,"selected_heading":heading,"raw_response":raw,"structural_scene":label,"planting_support_pass":label=="GROUND_STREET","final_layout_status":"LAYOUT_READY_FOR_IMAGE_EDITOR" if label=="GROUND_STREET" else "LAYOUT_REJECTED_STRUCTURAL_SCENE","stop_reason":None if label=="GROUND_STREET" else label.lower(),"inference_seconds":time.perf_counter()-tic,"full_gpu_verified":True})
            except Exception as error:
                log.exception("Point %s failed",point_id);failures.append({"point_id":point_id,"filename":"structural_scene_gate","error_message":f"{type(error).__name__}: {error}"})
        result=pd.DataFrame(rows);result.to_csv(out/"structural_scene_qc.csv",index=False,encoding="utf-8-sig");pd.DataFrame(failures,columns=["point_id","filename","error_message"]).to_csv(failed,index=False,encoding="utf-8-sig")
        peak=torch.cuda.max_memory_reserved()/1024**3;passed=not failures and len(result)==len(candidates) and peak<=float(cfg["runtime"]["maximum_peak_vram_gib"])
        summary={"status":"PASS" if passed else "FAIL","generated":datetime.now().astimezone().isoformat(),"runtime_seconds":time.perf_counter()-started,"candidate_points":len(candidates),"classified_points":len(result),"failure_count":len(failures),"scene_counts":result.structural_scene.value_counts().to_dict(),"planting_support_pass_count":int(result.planting_support_pass.sum()),"full_gpu_verified":True,"parameter_placement":placement,"peak_vram_gib":peak}
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        report=args.report.resolve() if args.report else resolve(cfg["outputs"]["report"]);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(f"""# stage_45b Multimodal structural scene gate

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Candidate layouts: {summary['candidate_points']}
- Scene counts: {summary['scene_counts']}
- Ground-street layouts retained: {summary['planting_support_pass_count']}
- Full GPU: {summary['full_gpu_verified']}
- Peak reserved VRAM: {summary['peak_vram_gib']:.3f} GiB

Only `GROUND_STREET` proceeds to image editing. Bridge/elevated, tunnel/covered, construction/blocked
and uncertain scenes stop automatically. This gate addresses visually walkable but physically
non-plantable infrastructure that semantic sidewalk masks alone cannot distinguish.
""",encoding="utf-8")
        log.info("stage_45b %s: %s",summary["status"],summary);return 0 if passed else 2
    except Exception as error:
        log.exception("stage_45b failed")
        with failed.open("a",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=["point_id","filename","error_message"])
            if handle.tell()==0: writer.writeheader()
            writer.writerow({"point_id":"","filename":"stage_45b","error_message":f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__=="__main__": raise SystemExit(main())
