"""stage_47: automatically accept or reject generated panoramas using GPU semantics."""

from __future__ import annotations

import argparse, csv, json, logging, sys, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation

ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/"streetscape/configs/automatic_generation_acceptance.yaml";LOG_DIR=ROOT/"artifacts/logs/streetscape_planning/stage_47"


def resolve(value):return ROOT/value


def get_logger():
    LOG_DIR.mkdir(parents=True,exist_ok=True);log=logging.getLogger("stage_47");log.handlers.clear();log.setLevel(logging.INFO);fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR/"stage_47.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)):handler.setFormatter(fmt);log.addHandler(handler)
    return log


def class_ids(model, names):
    mapping={str(value).lower():int(key) for key,value in model.config.id2label.items()};return [mapping[name.lower()] for name in names if name.lower() in mapping]


def semantic_ratios(path: Path, model, device, cfg):
    rgb=torch.from_numpy(np.asarray(Image.open(path).convert("RGB"),dtype=np.uint8).copy()).permute(2,0,1).unsqueeze(0).to(device,dtype=torch.float32)/255
    rgb=F.interpolate(rgb,(int(cfg["runtime"]["analysis_height"]),int(cfg["runtime"]["analysis_width"])),mode="bilinear",align_corners=False,antialias=True);mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1);std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):output=model(pixel_values=(rgb-mean)/std,pixel_mask=torch.ones(rgb.shape[0],rgb.shape[2],rgb.shape[3],device=device,dtype=torch.bool))
    classes=output.class_queries_logits.float().softmax(-1)[...,:-1];masks=output.masks_queries_logits.float().sigmoid();prob=torch.einsum("bqc,bqhw->bchw",classes,masks);labels=F.interpolate(prob,size=rgb.shape[2:],mode="bilinear",align_corners=False).argmax(1)[0]
    groups={"vegetation":["Vegetation"],"sky":["Sky"],"road":["Road","Lane Marking - General","Lane Marking - Crosswalk"],"building":["Building"]};result={}
    for key,names in groups.items():ids=class_ids(model,names);result[key]=float(sum((labels==index).float().mean() for index in ids)) if ids else 0.0
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--mode",choices=("check","run"),default="check");parser.add_argument("--overwrite",action="store_true");parser.add_argument("--agent-state",type=Path);parser.add_argument("--generation-results",type=Path);parser.add_argument("--output-directory",type=Path);parser.add_argument("--report",type=Path);args=parser.parse_args();log=get_logger();failed=LOG_DIR/"failed_files.csv"
    try:
        cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));state_path=args.agent_state.resolve() if args.agent_state else resolve(cfg["inputs"]["agent_state"]);results_path=args.generation_results.resolve() if args.generation_results else resolve(cfg["inputs"]["generation_results"]);state=json.loads(state_path.read_text(encoding="utf-8"));results=pd.read_csv(results_path,dtype={"point_id":str});model_path=resolve(cfg["model"]["semantic"])
        check={"status":"PASS" if torch.cuda.is_available() and state.get("status")=="PASS" and model_path.is_dir() else "FAIL","generated_variants":len(results),"safe_abstention":len(results)==0,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.mode=="check":print(json.dumps(check,ensure_ascii=False,indent=2));return 0 if check["status"]=="PASS" else 2
        if check["status"]!="PASS":raise RuntimeError(check)
        out=args.output_directory.resolve() if args.output_directory else resolve(cfg["outputs"]["directory"])
        if out.exists() and any(out.iterdir()) and not args.overwrite:raise FileExistsError(f"Output exists; use --overwrite: {out}")
        out.mkdir(parents=True,exist_ok=True);rows=[];failures=[];started=time.perf_counter();peak=0.0
        if len(results):
            device=torch.device("cuda:0");model=Mask2FormerForUniversalSegmentation.from_pretrained(model_path,local_files_only=True).to(device).eval();torch.cuda.reset_peak_memory_stats()
            for record in tqdm(results.to_dict("records"),desc="Accepting generated panoramas",unit="variant",dynamic_ncols=True):
                try:
                    before=semantic_ratios(Path(record["baseline_panorama_path"]),model,device,cfg);after=semantic_ratios(Path(record["panorama_path"]),model,device,cfg);delta_veg=after["vegetation"]-before["vegetation"];delta_svf=before["sky"]-after["sky"];road_loss=before["road"]-after["road"];building_loss=before["building"]-after["building"];building_gain=after["building"]-before["building"]
                    gates={"technical":bool(record["technical_pass"]),"outside_exact":int(record["outside_mask_max_difference"])==0,"north_seam":bool(record["north_seam_unchanged"]),"vegetation_gain":delta_veg>=float(cfg["quality_gates"]["minimum_vegetation_ratio_increase"]),"svf_reduction":delta_svf>=float(cfg["quality_gates"]["minimum_svf_reduction"]),"road_preservation":road_loss<=float(cfg["quality_gates"]["maximum_road_ratio_loss"]),"building_preservation":building_loss<=float(cfg["quality_gates"]["maximum_building_ratio_loss"]),"building_no_growth":building_gain<=float(cfg["quality_gates"].get("maximum_building_ratio_increase",0.005))};accepted=all(gates.values())
                    rows.append({**record,"baseline_svf":before["sky"],"scenario_svf":after["sky"],"svf_reduction":delta_svf,"baseline_vegetation_ratio":before["vegetation"],"scenario_vegetation_ratio":after["vegetation"],"vegetation_ratio_increase":delta_veg,"road_ratio_loss":road_loss,"building_ratio_loss":building_loss,"building_ratio_increase":building_gain,"automatic_acceptance":accepted,"acceptance_gates":json.dumps(gates,ensure_ascii=False,separators=(",",":")),"thermal_evaluation_state":"READY_FOR_FROZEN_MULTIDATE_FEATURE_RERUN" if accepted else "NOT_APPLICABLE_REJECTED"})
                except Exception as error:log.exception("Acceptance failed");failures.append({"point_id":record.get("point_id",""),"filename":record.get("panorama_path",""),"error_message":f"{type(error).__name__}: {error}"})
            peak=torch.cuda.max_memory_reserved()/1024**3
        columns=list(results.columns)+["baseline_svf","scenario_svf","svf_reduction","baseline_vegetation_ratio","scenario_vegetation_ratio","vegetation_ratio_increase","road_ratio_loss","building_ratio_loss","building_ratio_increase","automatic_acceptance","acceptance_gates","thermal_evaluation_state"]
        pd.DataFrame(rows,columns=columns).to_csv(out/"acceptance_results.csv",index=False,encoding="utf-8-sig");pd.DataFrame(failures,columns=["point_id","filename","error_message"]).to_csv(failed,index=False,encoding="utf-8-sig")
        accepted=sum(bool(row["automatic_acceptance"]) for row in rows);passed=not failures and len(rows)==len(results) and peak<=float(cfg["runtime"]["maximum_peak_vram_gib"]);summary={"status":"PASS" if passed else "FAIL","agent_state":"SAFE_ABSTENTION_NO_GENERATION_TO_ACCEPT" if not len(results) else "AUTOMATIC_ACCEPTANCE_COMPLETE","generated":datetime.now().astimezone().isoformat(),"runtime_seconds":time.perf_counter()-started,"generated_variants":len(results),"accepted_variants":accepted,"rejected_variants":len(rows)-accepted,"failure_count":len(failures),"full_gpu_required":True,"peak_vram_gib":peak,"thermal_claim_policy":"no_Tmrt_or_UTCI_number_until_frozen_multidate_features_are_recomputed"}
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");report=args.report.resolve() if args.report else resolve(cfg["outputs"]["report"]);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(f"""# stage_47 Automatic generation acceptance

Status: **{summary['status']}**  
Agent state: **{summary['agent_state']}**

- Generated variants: {summary['generated_variants']}
- Automatically accepted: {summary['accepted_variants']}
- Automatically rejected: {summary['rejected_variants']}
- Failures: {summary['failure_count']}
- Thermal claim policy: `{summary['thermal_claim_policy']}`

Acceptance combines exact edit-boundary preservation, north-seam preservation and full-GPU semantic
checks for vegetation gain, SVF reduction, road preservation and building preservation. Rejected
variants never reach thermal claims. Accepted variants must rerun the frozen multidate DINOv2 and
semantic feature chain before Tmrt or UTCI improvement is reported.
""",encoding="utf-8");log.info("stage_47 %s: %s",summary["status"],summary);return 0 if passed else 2
    except Exception as error:
        log.exception("stage_47 failed")
        with failed.open("a",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=["point_id","filename","error_message"])
            if handle.tell()==0:writer.writeheader()
            writer.writerow({"point_id":"","filename":"stage_47","error_message":f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__=="__main__":raise SystemExit(main())
