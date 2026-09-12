"""stage_46b: generate and reproject tree-first edits only for fully unlocked layouts."""

from __future__ import annotations

import argparse, csv, gc, json, logging, sys, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from safetensors.torch import load_model
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/panorama_tree_generation_agent.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_46b"


def resolve(value): return ROOT / value


def get_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True); log=logging.getLogger("stage_46b");log.handlers.clear();log.setLevel(logging.INFO);fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR/"stage_46b.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)):handler.setFormatter(fmt);log.addHandler(handler)
    return log


def load_jobs(cfg: dict, layout_root: Path, structural_path: Path, benchmark_path: Path) -> list[dict]:
    layout_qc=pd.read_csv(layout_root/"layout_qc.csv",dtype={"point_id":str}); structural=pd.read_csv(structural_path,dtype={"point_id":str}); benchmark=pd.read_csv(benchmark_path,dtype={"point_id":str})
    jobs=layout_qc.loc[layout_qc.status.eq("LAYOUT_READY")].merge(structural.loc[structural.planting_support_pass.astype(bool)],on="point_id",validate="one_to_one").merge(benchmark[["point_id","output_path"]],on="point_id",validate="one_to_one")
    return jobs.to_dict("records")


def gaussian_alpha(mask: torch.Tensor, radius: int) -> torch.Tensor:
    sigma=max(1.0,radius/2.5); x=torch.arange(-radius,radius+1,device=mask.device,dtype=torch.float32); kernel=torch.exp(-(x*x)/(2*sigma*sigma));kernel/=kernel.sum();a=F.conv2d(mask.float()[None,None],kernel.view(1,1,1,-1),padding=(0,radius));a=F.conv2d(a,kernel.view(1,1,-1,1),padding=(radius,0));return a[0,0].clamp(0,1)


def layout_mask(layout: dict, device: torch.device) -> torch.Tensor:
    y,x=torch.meshgrid(torch.arange(512,device=device,dtype=torch.float32),torch.arange(512,device=device,dtype=torch.float32),indexing="ij");mask=torch.zeros((512,512),device=device,dtype=torch.bool)
    for anchor in layout["tree_anchors"]:
        cx=float(anchor["crown_center_x"]);cy=float(anchor["crown_center_y"]);radius=float(anchor["crown_diameter_px"])/2; crown=((x-cx)/(radius*1.08))**2+((y-cy)/(radius*.86))**2<=1
        half=max(2.0,float(anchor["trunk_width_px"])/2); trunk=(x>=cx-half)&(x<=cx+half)&(y>=cy)&(y<=float(anchor["base_y"]));mask|=crown|trunk
    return mask


def reproject_delta(base_pano: torch.Tensor, base_window: torch.Tensor, generated: torch.Tensor, alpha: torch.Tensor, mapping_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    mapping=np.load(mapping_path);xy=torch.from_numpy(mapping["panorama_xy"].astype(np.float32)).to(base_pano.device);valid=torch.from_numpy(mapping["valid"]).to(base_pano.device)&(alpha>1e-5);px=xy[0][valid];py=xy[1][valid];a=alpha[valid];diff=(generated-base_window)[:,valid]
    height,width=base_pano.shape[1:];numer=torch.zeros((3,height*width),device=base_pano.device);denom=torch.zeros(height*width,device=base_pano.device)
    x0=torch.floor(px).long();y0=torch.floor(py).long();fx=px-x0.float();fy=py-y0.float()
    for dx,dy,w in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        xi=(x0+dx)%width;yi=(y0+dy).clamp(0,height-1);index=yi*width+xi;weight=w*a;denom.scatter_add_(0,index,weight)
        for channel in range(3):numer[channel].scatter_add_(0,index,diff[channel]*weight)
    active=denom>1e-6;flat=base_pano.reshape(3,-1).clone();flat[:,active]+=numer[:,active]/denom[active];return flat.reshape_as(base_pano).clamp(0,255),denom.reshape(height,width)


def load_pipe(cfg: dict):
    code=resolve(cfg["models"]["code"]);sys.path.insert(0,str(code));from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline;from powerpaint.utils.utils import TokenizerWrapper,add_tokens
    base=resolve(cfg["models"]["base"]);weights=resolve(cfg["models"]["powerpaint"]);pipe=StableDiffusionInpaintPipeline.from_pretrained(base,torch_dtype=torch.float16,variant="fp16",use_safetensors=True,local_files_only=True);pipe.tokenizer=TokenizerWrapper(from_pretrained=base,subfolder="tokenizer",local_files_only=True);add_tokens(tokenizer=pipe.tokenizer,text_encoder=pipe.text_encoder,placeholder_tokens=["P_ctxt","P_shape","P_obj"],initialize_tokens=["a","a","a"],num_vectors_per_token=10)
    mismatch=(load_model(pipe.unet,weights/"unet/unet.safetensors"),load_model(pipe.text_encoder,weights/"text_encoder/text_encoder.safetensors"))
    if any(part for pair in mismatch for part in pair):raise RuntimeError(f"PowerPaint weight mismatch: {mismatch}")
    pipe.to("cuda:0");
    for name,module in {"unet":pipe.unet,"vae":pipe.vae,"text_encoder":pipe.text_encoder,"safety_checker":pipe.safety_checker}.items():
        devices={str(p.device) for p in module.parameters()};hooks=sum(getattr(child,"_hf_hook",None) is not None for child in module.modules())
        if devices!={"cuda:0"} or hooks:raise RuntimeError(f"{name} is not full CUDA: {devices}, hooks={hooks}")
    return pipe


def save_tensor(image: torch.Tensor,path: Path):Image.fromarray(image.round().clamp(0,255).to(torch.uint8).permute(1,2,0).cpu().numpy(),"RGB").save(path,compress_level=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--mode",choices=("check","run"),default="check");parser.add_argument("--overwrite",action="store_true");parser.add_argument("--layout-directory",type=Path);parser.add_argument("--structural-qc",type=Path);parser.add_argument("--benchmark",type=Path);parser.add_argument("--windows-directory",type=Path);parser.add_argument("--output-directory",type=Path);parser.add_argument("--report",type=Path);args=parser.parse_args();log=get_logger();failed=LOG_DIR/"failed_files.csv";pipe=None
    try:
        cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));preflight=json.loads(resolve(cfg["inputs"]["editor_preflight"]).read_text(encoding="utf-8"));layout_root=args.layout_directory.resolve() if args.layout_directory else resolve(cfg["inputs"]["layout_directory"]);structural=args.structural_qc.resolve() if args.structural_qc else resolve(cfg["inputs"]["structural_qc"]);benchmark=args.benchmark.resolve() if args.benchmark else resolve(cfg["inputs"]["benchmark"]);windows=args.windows_directory.resolve() if args.windows_directory else resolve(cfg["inputs"]["windows_directory"]);jobs=load_jobs(cfg,layout_root,structural,benchmark)
        check={"status":"PASS" if torch.cuda.is_available() and preflight.get("status")=="PASS" else "FAIL","eligible_generation_points":len(jobs),"safe_abstention":len(jobs)==0,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.mode=="check":print(json.dumps(check,ensure_ascii=False,indent=2));return 0 if check["status"]=="PASS" else 2
        if check["status"]!="PASS":raise RuntimeError(check)
        out=args.output_directory.resolve() if args.output_directory else resolve(cfg["outputs"]["directory"])
        if out.exists() and any(out.iterdir()) and not args.overwrite:raise FileExistsError(f"Output exists; use --overwrite: {out}")
        out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();records=[];failures=[];peak=0.0
        if jobs:
            torch.cuda.set_device("cuda:0");pipe=load_pipe(cfg);torch.cuda.reset_peak_memory_stats();device=torch.device("cuda:0")
            for job in tqdm(jobs,desc="Generating panorama tree scenarios",unit="point",dynamic_ncols=True):
                pid=str(job["point_id"])
                try:
                    layout=json.loads((layout_root/"points"/pid/"layout.json").read_text(encoding="utf-8"));heading=int(layout["selected_heading"]);window_path=windows/"points"/pid/"windows"/f"heading_{heading:03d}.png";mapping_path=windows/"points"/pid/"correspondence"/f"heading_{heading:03d}.npz";point_dir=out/"points"/pid;point_dir.mkdir(parents=True,exist_ok=True)
                    base_window=torch.from_numpy(np.asarray(Image.open(window_path).convert("RGB"),dtype=np.uint8).copy()).permute(2,0,1).to(device,dtype=torch.float32);base_pano=torch.from_numpy(np.asarray(Image.open(job["output_path"]).convert("RGB"),dtype=np.uint8).copy()).permute(2,0,1).to(device,dtype=torch.float32);hard=layout_mask(layout,device);alpha=gaussian_alpha(hard,int(cfg["generation"]["mask_feather_pixels"]));mask_image=Image.fromarray((hard*255).to(torch.uint8).cpu().numpy(),"L");mask_image.save(point_dir/"edit_mask.png")
                    for seed in cfg["generation"]["seeds"]:
                        generator=torch.Generator(device="cuda:0").manual_seed(int(seed));tic=time.perf_counter();negative=cfg["generation"]["negative_prompt"]
                        with torch.inference_mode():result=pipe(promptA=cfg["generation"]["prompt"],promptB=cfg["generation"]["context_prompt"],negative_promptA=negative+" P_shape",negative_promptB=negative+" P_ctxt",image=Image.open(window_path).convert("RGB"),mask=mask_image,width=512,height=512,strength=1.0,tradoff=float(cfg["generation"]["fitting_degree"]),tradoff_nag=float(cfg["generation"]["fitting_degree"]),guidance_scale=float(cfg["generation"]["guidance_scale"]),num_inference_steps=int(cfg["generation"]["inference_steps"]),generator=generator)
                        torch.cuda.synchronize();generated=torch.from_numpy(np.asarray(result.images[0].convert("RGB"),dtype=np.uint8).copy()).permute(2,0,1).to(device,dtype=torch.float32);panorama,coverage=reproject_delta(base_pano,base_window,generated,alpha,mapping_path);window_out=point_dir/f"seed_{seed}_window.png";pano_out=point_dir/f"seed_{seed}_panorama.png";save_tensor(generated,window_out);save_tensor(panorama,pano_out);outside=coverage==0;outside_max=int((panorama[:,outside]-base_pano[:,outside]).abs().max().item());north_unchanged=bool(torch.equal(panorama[:,:,0].round().byte(),base_pano[:,:,0].round().byte()) and torch.equal(panorama[:,:,-1].round().byte(),base_pano[:,:,-1].round().byte()))
                        records.append({"point_id":pid,"heading":heading,"seed":seed,"baseline_window_path":str(window_path.resolve()),"baseline_panorama_path":str(Path(job["output_path"]).resolve()),"window_path":str(window_out.resolve()),"panorama_path":str(pano_out.resolve()),"mask_path":str((point_dir/'edit_mask.png').resolve()),"outside_mask_max_difference":outside_max,"north_seam_unchanged":north_unchanged,"edited_panorama_ratio":float((coverage>0).float().mean()),"inference_seconds":time.perf_counter()-tic,"full_gpu_verified":True,"technical_pass":outside_max==0 and north_unchanged})
                except Exception as error:log.exception("Point %s failed",pid);failures.append({"point_id":pid,"filename":"panorama_generation","error_message":f"{type(error).__name__}: {error}"})
            peak=torch.cuda.max_memory_reserved()/1024**3
        columns=["point_id","heading","seed","baseline_window_path","baseline_panorama_path","window_path","panorama_path","mask_path","outside_mask_max_difference","north_seam_unchanged","edited_panorama_ratio","inference_seconds","full_gpu_verified","technical_pass"]
        pd.DataFrame(records,columns=columns).to_csv(out/"generation_results.csv",index=False,encoding="utf-8-sig");pd.DataFrame(failures,columns=["point_id","filename","error_message"]).to_csv(failed,index=False,encoding="utf-8-sig")
        expected=len(jobs)*len(cfg["generation"]["seeds"]);passed=not failures and len(records)==expected and all(row["technical_pass"] for row in records) and peak<=float(cfg["runtime"]["maximum_peak_vram_gib"])
        state="GENERATED_AWAITING_AUTOMATIC_ACCEPTANCE" if records else "SAFE_ABSTENTION_NO_ELIGIBLE_SCENE";summary={"status":"PASS" if passed else "FAIL","agent_state":state,"operational_complete":True,"generated":datetime.now().astimezone().isoformat(),"runtime_seconds":time.perf_counter()-started,"eligible_generation_points":len(jobs),"requested_variants":expected,"generated_variants":len(records),"failure_count":len(failures),"full_gpu_required":True,"peak_vram_gib":peak,"generation_bypass_forbidden":True}
        (out/"agent_state.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");report=args.report.resolve() if args.report else resolve(cfg["outputs"]["report"]);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(f"""# stage_46b Panorama tree generation agent

Status: **{summary['status']}**  
Agent state: **{summary['agent_state']}**

- Eligible points: {summary['eligible_generation_points']}
- Generated variants: {summary['generated_variants']}
- Failures: {summary['failure_count']}
- Full-GPU policy: {summary['full_gpu_required']}
- Generation bypass forbidden: {summary['generation_bypass_forbidden']}

The agent is operational even when no scene is eligible: it returns a traceable safe abstention and
does not load the image editor. Once a point passes ground-street, tree-continuity, semantic, depth
and 2.5D layout gates, the same code generates two perspective-window variants and reprojects their
GPU-blended deltas to the north-aligned panorama while preserving the north seam.
""",encoding="utf-8");log.info("stage_46b %s: %s",summary["status"],summary);return 0 if passed else 2
    except Exception as error:
        log.exception("stage_46b failed")
        with failed.open("a",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=["point_id","filename","error_message"])
            if handle.tell()==0:writer.writeheader()
            writer.writerow({"point_id":"","filename":"stage_46b","error_message":f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1
    finally:
        pipe=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=="__main__":raise SystemExit(main())
