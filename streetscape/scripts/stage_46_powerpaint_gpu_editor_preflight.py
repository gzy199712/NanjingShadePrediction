"""stage_46: verify the frozen PowerPaint editor with a full-CUDA synthetic window."""

from __future__ import annotations

import argparse, csv, gc, json, logging, sys, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFilter
from safetensors.torch import load_model
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/powerpaint_gpu_editor_preflight.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_46"


def resolve(value): return ROOT / value


def logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True); log = logging.getLogger("stage_46"); log.handlers.clear(); log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_46.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)): handler.setFormatter(fmt); log.addHandler(handler)
    return log


def active_hooks(module): return sum(getattr(child, "_hf_hook", None) is not None for child in module.modules())


def synthetic_fixture(size: int) -> tuple[Image.Image, Image.Image, Image.Image]:
    image = Image.new("RGB", (size, size), (164, 205, 229)); draw = ImageDraw.Draw(image)
    draw.rectangle((0, 260, size, size), fill=(119, 119, 116)); draw.polygon([(0, 360), (size, 310), (size, 405), (0, 485)], fill=(188, 184, 172))
    draw.rectangle((0, 210, size, 275), fill=(193, 177, 147)); draw.rectangle((40, 115, 190, 260), fill=(179, 167, 143)); draw.rectangle((330, 105, 490, 260), fill=(185, 174, 151))
    mask = Image.new("L", (size, size), 0); m = ImageDraw.Draw(mask)
    for x, scale in [(95, 1.0), (210, .85), (310, .70), (390, .58)]:
        diameter = int(135 * scale); cy = int(215 + (1-scale)*80); m.ellipse((x-diameter//2, cy-diameter//2, x+diameter//2, cy+diameter//2), fill=255); m.rectangle((x-max(2,int(5*scale)), cy, x+max(2,int(5*scale)), int(390-(1-scale)*70)), fill=255)
    return image, mask, mask.filter(ImageFilter.GaussianBlur(10))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", choices=("check", "run"), default="check"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args(); log = logger(); failed = LOG_DIR / "failed_files.csv"
    pipe = None
    try:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); paths = {name: resolve(cfg["models"][name]) for name in ("base", "powerpaint", "code")}
        check = {"status": "PASS" if torch.cuda.is_available() and all(path.is_dir() for path in paths.values()) else "FAIL", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "paths": {k: str(v) for k,v in paths.items()}}
        if args.mode == "check": print(json.dumps(check, ensure_ascii=False, indent=2)); return 0 if check["status"] == "PASS" else 2
        if check["status"] != "PASS": raise RuntimeError(check)
        out = resolve(cfg["outputs"]["directory"])
        if out.exists() and any(out.iterdir()) and not args.overwrite: raise FileExistsError(f"Output exists; use --overwrite: {out}")
        out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(paths["code"]))
        from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline
        from powerpaint.utils.utils import TokenizerWrapper, add_tokens
        torch.cuda.set_device("cuda:0"); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        pipe = StableDiffusionInpaintPipeline.from_pretrained(paths["base"], torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True)
        pipe.tokenizer = TokenizerWrapper(from_pretrained=paths["base"], subfolder="tokenizer", local_files_only=True)
        add_tokens(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, placeholder_tokens=["P_ctxt", "P_shape", "P_obj"], initialize_tokens=["a", "a", "a"], num_vectors_per_token=10)
        um, uu = load_model(pipe.unet, paths["powerpaint"] / "unet/unet.safetensors"); tm, tu = load_model(pipe.text_encoder, paths["powerpaint"] / "text_encoder/text_encoder.safetensors")
        if um or uu or tm or tu: raise RuntimeError(f"PowerPaint weight mismatch: unet={(um,uu)} text={(tm,tu)}")
        pipe.to("cuda:0"); audits = []
        for name, module in tqdm({"unet":pipe.unet,"vae":pipe.vae,"text_encoder":pipe.text_encoder,"safety_checker":pipe.safety_checker}.items(), desc="Auditing editor modules", unit="module", dynamic_ncols=True):
            devices = sorted({str(parameter.device) for parameter in module.parameters()}); hooks = active_hooks(module); audits.append({"module":name,"parameter_devices":"|".join(devices),"active_offload_hooks":hooks})
            if devices != ["cuda:0"] or hooks: raise RuntimeError(f"{name} placement failed: devices={devices}, hooks={hooks}")
        size = int(cfg["generation"]["width"]); baseline, mask, feather = synthetic_fixture(size); baseline.save(out/"fixture_baseline.png"); mask.save(out/"fixture_mask.png")
        negative = cfg["generation"]["negative_prompt"] + " P_shape"; generator = torch.Generator(device="cuda:0").manual_seed(int(cfg["generation"]["seed"])); tic = time.perf_counter()
        with torch.inference_mode():
            result = pipe(promptA=cfg["generation"]["prompt"], promptB=cfg["generation"]["context_prompt"], negative_promptA=negative, negative_promptB=negative.replace("P_shape","P_ctxt"), image=baseline, mask=mask, width=size, height=size, strength=1.0, tradoff=float(cfg["generation"]["fitting_degree"]), tradoff_nag=float(cfg["generation"]["fitting_degree"]), guidance_scale=float(cfg["generation"]["guidance_scale"]), num_inference_steps=int(cfg["generation"]["inference_steps"]), generator=generator)
        torch.cuda.synchronize(); generated = result.images[0].convert("RGB"); generated.save(out/"fixture_generated.png")
        base = np.asarray(baseline,dtype=np.float32); gen = np.asarray(generated,dtype=np.float32); alpha=np.asarray(feather,dtype=np.float32)[...,None]/255.0; composite=np.clip(base*(1-alpha)+gen*alpha,0,255).astype(np.uint8); Image.fromarray(composite).save(out/"fixture_composite.png")
        # Feathering intentionally extends beyond the hard mask. Preservation is
        # measured only where the actual composite alpha is exactly zero.
        outside = alpha[..., 0] == 0
        outside_max = int(np.abs(composite.astype(np.int16)-base.astype(np.int16))[outside].max()); peak = torch.cuda.max_memory_reserved()/1024**3
        summary={"status":"PASS" if outside_max==0 and peak<=float(cfg["runtime"]["maximum_peak_vram_gib"]) else "FAIL","generated":datetime.now().astimezone().isoformat(),"runtime_seconds":time.perf_counter()-started,"inference_seconds":time.perf_counter()-tic,"full_gpu_verified":True,"module_audit":audits,"active_cpu_offload_hooks":sum(row["active_offload_hooks"] for row in audits),"peak_vram_gib":peak,"outside_mask_max_difference":outside_max,"model_revision":cfg["models"]["model_revision"],"code_revision":cfg["models"]["code_revision"]}
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); resolve(cfg["outputs"]["report"]).write_text(f"""# stage_46 PowerPaint full-GPU editor preflight

Status: **{summary['status']}**  
Generated: {summary['generated']}

- All editor modules on CUDA: {summary['full_gpu_verified']}
- CPU offload hooks: {summary['active_cpu_offload_hooks']}
- Peak reserved VRAM: {summary['peak_vram_gib']:.3f} GiB
- Inference time: {summary['inference_seconds']:.2f} s
- Outside-mask maximum change: {summary['outside_mask_max_difference']}
- Model revision: `{summary['model_revision']}`
- Code revision: `{summary['code_revision']}`

This verifies editor capability only. A real panorama can be edited only after the structural,
tree-continuity, semantic and 2.5D geometry gates all pass.
""",encoding="utf-8"); pd_path=out/"module_device_audit.csv"
        with pd_path.open("w",newline="",encoding="utf-8-sig") as handle: writer=csv.DictWriter(handle,fieldnames=audits[0].keys());writer.writeheader();writer.writerows(audits)
        log.info("stage_46 %s: %s",summary["status"],summary); return 0 if summary["status"]=="PASS" else 2
    except Exception as error:
        log.exception("stage_46 failed")
        with failed.open("a",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=["point_id","filename","error_message"])
            if handle.tell()==0:writer.writeheader()
            writer.writerow({"point_id":"","filename":"stage_46","error_message":f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1
    finally:
        pipe=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=="__main__": raise SystemExit(main())
