"""Generate tree-crown edits with official PowerPaint v2-1 BrushNet on full CUDA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler
from PIL import Image
from safetensors.torch import load_model
from transformers import CLIPTextModel


ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "artifacts/retired/failed_experiment_dependencies/third_party/powerpaint"
CHECKPOINT = ROOT / "models/planner/powerpaint-v2-1"
BASE = CHECKPOINT / "realisticVisionV60B1_v51VAE"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CODE))
from routing.app.backend.planner_realistic_tree_worker import inward_feather, prepare_edit_masks
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.utils.utils import TokenizerWrapper, add_tokens


SEEDS = (20261201, 20261213)


def load_pipeline() -> StableDiffusionPowerPaintBrushNetPipeline:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    dtype = torch.float16
    init_unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=dtype, local_files_only=True)
    brush_text = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True)
    brushnet = BrushNetModel.from_unet(init_unet)
    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        BASE, brushnet=brushnet, text_encoder_brushnet=brush_text,
        torch_dtype=dtype, low_cpu_mem_usage=False, safety_checker=None,
        local_files_only=True,
    )
    pipe.unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=dtype, local_files_only=True)
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained=BASE, subfolder="tokenizer", torch_type=dtype, local_files_only=True,
    )
    add_tokens(
        tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder_brushnet,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"], num_vectors_per_token=10,
    )
    mismatch = load_model(pipe.brushnet, CHECKPOINT / "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors")
    if any(mismatch):
        raise RuntimeError(f"BrushNet weight mismatch: {mismatch}")
    text_state = torch.load(CHECKPOINT / "PowerPaint_Brushnet/pytorch_model.bin", map_location="cpu", weights_only=True)
    pipe.text_encoder_brushnet.load_state_dict(text_state, strict=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda:0")
    modules = (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_brushnet, pipe.brushnet)
    devices = {str(parameter.device) for module in modules for parameter in module.parameters()}
    hooks = sum(getattr(child, "_hf_hook", None) is not None for module in modules for child in module.modules())
    if devices != {"cuda:0"} or hooks:
        raise RuntimeError(f"PowerPaint v2-1 is not full CUDA: devices={devices}, hooks={hooks}")
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--vegetation-mask", type=Path)
    parser.add_argument("--semantic-labels", type=Path)
    parser.add_argument("--protected-mask", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-vegetation-mask", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--single-seed-preflight", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter(); pipe = None
    try:
        original = Image.open(args.image).convert("RGB")
        original_size = original.size
        source = original.resize((512, 512), Image.Resampling.LANCZOS)
        proposal = np.asarray(Image.open(args.mask).convert("L").resize((512, 512), Image.Resampling.NEAREST), dtype=np.uint8) > 127
        semantics = Image.open(args.semantic_labels).convert("L") if args.semantic_labels and args.semantic_labels.is_file() else None
        protected = Image.open(args.protected_mask).convert("L") if args.protected_mask and args.protected_mask.is_file() else None
        core_np, hard_np = prepare_edit_masks(proposal, semantics, protected)
        if float(core_np.mean()) < .0005:
            raise RuntimeError("No reliable crown proposal region is available")
        mask_image = Image.fromarray((hard_np * 255).astype(np.uint8), "L")
        masked_source_np = np.asarray(source, dtype=np.uint8).copy()
        masked_source_np[hard_np] = 0
        masked_source = Image.fromarray(masked_source_np, "RGB")
        pipe = load_pipeline(); torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda:0")
        hard = torch.from_numpy(hard_np.copy()).to(device)
        core = torch.from_numpy(core_np.copy()).to(device)
        alpha = inward_feather(hard, radius=11)
        base = torch.from_numpy(np.asarray(source, dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
        base_pixels = base[:, core]
        base_foliage = ((base_pixels[1] > base_pixels[0] * 1.03) & (base_pixels[1] > base_pixels[2] * .96) & (base_pixels[1] > 38)).float().mean().item()
        prompt = (
            "photorealistic continuous mature broadleaf street tree crowns, irregular natural canopy entering from existing "
            "roadside vegetation, large high crown scale, coherent street perspective, realistic leaf texture and sunlight"
        )
        negative = (
            "floating crown, isolated spherical topiary, pasted texture, repeated pattern, building, roof, wall, scaffold, "
            "billboard, awning, hedge, shrub, tiny tree, tree trunk in road, road obstruction, distortion, text, blur"
        )
        args.output_directory.mkdir(parents=True, exist_ok=True)
        variants = []
        for seed in (SEEDS[:1] if args.single_seed_preflight else SEEDS):
            tic = time.perf_counter()
            with torch.inference_mode():
                result = pipe(
                    promptA="P_shape", promptB="P_ctxt", promptU=prompt,
                    tradoff=.60, tradoff_nag=.60,
                    image=masked_source, mask=mask_image.convert("RGB"),
                    num_inference_steps=32,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    brushnet_conditioning_scale=1.0,
                    negative_promptA="P_shape", negative_promptB="P_ctxt", negative_promptU=negative,
                    guidance_scale=10.0, width=512, height=512,
                ).images[0].convert("RGB")
            torch.cuda.synchronize()
            generated = torch.from_numpy(np.asarray(result, dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
            composite = base * (1 - alpha.unsqueeze(0)) + generated * alpha.unsqueeze(0)
            outside_max = int((composite[:, ~hard] - base[:, ~hard]).abs().max().item())
            changed = float(((composite - base).abs().mean(0) > 4)[core].float().mean().item())
            final_pixels = composite[:, core]
            final_foliage = ((final_pixels[1] > final_pixels[0] * 1.03) & (final_pixels[1] > final_pixels[2] * .96) & (final_pixels[1] > 38)).float().mean().item()
            output = Image.fromarray(composite.round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy(), "RGB")
            if original_size != (512, 512):
                output = output.resize(original_size, Image.Resampling.LANCZOS)
            path = args.output_directory / f"powerpaint_v2_seed_{seed}.png"; output.save(path, compress_level=3)
            variants.append({
                "seed": seed, "path": str(path.resolve()),
                "generation_method": "powerpaint_v2_1_realisticvision_brushnet_original_scene_inpaint",
                "outside_mask_max_difference": outside_max,
                "edit_context_ratio": round(float(hard_np.mean()), 4),
                "changed_fraction_inside_mask": round(changed, 4),
                "foliage_fraction_inside_mask_before": round(float(base_foliage), 4),
                "foliage_fraction_inside_mask_after": round(float(final_foliage), 4),
                "foliage_gain_inside_mask": round(float(final_foliage - base_foliage), 4),
                "inference_seconds": round(time.perf_counter() - tic, 3),
            })
        print(json.dumps({
            "status": "PASS", "model": "PowerPaint v2-1 RealisticVision BrushNet",
            "generation_mode": "original_scene_shape_guided_brushnet_inpaint",
            "conditioning_method": "semantic_mask_plus_realisticvision_context_and_task_tokens",
            "full_gpu_verified": True, "cpu_fallback": False,
            "mask_ratio": round(float(core_np.mean()), 4), "edit_context_ratio": round(float(hard_np.mean()), 4),
            "variants": variants,
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }, ensure_ascii=False)); return 0
    except Exception as error:
        print(json.dumps({
            "status": "FAIL", "error": f"{type(error).__name__}: {error}",
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3) if torch.cuda.is_available() else None,
        }, ensure_ascii=False)); return 1
    finally:
        pipe = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
