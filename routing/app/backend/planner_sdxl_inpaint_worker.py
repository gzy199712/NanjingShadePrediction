"""Generate mature crown alternatives with official SDXL Inpainting on full CUDA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from routing.app.backend.planner_realistic_tree_worker import inward_feather, prepare_edit_masks


MODEL = ROOT / "models/planner/sdxl-inpainting-1.0"
SEEDS = (20261103, 20261117)


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
    parser.add_argument("--inference-steps", type=int, default=36)
    args = parser.parse_args()
    started = time.perf_counter(); pipe = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        original = Image.open(args.image).convert("RGB")
        original_size = original.size
        source = original.resize((512, 512), Image.Resampling.LANCZOS)
        proposal = np.asarray(
            Image.open(args.mask).convert("L").resize((512, 512), Image.Resampling.NEAREST),
            dtype=np.uint8,
        ) > 127
        semantics = Image.open(args.semantic_labels).convert("L") if args.semantic_labels and args.semantic_labels.is_file() else None
        protected = Image.open(args.protected_mask).convert("L") if args.protected_mask and args.protected_mask.is_file() else None
        core_np, hard_np = prepare_edit_masks(proposal, semantics, protected)
        if float(core_np.mean()) < .0005:
            raise RuntimeError("No reliable crown proposal region is available")
        mask_image = Image.fromarray((hard_np * 255).astype(np.uint8), "L")
        pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL, torch_dtype=torch.float16, variant="fp16", local_files_only=True,
        ).to("cuda:0")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # PyTorch 2 scaled-dot-product attention is already memory efficient at
        # 512 px. Maximal per-head slicing made one step take ~38 seconds and is
        # therefore unsuitable for an interactive planner workflow.
        pipe.enable_vae_slicing()
        modules = (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2)
        devices = {str(parameter.device) for module in modules for parameter in module.parameters()}
        hooks = sum(getattr(child, "_hf_hook", None) is not None for module in modules for child in module.modules())
        if devices != {"cuda:0"} or hooks:
            raise RuntimeError(f"SDXL is not full CUDA: devices={devices}, hooks={hooks}")
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda:0")
        hard = torch.from_numpy(hard_np.copy()).to(device)
        core = torch.from_numpy(core_np.copy()).to(device)
        alpha = inward_feather(hard, radius=12)
        base = torch.from_numpy(np.asarray(source, dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
        base_pixels = base[:, core]
        base_foliage = ((base_pixels[1] > base_pixels[0] * 1.03) & (base_pixels[1] > base_pixels[2] * .96) & (base_pixels[1] > 38)).float().mean().item()
        prompt = (
            "documentary street-view photograph, mature broadleaf street tree canopy only inside the masked sky gap, "
            "a continuous irregular row of high tree crowns extending from existing roadside vegetation, "
            "large natural crown scale, coherent one-point street perspective, realistic branches anchored behind the road edge, "
            "matching local sunlight, atmospheric depth, leaf texture and camera exposure, seamless natural boundary"
        )
        negative = (
            "floating tree crown, isolated ball-shaped topiary, pasted texture, clone stamp, repeating tile, building, roof, "
            "wall, window, scaffold, billboard, awning, hedge, shrub, tiny tree, multiple trunks, tree in road, warped pole, text"
        )
        args.output_directory.mkdir(parents=True, exist_ok=True)
        variants = []
        seeds = SEEDS[:1] if args.single_seed_preflight else SEEDS
        for seed in seeds:
            tic = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                generated_image = pipe(
                    prompt=prompt, negative_prompt=negative,
                    image=source, mask_image=mask_image,
                    width=512, height=512, strength=.92,
                    num_inference_steps=args.inference_steps, guidance_scale=9.0,
                    generator=generator,
                ).images[0].convert("RGB")
            torch.cuda.synchronize()
            generated = torch.from_numpy(np.asarray(generated_image, dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
            composite = base * (1 - alpha.unsqueeze(0)) + generated * alpha.unsqueeze(0)
            outside_max = int((composite[:, ~hard] - base[:, ~hard]).abs().max().item())
            changed = float(((composite - base).abs().mean(0) > 4)[core].float().mean().item())
            final_pixels = composite[:, core]
            final_foliage = ((final_pixels[1] > final_pixels[0] * 1.03) & (final_pixels[1] > final_pixels[2] * .96) & (final_pixels[1] > 38)).float().mean().item()
            output = Image.fromarray(composite.round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy(), "RGB")
            if original_size != (512, 512):
                output = output.resize(original_size, Image.Resampling.LANCZOS)
            path = args.output_directory / f"sdxl_inpaint_seed_{seed}.png"
            output.save(path, compress_level=3)
            variants.append({
                "seed": seed, "path": str(path.resolve()),
                "generation_method": "sdxl_original_scene_semantic_mask_inpaint",
                "outside_mask_max_difference": outside_max,
                "edit_context_ratio": round(float(hard_np.mean()), 4),
                "changed_fraction_inside_mask": round(changed, 4),
                "foliage_fraction_inside_mask_before": round(float(base_foliage), 4),
                "foliage_fraction_inside_mask_after": round(float(final_foliage), 4),
                "foliage_gain_inside_mask": round(float(final_foliage - base_foliage), 4),
                "inference_seconds": round(time.perf_counter() - tic, 3),
            })
        print(json.dumps({
            "status": "PASS", "model": "Official SDXL 1.0 Inpainting",
            "generation_mode": "original_scene_high_fidelity_diffusion_inpaint",
            "conditioning_method": "semantic_mask_plus_original_context_and_roadside_anchor_prompt",
            "full_gpu_verified": True, "cpu_fallback": False,
            "mask_ratio": round(float(core_np.mean()), 4), "edit_context_ratio": round(float(hard_np.mean()), 4),
            "variants": variants,
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({
            "status": "FAIL", "error": f"{type(error).__name__}: {error}",
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3) if torch.cuda.is_available() else None,
        }, ensure_ascii=False))
        return 1
    finally:
        pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
