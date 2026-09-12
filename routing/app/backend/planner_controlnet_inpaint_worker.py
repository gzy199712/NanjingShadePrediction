"""Generate structure-preserving crown edits with ControlNet inpainting on full CUDA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from routing.app.backend.planner_realistic_tree_worker import inward_feather, prepare_edit_masks


BASE = ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/stable-diffusion-v1-5-inpainting"
CONTROL = ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/control_v11p_sd15_inpaint"
SEEDS = (20261007, 20261019)


def inpaint_condition(image: Image.Image, mask: Image.Image) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    selected = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    rgb[selected] = -1.0
    return torch.from_numpy(rgb[None].transpose(0, 3, 1, 2))


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
        control_image = inpaint_condition(source, mask_image)
        controlnet = ControlNetModel.from_pretrained(
            CONTROL, torch_dtype=torch.float16, variant="fp16", local_files_only=True,
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            BASE, controlnet=controlnet, torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True, local_files_only=True,
            safety_checker=None,
        ).to("cuda:0")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        for name in ("unet", "vae", "text_encoder", "controlnet"):
            module = getattr(pipe, name)
            devices = {str(parameter.device) for parameter in module.parameters()}
            hooks = sum(getattr(child, "_hf_hook", None) is not None for child in module.modules())
            if devices != {"cuda:0"} or hooks:
                raise RuntimeError(f"{name} is not full CUDA: devices={devices}, hooks={hooks}")
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda:0")
        hard = torch.from_numpy(hard_np.copy()).to(device)
        core = torch.from_numpy(core_np.copy()).to(device)
        alpha = inward_feather(hard, radius=11)
        base = torch.from_numpy(np.asarray(source, dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
        base_pixels = base[:, core]
        base_foliage = ((base_pixels[1] > base_pixels[0] * 1.03) & (base_pixels[1] > base_pixels[2] * .96) & (base_pixels[1] > 38)).float().mean().item()
        prompt = (
            "photorealistic mature broadleaf street tree canopy, connected irregular leafy crowns, "
            "crowns enter naturally from the left or right roadside edge, coherent scale and street perspective, "
            "branches visually anchored behind the sidewalk boundary, natural sunlight and soft foliage boundary"
        )
        negative = (
            "floating crown, isolated spherical topiary, pasted texture, repeated tile, building, roof, wall, "
            "window, billboard, shrub, hedge, tree trunk in road, road obstruction, warped pole, text, blur"
        )
        args.output_directory.mkdir(parents=True, exist_ok=True)
        variants = []
        for seed in (SEEDS[:1] if args.single_seed_preflight else SEEDS):
            tic = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                generated_image = pipe(
                    prompt=prompt, negative_prompt=negative,
                    image=source, mask_image=mask_image,
                    control_image=control_image.to(device=device, dtype=torch.float16),
                    strength=.82, num_inference_steps=36, guidance_scale=9.0,
                    controlnet_conditioning_scale=.88,
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
            path = args.output_directory / f"controlnet_inpaint_seed_{seed}.png"
            output.save(path, compress_level=3)
            variants.append({
                "seed": seed, "path": str(path.resolve()),
                "generation_method": "controlnet_original_scene_structure_preserving_inpaint",
                "outside_mask_max_difference": outside_max,
                "edit_context_ratio": round(float(hard_np.mean()), 4),
                "changed_fraction_inside_mask": round(changed, 4),
                "foliage_fraction_inside_mask_before": round(float(base_foliage), 4),
                "foliage_fraction_inside_mask_after": round(float(final_foliage), 4),
                "foliage_gain_inside_mask": round(float(final_foliage - base_foliage), 4),
                "inference_seconds": round(time.perf_counter() - tic, 3),
            })
        print(json.dumps({
            "status": "PASS", "model": "ControlNet v1.1 structure-preserving inpaint",
            "generation_mode": "original_scene_control_conditioned_diffusion_crown_only",
            "conditioning_method": "masked_original_scene_control_plus_roadside_anchor_prompt",
            "full_gpu_verified": True, "cpu_fallback": False,
            "mask_ratio": round(float(core_np.mean()), 4), "edit_context_ratio": round(float(hard_np.mean()), 4),
            "variants": variants,
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1
    finally:
        pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
