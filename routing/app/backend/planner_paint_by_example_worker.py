"""Generate crown-only alternatives with Paint-by-Example on full CUDA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import PaintByExamplePipeline
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from routing.app.backend.planner_realistic_tree_worker import (
    foliage_conditioned_source,
    most_obvious_crown,
    prepare_edit_masks,
)


MODEL = ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/paint-by-example"
SEEDS = (20260903, 20260911)


def inward_feather(mask: torch.Tensor, radius: int = 13) -> torch.Tensor:
    sigma = max(1.0, radius / 2.5)
    x = torch.arange(-radius, radius + 1, device=mask.device, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma)); kernel /= kernel.sum()
    alpha = F.conv2d(mask[None, None].float(), kernel.view(1, 1, 1, -1), padding=(0, radius))
    alpha = F.conv2d(alpha, kernel.view(1, 1, -1, 1), padding=(radius, 0))[0, 0]
    return alpha.clamp(0, 1) * mask.float()


def crown_example(
    source: Image.Image, source_mask: Image.Image | None,
    reference: Path | None, reference_mask: Image.Image | None,
) -> tuple[Image.Image, str, int]:
    rgb, foliage, area = most_obvious_crown(source, source_mask)
    method = "same_image_semantic_most_obvious_crown"
    if area < 240 and reference is not None and reference.is_file():
        rgb, foliage, area = most_obvious_crown(Image.open(reference).convert("RGB"), reference_mask)
        method = "nearest_coordinate_semantic_crown_fallback"
    if area < 160:
        raise RuntimeError("No reliable semantic crown exemplar is available")
    ys, xs = np.where(foliage)
    margin = max(12, int(max(xs.max() - xs.min(), ys.max() - ys.min()) * .08))
    x0, x1 = max(0, int(xs.min()) - margin), min(512, int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(512, int(ys.max()) + margin + 1)
    crop = Image.fromarray(rgb[y0:y1, x0:x1], "RGB")
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), tuple(np.median(rgb[foliage], axis=0).astype(np.uint8)))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas.resize((512, 512), Image.Resampling.LANCZOS), method, area


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
        source_original = Image.open(args.image).convert("RGB")
        original_size = source_original.size
        source = source_original.resize((512, 512), Image.Resampling.LANCZOS)
        label = np.asarray(Image.open(args.mask).convert("L").resize((512, 512), Image.Resampling.NEAREST), dtype=np.uint8)
        proposal_np = label > 127
        semantic_labels = Image.open(args.semantic_labels).convert("L") if args.semantic_labels and args.semantic_labels.is_file() else None
        protected = Image.open(args.protected_mask).convert("L") if args.protected_mask and args.protected_mask.is_file() else None
        core_np, hard_np = prepare_edit_masks(proposal_np, semantic_labels, protected)
        if float(core_np.mean()) < .0005:
            raise RuntimeError("No reliable crown proposal region is available")
        source_semantic = Image.open(args.vegetation_mask).convert("L") if args.vegetation_mask and args.vegetation_mask.is_file() else None
        reference_semantic = Image.open(args.reference_vegetation_mask).convert("L") if args.reference_vegetation_mask and args.reference_vegetation_mask.is_file() else None
        example, material_method, exemplar_area = crown_example(
            source, source_semantic, args.reference, reference_semantic,
        )
        conditioned_source, layout_method, _ = foliage_conditioned_source(
            source, args.reference, hard_np, source_semantic, reference_semantic,
        )
        args.output_directory.mkdir(parents=True, exist_ok=True)
        example.save(args.output_directory / "crown_exemplar.png", compress_level=3)
        device = torch.device("cuda:0")
        pipe = PaintByExamplePipeline.from_pretrained(
            MODEL, torch_dtype=torch.float16, local_files_only=True,
        ).to(device)
        for name in ("unet", "vae", "image_encoder"):
            module = getattr(pipe, name)
            devices = {str(parameter.device) for parameter in module.parameters()}
            hooks = sum(getattr(child, "_hf_hook", None) is not None for child in module.modules())
            if devices != {"cuda:0"} or hooks:
                raise RuntimeError(f"{name} is not full CUDA: devices={devices}, hooks={hooks}")
        torch.cuda.reset_peak_memory_stats()
        hard = torch.from_numpy(hard_np.copy()).to(device)
        core = torch.from_numpy(core_np.copy()).to(device)
        alpha = inward_feather(hard)
        base = torch.from_numpy(np.asarray(source, dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
        base_pixels = base[:, core]
        base_foliage = ((base_pixels[1] > base_pixels[0] * 1.03) & (base_pixels[1] > base_pixels[2] * .96) & (base_pixels[1] > 38)).float().mean().item()
        mask_image = Image.fromarray((hard_np * 255).astype(np.uint8), "L")
        variants = []
        for seed in (SEEDS[:1] if args.single_seed_preflight else SEEDS):
            tic = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                generated_image = pipe(
                    # The untouched street view and the crown exemplar are separate
                    # conditions. No crown pixels are tiled into the source image.
                    image=source, mask_image=mask_image, example_image=example,
                    num_inference_steps=40, guidance_scale=5.0, generator=generator,
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
            path = args.output_directory / f"paint_by_example_seed_{seed}.png"
            output.save(path, compress_level=3)
            variants.append({
                "seed": seed, "path": str(path.resolve()),
                "generation_method": "paint_by_example_original_scene_exemplar_guided_inpaint",
                "outside_mask_max_difference": outside_max,
                "edit_context_ratio": round(float(hard_np.mean()), 4),
                "changed_fraction_inside_mask": round(changed, 4),
                "foliage_fraction_inside_mask_before": round(float(base_foliage), 4),
                "foliage_fraction_inside_mask_after": round(float(final_foliage), 4),
                "foliage_gain_inside_mask": round(float(final_foliage - base_foliage), 4),
                "inference_seconds": round(time.perf_counter() - tic, 3),
            })
        print(json.dumps({
            "status": "PASS", "model": "Paint-by-Example semantic crown reference editor",
            "generation_mode": "original_scene_exemplar_conditioned_diffusion_crown_only",
            "full_gpu_verified": True, "cpu_fallback": False,
            "conditioning_method": f"{material_method}+{layout_method}", "exemplar_area_pixels": exemplar_area,
            "reference_image": str(args.reference.resolve()) if args.reference and args.reference.is_file() else None,
            "mask_ratio": round(float(core_np.mean()), 4), "edit_context_ratio": round(float(hard_np.mean()), 4), "variants": variants,
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1
    finally:
        pipe = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
