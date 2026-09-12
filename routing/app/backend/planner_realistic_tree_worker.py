"""Generate two tree-first realistic directional scenarios with full-GPU PowerPaint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image, ImageFilter
from safetensors.torch import load_model


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/stable-diffusion-v1-5-inpainting"
WEIGHTS = ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/powerpaint-v1"
CODE = ROOT / "artifacts/retired/failed_experiment_dependencies/third_party/powerpaint"
SEEDS = (20260817, 20260829)


def prepare_edit_masks(
    proposal: np.ndarray, semantic_labels: Image.Image | None, protected: Image.Image | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a must-fill core plus a wider blending context that preserves street objects."""
    core = proposal.astype(bool)
    immutable = np.zeros_like(core, dtype=bool)
    if semantic_labels is not None:
        labels = np.asarray(semantic_labels.convert("L").resize((512, 512), Image.Resampling.NEAREST), dtype=np.uint8)
        # Cityscapes: keep roads, sidewalks, poles/signs/lights, people, vehicles and cycles immutable.
        immutable = np.isin(labels, np.asarray((0, 1, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18), dtype=np.uint8))
    protected_np = np.zeros_like(core, dtype=bool)
    if protected is not None:
        protected_np = np.asarray(protected.convert("L").resize((512, 512), Image.Resampling.NEAREST), dtype=np.uint8) > 127
    core &= ~immutable & ~protected_np
    expanded = cv2.dilate(core.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=1).astype(bool)
    expanded &= ~immutable & ~protected_np
    return core, expanded


def load_pipeline():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    sys.path.insert(0, str(CODE))
    from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline
    from powerpaint.utils.utils import TokenizerWrapper, add_tokens

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, local_files_only=True,
    )
    pipe.tokenizer = TokenizerWrapper(from_pretrained=BASE, subfolder="tokenizer", local_files_only=True)
    add_tokens(
        tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"], num_vectors_per_token=10,
    )
    mismatch = (
        load_model(pipe.unet, WEIGHTS / "unet/unet.safetensors"),
        load_model(pipe.text_encoder, WEIGHTS / "text_encoder/text_encoder.safetensors"),
    )
    if any(part for pair in mismatch for part in pair):
        raise RuntimeError(f"PowerPaint weight mismatch: {mismatch}")
    pipe.to("cuda:0")
    for name, module in {"unet": pipe.unet, "vae": pipe.vae, "text_encoder": pipe.text_encoder}.items():
        devices = {str(parameter.device) for parameter in module.parameters()}
        hooks = sum(getattr(child, "_hf_hook", None) is not None for child in module.modules())
        if devices != {"cuda:0"} or hooks:
            raise RuntimeError(f"{name} is not full CUDA: devices={devices}, hooks={hooks}")
    return pipe


def inward_feather(mask: torch.Tensor, radius: int = 9) -> torch.Tensor:
    sigma = max(1.0, radius / 2.5)
    x = torch.arange(-radius, radius + 1, device=mask.device, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma)); kernel /= kernel.sum()
    alpha = F.conv2d(mask[None, None].float(), kernel.view(1, 1, 1, -1), padding=(0, radius))
    alpha = F.conv2d(alpha, kernel.view(1, 1, -1, 1), padding=(radius, 0))[0, 0]
    return alpha.clamp(0, 1) * mask.float()


def foliage_palette(path: Path | None) -> tuple[float, float, float] | None:
    if path is None or not path.is_file():
        return None
    rgb = np.asarray(Image.open(path).convert("RGB").resize((256, 256)), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    selected = (g > r * 1.05) & (g > b * .98) & (g > 35) & ((g - r) > 6)
    if int(selected.sum()) < 100:
        return None
    return tuple(float(value) for value in np.median(rgb[selected], axis=0))


def most_obvious_crown(image: Image.Image, semantic_vegetation: Image.Image | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    rgb=np.asarray(image.convert("RGB").resize((512,512),Image.Resampling.LANCZOS),dtype=np.uint8)
    f=rgb.astype(np.float32);r,g,b=f[...,0],f[...,1],f[...,2]
    # The semantic model can occasionally leak road, sky or the survey-car bonnet into
    # the vegetation class.  It is therefore a spatial prior, never a donor-pixel mask
    # by itself.  Real crown material must also have leaf-like colour/texture evidence.
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_mean=cv2.GaussianBlur(gray,(0,0),3.0)
    local_square=cv2.GaussianBlur(gray*gray,(0,0),3.0)
    local_std=np.sqrt(np.maximum(0.0,local_square-local_mean*local_mean))
    excess_green=g-(r+b)*.5
    green=(g>34)&(g>r*.96)&(g>b*.91)&(excess_green>2.0)
    dark_leaf=(g>22)&(g<145)&(g>r*.90)&(g>b*.86)&(local_std>10.0)
    leaf_evidence=(green|dark_leaf)&(local_std>6.0)
    if semantic_vegetation is not None:
        semantic=np.asarray(
            semantic_vegetation.convert("L").resize((512,512),Image.Resampling.NEAREST),
            dtype=np.uint8,
        )>127
        foliage=(semantic&leaf_evidence).astype(np.uint8)
    else:
        foliage=leaf_evidence.astype(np.uint8)
    foliage[int(rgb.shape[0]*.78):,:]=0
    foliage=cv2.morphologyEx(foliage,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)
    foliage=cv2.morphologyEx(foliage,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=1)
    # A continuous tree belt may join across most of the frame. Returning that whole
    # component makes its bounding box contain sky, roads and buildings. Select a
    # dense leaf-texture window so the visual reference is a real crown close-up.
    integral=cv2.integral(foliage.astype(np.uint8))
    best_window=None;best_window_score=-1.0
    for size in (72,96,128,160):
        for y in range(0,min(376,512-size)+1,16):
            for x in range(0,512-size+1,16):
                area=int(integral[y+size,x+size]-integral[y,x+size]-integral[y+size,x]+integral[y,x])
                density=area/float(size*size)
                if area<360 or density<.16:continue
                upper_weight=1.15-.35*((y+size*.5)/512.0)
                score=(density**1.7)*(size**1.25)*upper_weight
                if score>best_window_score:
                    best_window_score=score;best_window=(x,y,size)
    if best_window is not None:
        x,y,size=best_window
        selected=np.zeros_like(foliage,dtype=bool)
        selected[y:y+size,x:x+size]=foliage[y:y+size,x:x+size]>0
        return rgb,selected,int(selected.sum())
    count,labels,stats,_=cv2.connectedComponentsWithStats(foliage,8)
    if count<=1:return rgb,np.zeros((512,512),dtype=bool),0
    candidates=[]
    for index in range(1,count):
        x,y,w,h,area=[int(value) for value in stats[index]]
        aspect=w/max(h,1)
        if area>=80 and w>=12 and h>=12 and y<360 and aspect<6.0:
            candidates.append(index)
    if not candidates:return rgb,np.zeros((512,512),dtype=bool),0
    def crown_score(index: int) -> float:
        x,y,w,h,area=[int(value) for value in stats[index]]
        centre_y=y+h/2
        position_weight=max(.35,1.35-centre_y/512)
        shape_weight=min(1.0,4.0/max(w/max(h,1),1.0))
        return area*position_weight*shape_weight
    best=max(candidates,key=crown_score)
    return rgb,labels==best,int(stats[best,cv2.CC_STAT_AREA])


def foliage_conditioned_source(
    source: Image.Image, reference: Path | None, hard_mask: np.ndarray,
    semantic_vegetation: Image.Image | None = None,
    reference_semantic_vegetation: Image.Image | None = None,
) -> tuple[Image.Image, str, np.ndarray]:
    """Clone-stamp real crown texture into the editable region; never copy trunks."""
    source_rgb,source_foliage,source_area=most_obvious_crown(source,semantic_vegetation)
    if source_area>=240:
        donor=source_rgb;foliage=source_foliage;method="same_image_most_obvious_crown_first"
    elif reference and reference.is_file():
        donor,foliage,reference_area=most_obvious_crown(Image.open(reference).convert("RGB"),reference_semantic_vegetation)
        method="nearby_coordinate_most_obvious_crown_fallback"
        if reference_area<160:donor,foliage,reference_area=source_rgb,source_foliage,source_area;method="same_image_weak_crown_last_resort"
    else:
        donor=source_rgb;foliage=source_foliage;method="same_image_weak_crown_last_resort"
    if int(foliage.sum()) < 100:
        return source, "no_reliable_foliage_prefill", np.zeros_like(hard_mask, dtype=np.float32)
    # Crop a dense interior patch, not the full component bounding box.  The latter
    # often contains large sky gaps even when the component itself is a valid tree.
    dense_integral=cv2.integral(foliage.astype(np.uint8))
    dense_choice=None
    for size in (96,80,72,64,48,32):
        best_density=-1.0;best_xy=None
        for yy in range(0,513-size,4):
            for xx in range(0,513-size,4):
                amount=int(dense_integral[yy+size,xx+size]-dense_integral[yy,xx+size]-dense_integral[yy+size,xx]+dense_integral[yy,xx])
                density=amount/float(size*size)
                if density>best_density:best_density=density;best_xy=(xx,yy)
        if best_xy is not None and best_density>=.82:
            x0,y0=best_xy;x1,y1=x0+size,y0+size;dense_choice=True;break
    if dense_choice is None:
        ys, xs = np.where(foliage);margin = 2
        y0,y1=max(0,int(ys.min())-margin),min(512,int(ys.max())+margin+1)
        x0,x1=max(0,int(xs.min())-margin),min(512,int(xs.max())+margin+1)
    crop=donor[y0:y1,x0:x1];crop_mask=foliage[y0:y1,x0:x1]
    crop=crop.astype(np.uint8);crop_mask=cv2.morphologyEx(crop_mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8),iterations=1)
    # Nearest-leaf content repair: every texture pixel is sourced from a semantic vegetation
    # pixel, so sky, signs, fences and poles inside the crown bounding box cannot leak in.
    non_foliage=(1-crop_mask).astype(bool)
    # Exact nearest-leaf indices prevent holes in a label lookup table from leaking
    # background structures into the crown texture.
    _,nearest_indices=ndimage.distance_transform_edt(non_foliage,return_indices=True)
    repaired_crop=crop[nearest_indices[0],nearest_indices[1]]
    conditioned=np.asarray(source,dtype=np.uint8).copy();organic=np.zeros(hard_mask.shape,dtype=np.float32)
    component_count,labels,stats,_=cv2.connectedComponentsWithStats(hard_mask.astype(np.uint8),8)
    donor_aspect=max(.55,min(1.8,crop.shape[1]/max(crop.shape[0],1)))
    for component in range(1,component_count):
        x,y,w,h,area=[int(v) for v in stats[component]]
        if area<20:continue
        tile_h=max(24,min(max(h,24),crop.shape[0]))
        tile_w=max(24,int(tile_h*donor_aspect))
        tile=cv2.resize(repaired_crop,(tile_w,tile_h),interpolation=cv2.INTER_LANCZOS4)
        repeat_y=max(1,int(np.ceil(h/tile_h)));repeat_x=max(1,int(np.ceil(w/tile_w)))
        texture=np.tile(tile,(repeat_y,repeat_x,1))[:h,:w]
        component_clip=(labels[y:y+h,x:x+w]==component)
        region=conditioned[y:y+h,x:x+w]
        region[component_clip]=texture[component_clip]
        conditioned[y:y+h,x:x+w]=region
        organic[y:y+h,x:x+w]=np.maximum(organic[y:y+h,x:x+w],component_clip.astype(np.float32))
    # Feather the copied crown inside the immutable planner mask for a natural repair boundary.
    organic=np.asarray(Image.fromarray((organic*255).astype(np.uint8),"L").filter(ImageFilter.GaussianBlur(5.0)),dtype=np.float32)/255.0
    organic*=hard_mask.astype(np.float32)
    return Image.fromarray(conditioned,"RGB"),method,organic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-vegetation-mask", type=Path)
    parser.add_argument("--vegetation-mask", type=Path)
    parser.add_argument("--semantic-labels", type=Path)
    parser.add_argument("--protected-mask", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--single-seed-preflight", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter(); pipe = None
    try:
        source_original = Image.open(args.image).convert("RGB")
        original_size = source_original.size
        source = source_original.resize((512, 512), Image.Resampling.LANCZOS)
        label = np.asarray(Image.open(args.mask).convert("L").resize((512, 512), Image.Resampling.NEAREST), dtype=np.uint8)
        proposal_np = label > 127
        semantic_labels = Image.open(args.semantic_labels).convert("L") if args.semantic_labels and args.semantic_labels.is_file() else None
        protected = Image.open(args.protected_mask).convert("L") if args.protected_mask and args.protected_mask.is_file() else None
        core_np, hard_np = prepare_edit_masks(proposal_np, semantic_labels, protected)
        if float(core_np.mean()) < .0005:
            raise RuntimeError("No reliable tree-canopy proposal region is available")
        device = torch.device("cuda:0")
        hard = torch.from_numpy(hard_np.copy()).to(device)
        alpha = inward_feather(hard)
        mask_image = Image.fromarray((hard_np * 255).astype(np.uint8), "L")
        base = torch.from_numpy(np.asarray(source, dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
        reference_palette = foliage_palette(args.reference)
        semantic_vegetation=Image.open(args.vegetation_mask).convert("L") if args.vegetation_mask and args.vegetation_mask.is_file() else None
        reference_semantic=Image.open(args.reference_vegetation_mask).convert("L") if args.reference_vegetation_mask and args.reference_vegetation_mask.is_file() else None
        # The retrieved crown is a visual reference and palette cue only. It must
        # never be pasted into the diffusion input: doing so makes low-strength
        # inpainting preserve a clone-stamp-like pre-composite instead of performing
        # genuine context-aware image editing.
        conditioned_source, conditioning_method, organic_alpha_np = foliage_conditioned_source(source, args.reference, hard_np, semantic_vegetation, reference_semantic)
        args.output_directory.mkdir(parents=True, exist_ok=True)
        mask_image.save(args.output_directory / "tree_edit_mask.png")
        pipe = load_pipeline(); torch.cuda.reset_peak_memory_stats()
        prompt = "P_shape broad connected mature leafy crowns, dense natural canopy, correct street perspective, photorealistic"
        context = "P_ctxt fill the mask with crown foliage; preserve road, sky, buildings and street objects"
        negative = "tree trunk, branch, root, whole tree, building, wall, roof, window, billboard, awning, shrub, hedge, road obstruction, distortion, text, blur"
        core = torch.from_numpy(core_np.copy()).to(device)
        base_pixels = base[:, core]
        base_foliage = ((base_pixels[1] > base_pixels[0] * 1.03) & (base_pixels[1] > base_pixels[2] * .96) & (base_pixels[1] > 38)).float().mean().item()
        records = []
        conditioned_gpu = torch.from_numpy(np.asarray(conditioned_source, dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
        organic_alpha=torch.from_numpy(organic_alpha_np).to(device=device,dtype=torch.float32)
        reference_composite = base * (1 - organic_alpha.unsqueeze(0)) + conditioned_gpu * organic_alpha.unsqueeze(0)
        reference_pixels = reference_composite[:, hard]
        reference_foliage = ((reference_pixels[1] > reference_pixels[0] * 1.03) & (reference_pixels[1] > reference_pixels[2] * .96) & (reference_pixels[1] > 38)).float().mean().item()
        reference_changed = float(((reference_composite - base).abs().mean(0) > 4)[hard].float().mean().item())
        reference_output = Image.fromarray(reference_composite.round().clamp(0,255).byte().permute(1,2,0).cpu().numpy(),"RGB")
        if original_size != (512,512):reference_output=reference_output.resize(original_size,Image.Resampling.LANCZOS)
        reference_path=args.output_directory/"tree_scenario_reference_foliage.png";reference_output.save(reference_path,compress_level=3)
        for seed in (SEEDS[:1] if args.single_seed_preflight else SEEDS):
            tic = time.perf_counter()
            generator = torch.Generator(device="cuda:0").manual_seed(seed)
            with torch.inference_mode():
                result = pipe(
                    promptA=prompt, promptB=context,
                    negative_promptA="P_shape " + negative, negative_promptB="P_ctxt " + negative,
                    image=source, mask=mask_image, width=512, height=512,
                    strength=.68, tradoff=.60, tradoff_nag=.60,
                    guidance_scale=10.5, num_inference_steps=40, generator=generator,
                )
            torch.cuda.synchronize()
            generated = torch.from_numpy(np.asarray(result.images[0].convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1).to(device, dtype=torch.float32)
            if reference_palette is not None:
                target = torch.tensor(reference_palette, device=device).view(3, 1)
                pixels = generated[:, hard]
                green = (pixels[1] > pixels[0] * 1.03) & (pixels[1] > pixels[2] * .96)
                if bool(green.any()):
                    current = pixels[:, green].median(dim=1).values.view(3, 1)
                    adjusted = (pixels[:, green] + (target - current) * .18).clamp(0, 255)
                    pixels[:, green] = adjusted; generated[:, hard] = pixels
            composite = base * (1 - alpha.unsqueeze(0)) + generated * alpha.unsqueeze(0)
            outside_max = int((composite[:, ~hard] - base[:, ~hard]).abs().max().item())
            changed = float(((composite - base).abs().mean(0) > 4)[core].float().mean().item())
            final_pixels = composite[:, core]
            final_foliage = ((final_pixels[1] > final_pixels[0] * 1.03) & (final_pixels[1] > final_pixels[2] * .96) & (final_pixels[1] > 38)).float().mean().item()
            output = Image.fromarray(composite.round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy(), "RGB")
            if original_size != (512, 512):
                output = output.resize(original_size, Image.Resampling.LANCZOS)
            path = args.output_directory / f"tree_scenario_seed_{seed}.png"; output.save(path, compress_level=3)
            records.append({
                "seed": seed, "path": str(path.resolve()),
                "generation_method": "powerpaint_original_scene_shape_guided_diffusion_inpaint",
                "outside_mask_max_difference": outside_max,
                "edit_context_ratio": round(float(hard_np.mean()), 4),
                "changed_fraction_inside_mask": round(changed, 4),
                "foliage_fraction_inside_mask_before": round(float(base_foliage), 4),
                "foliage_fraction_inside_mask_after": round(float(final_foliage), 4),
                "foliage_gain_inside_mask": round(float(final_foliage - base_foliage), 4),
                "inference_seconds": round(time.perf_counter() - tic, 3),
            })
        summary = {
            "status": "PASS", "model": "PowerPaint-v1 semantic-crown layout diffusion editor",
            "generation_mode": "original_scene_shape_guided_diffusion_with_reference_palette_only",
            "full_gpu_verified": True, "cpu_fallback": False,
            "reference_image": str(args.reference.resolve()) if args.reference and args.reference.is_file() else None,
            "reference_foliage_palette_rgb": reference_palette,
            "conditioning_method": f"{conditioning_method}_reference_only_not_pixel_prefill",
            "mask_ratio": round(float(core_np.mean()), 4), "edit_context_ratio": round(float(hard_np.mean()), 4), "variants": records,
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }
        print(json.dumps(summary, ensure_ascii=False)); return 0
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False)); return 1
    finally:
        pipe = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
