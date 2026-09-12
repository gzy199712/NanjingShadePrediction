"""stage_41b: rerun targeted walkable segmentation and depth-ground localization on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation, Mask2FormerForUniversalSegmentation
import yaml

from stage_29_streaming_gpu_localization import depth_input, group_matrix, raw_image, semantic_input, surface_geometry


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/targeted_pixel_relocalization.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_41b"


def resolve(value: str) -> Path:
    return ROOT / value


def logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_41b")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_41b.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def headings(value: str) -> list[int]:
    return [int(item) for item in str(value).split("|") if str(item).strip()]


def jobs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_csv(resolve(cfg["inputs"]["fusion_decisions"]), dtype={"point_id": str})
    selected = decisions.loc[decisions.fusion_decision.eq("PROMOTE_TO_PIXEL_LOCALIZATION")].copy()
    manifest = pd.read_csv(resolve(cfg["inputs"]["rescue_manifest"]), dtype={"point_id": str})
    rows = []
    for decision in selected.itertuples(index=False):
        record = manifest.loc[manifest.point_id.eq(decision.point_id)].iloc[0]
        for center in headings(decision.tool_supported_headings):
            for offset in cfg["rules"]["neighbor_offsets"]:
                heading = int((center + int(offset)) % 360)
                # Four-view manifest stores cardinals only; derive adjacent filenames from the same point prefix.
                cardinal_path = Path(str(record[f"heading_{center}_path"]))
                parts = cardinal_path.stem.split("_")
                parts[3] = str(heading)
                path = cardinal_path.with_name("_".join(parts) + cardinal_path.suffix)
                rows.append({"point_id": str(decision.point_id), "benchmark_role": decision.benchmark_role, "center_heading": center, "heading": heading, "filepath": str(path)})
    frame = pd.DataFrame(rows).drop_duplicates(["point_id", "heading"]).sort_values(["point_id", "heading"])
    frame["file_exists"] = frame.filepath.map(lambda value: Path(value).is_file())
    return selected, frame


def save_gpu_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def localize(semantic_model, depth_model, group_map: torch.Tensor, rgb: torch.Tensor, base: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[float, torch.Tensor], torch.Tensor]:
    size = int(cfg["runtime"]["semantic_probability_size"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = semantic_model(
            pixel_values=semantic_input(rgb, int(base["camera"]["image_size"])),
            pixel_mask=torch.ones((1, int(base["camera"]["image_size"]), int(base["camera"]["image_size"])), device=rgb.device, dtype=torch.bool),
        )
        query_class = output.class_queries_logits.float().softmax(-1)[0, :, :-1]
        query_mask = output.masks_queries_logits.float().sigmoid()[0]
        probability = torch.einsum("qc,qhw->chw", query_class, query_mask)
        probability = F.interpolate(probability[None], size=(size, size), mode="bilinear", align_corners=False)[0]
        probability = probability / probability.sum(0, keepdim=True).clamp_min(1e-8)
        depth = depth_model(pixel_values=depth_input(rgb)).predicted_depth
        depth = F.interpolate(depth.unsqueeze(1), size=(size, size), mode="bicubic", align_corners=False, antialias=True)[0, 0].float().clamp(0.05, 80)
    common = torch.einsum("gc,chw->ghw", group_map, probability).clamp(0, 1)
    confidence = probability.max(0).values
    normal, jump, valid = surface_geometry(depth, float(base["camera"]["fov_degrees"]), int(base["processing"]["depth_smoothing_kernel"]))
    ground = normal[1].abs().clamp(0, 1)
    occlusion = (jump >= float(cfg["rules"]["depth_discontinuity_log_threshold"])) & valid
    walkable, obstacle = common[0], common[3]
    score = walkable * (1 - obstacle) * ground * confidence
    lower = torch.arange(size, device=rgb.device)[:, None] >= size // 2
    base_gate = valid & ~occlusion & (ground >= float(cfg["rules"]["ground_normal_abs_y_minimum"])) & (walkable >= float(cfg["rules"]["semantic_walkable_probability_minimum"]))
    if cfg["rules"]["lower_half_only"]:
        base_gate &= lower
    masks = {float(threshold): base_gate & (score >= float(threshold)) for threshold in cfg["rules"]["score_thresholds"]}
    primary = masks[float(cfg["rules"]["primary_score_threshold"])]
    y, x = torch.where(primary)
    bbox = None if not x.numel() else [float(x.min() / size), float(y.min() / size), float((x.max() + 1) / size), float((y.max() + 1) / size)]
    metrics = {
        "walkable_probability_mean": float(walkable.mean()),
        "lower_half_walkable_probability": float(walkable[lower.expand_as(walkable)].mean()),
        "ground_score_mean": float(ground[valid].mean()),
        "obstacle_probability_mean": float(obstacle.mean()),
        "primary_mask_ratio": float(primary.float().mean()),
        "primary_bbox_norm": json.dumps(bbox, separators=(",", ":")) if bbox else "",
        **{f"mask_ratio_{str(threshold).replace('.', '_')}": float(mask.float().mean()) for threshold, mask in masks.items()},
    }
    return metrics, masks, score


def overlay(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    image = F.interpolate(rgb[None].float(), size=mask.shape, mode="bilinear", align_corners=False)[0]
    color = torch.tensor([20.0, 220.0, 120.0], device=rgb.device)[:, None, None]
    alpha = mask.float()[None] * 0.55
    return image * (1 - alpha) + color * alpha


def summarize_points(view_metrics: pd.DataFrame, selected: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for decision in selected.itertuples(index=False):
        point = view_metrics.loc[view_metrics.point_id.eq(str(decision.point_id))]
        confirmed = []
        for center in headings(decision.tool_supported_headings):
            neighbors = {(center + int(offset)) % 360 for offset in cfg["rules"]["neighbor_offsets"]}
            support = int(point.loc[point.heading.isin(neighbors), "primary_mask_ratio"].ge(float(cfg["rules"]["minimum_primary_mask_ratio"])).sum())
            if support >= int(cfg["rules"]["minimum_supported_neighbor_views"]):
                confirmed.append(center)
        rows.append({
            "point_id": str(decision.point_id), "benchmark_role": decision.benchmark_role,
            "confirmed_center_headings": "|".join(map(str, confirmed)), "confirmed_center_count": len(confirmed),
            "pixel_localization_decision": "PIXEL_LOCALIZATION_CONFIRMED" if confirmed else "PIXEL_LOCALIZATION_REJECTED",
            "generation_unlocked": False,
        })
    return pd.DataFrame(rows)


def write_report(summary: dict[str, Any], points: pd.DataFrame, path: Path) -> None:
    rows = "\n".join(f"| {row.point_id} | {row.benchmark_role} | {row.confirmed_center_headings or '-'} | {row.pixel_localization_decision} |" for row in points.itertuples(index=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# stage_41b Targeted GPU pixel relocalization

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Views: {summary['view_count']}
- Failures: {summary['failure_count']}
- Peak reserved VRAM: {summary['peak_vram_gib']:.3f} GiB
- Confirmed points: {summary['confirmed_point_count']}
- Generation unlock count: 0

| Point | Role | Confirmed headings | Decision |
|---|---|---|---|
{rows}

Masks are generated by CUDA semantic and metric-depth models with ground-normal and depth-discontinuity
gates. Confirmation requires two adjacent views. It is evidence for later planting-space analysis, not
a planting design and not permission to generate an image.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    log = logger()
    failed_path = LOG_DIR / "failed_files.csv"
    try:
        cfg = config()
        base = yaml.safe_load(resolve(cfg["inputs"]["base_localization_config"]).read_text(encoding="utf-8"))
        for value in cfg["inputs"].values():
            require(resolve(value))
        selected, frame = jobs(cfg)
        model_paths = {key: Path(value["local_path"]) for key, value in base["models"].items() if key in {"semantic", "depth"}}
        check = {"status": "PASS" if torch.cuda.is_available() and len(selected) == 2 and len(frame) > 0 and frame.file_exists.all() and all(path.exists() for path in model_paths.values()) else "FAIL", "selected_points": len(selected), "views": len(frame), "missing_views": int((~frame.file_exists).sum()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.mode == "check":
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0 if check["status"] == "PASS" else 2
        out = resolve(cfg["outputs"]["directory"])
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {out}")
        out.mkdir(parents=True, exist_ok=True)
        device = torch.device(cfg["runtime"]["device"])
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sem = Mask2FormerForUniversalSegmentation.from_pretrained(model_paths["semantic"], local_files_only=True, dtype=torch.float16).to(device).eval()
        dep = AutoModelForDepthEstimation.from_pretrained(model_paths["depth"], local_files_only=True, dtype=torch.float16).to(device).eval()
        if any(parameter.device.type != "cuda" for model in (sem, dep) for parameter in model.parameters()):
            raise RuntimeError("Non-CUDA model tensor detected")
        labels = {int(key): value for key, value in sem.config.id2label.items()}
        groups = group_matrix(labels, base["semantic_groups"], device)
        rows, errors = [], []
        for job in tqdm(frame.to_dict("records"), desc="Targeted CUDA pixel relocalization", unit="view", dynamic_ncols=True):
            try:
                _, rgb = raw_image(Path(job["filepath"]), device)
                metrics, masks, _ = localize(sem, dep, groups, rgb, base, cfg)
                point_dir = out / "masks" / str(job["point_id"])
                primary = masks[float(cfg["rules"]["primary_score_threshold"])]
                save_gpu_image(primary.float()[None].repeat(3, 1, 1) * 255, point_dir / f"{job['heading']}_walkable_mask.png")
                save_gpu_image(overlay(rgb, primary), point_dir / f"{job['heading']}_overlay.png")
                rows.append({**{key: job[key] for key in ("point_id", "benchmark_role", "center_heading", "heading", "filepath")}, **metrics})
            except Exception as error:
                log.exception("View %s/%s failed", job["point_id"], job["heading"])
                errors.append({"point_id": job["point_id"], "filename": job["filepath"], "error_message": f"{type(error).__name__}: {error}"})
                torch.cuda.empty_cache()
        pd.DataFrame(errors, columns=["point_id", "filename", "error_message"]).to_csv(failed_path, index=False, encoding="utf-8-sig")
        metrics = pd.DataFrame(rows)
        metrics.to_csv(out / "view_pixel_metrics.csv", index=False, encoding="utf-8-sig")
        points = summarize_points(metrics, selected, cfg)
        points.to_csv(out / "point_pixel_decisions.csv", index=False, encoding="utf-8-sig")
        summary = {**check, "status": "PILOT_COMPLETE" if len(metrics) == len(frame) and not errors else "PILOT_INCOMPLETE", "generated": datetime.now().astimezone().isoformat(), "view_count": len(metrics), "failure_count": len(errors), "peak_vram_gib": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3), "confirmed_point_count": int(points.confirmed_center_count.gt(0).sum()), "generation_unlock_count": 0}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(summary, points, resolve(cfg["outputs"]["report"]))
        log.info("stage_41b %s: views=%d confirmed=%d peak=%.3f GiB", summary["status"], len(metrics), summary["confirmed_point_count"], summary["peak_vram_gib"])
        return 0 if summary["status"] == "PILOT_COMPLETE" else 2
    except Exception as error:
        log.exception("stage_41b failed")
        with failed_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_41b", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

