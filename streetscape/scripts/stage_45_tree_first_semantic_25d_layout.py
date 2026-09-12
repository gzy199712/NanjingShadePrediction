"""stage_45: full-CUDA semantic/depth analysis and tree-first 2.5D candidate layout."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import sys
import time
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetscape.scripts.stage_29_streaming_gpu_localization import (  # noqa: E402
    depth_input,
    group_matrix,
    semantic_input,
    surface_geometry,
)


CONFIG = ROOT / "streetscape/configs/tree_first_25d_layout.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_45"
GROUPS = {
    "walkable": ["Sidewalk", "Pedestrian Area", "Curb Cut", "Bike Lane", "Crosswalk - Plain"],
    "vegetation": ["Vegetation"],
    "sky": ["Sky"],
    "obstacle": ["Fence", "Guard Rail", "Barrier", "Wall", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Street Light", "Pole", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Bicycle", "Bus", "Car", "Motorcycle", "Other Vehicle", "Truck"],
}


def resolve(value: str) -> Path:
    return ROOT / value


def make_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_45")
    log.setLevel(logging.INFO); log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_45.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); log.addHandler(handler)
    return log


def analyze_window(rgb: torch.Tensor, semantic, depth_model, groups: torch.Tensor, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    size = int(cfg["projection"]["analysis_size"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = semantic(
            pixel_values=semantic_input(rgb, int(cfg["projection"]["window_size"])),
            pixel_mask=torch.ones((1, int(cfg["projection"]["window_size"]), int(cfg["projection"]["window_size"])), device=rgb.device, dtype=torch.bool),
        )
        query_class = output.class_queries_logits.float().softmax(-1)[0, :, :-1]
        query_mask = output.masks_queries_logits.float().sigmoid()[0]
        probability = torch.einsum("qc,qhw->chw", query_class, query_mask)
        probability = F.interpolate(probability[None], size=(size, size), mode="bilinear", align_corners=False)[0]
        probability = probability / probability.sum(0, keepdim=True).clamp_min(1e-8)
        depth = depth_model(pixel_values=depth_input(rgb)).predicted_depth
        depth = F.interpolate(depth.unsqueeze(1), size=(size, size), mode="bicubic", align_corners=False, antialias=True)[0, 0].float().clamp(.05, 80)
    common = torch.einsum("gc,chw->ghw", groups, probability).clamp(0, 1)
    confidence = probability.max(0).values
    normal, jump, valid = surface_geometry(depth, float(cfg["projection"]["fov_degrees"]), 5)
    walkable, vegetation, _, obstacle = common
    ground = normal[1].abs().clamp(0, 1)
    layout = cfg["layout"]
    lower = torch.arange(size, device=rgb.device)[:, None] >= size // 2
    candidate = (
        (walkable * (1 - obstacle) * ground * confidence >= float(layout["walkable_score_minimum"]))
        & (confidence >= float(layout["semantic_confidence_minimum"]))
        & (ground >= float(layout["ground_normal_abs_y_minimum"]))
        & (obstacle <= float(layout["obstacle_probability_maximum"]))
        & (vegetation <= float(layout["vegetation_probability_maximum"]))
        & (depth >= float(layout["depth_minimum_m"]))
        & (depth <= float(layout["depth_maximum_m"]))
        & valid & lower
    )
    return {"candidate": candidate, "depth": depth, "walkable": walkable, "vegetation": vegetation, "obstacle": obstacle, "confidence": confidence, "ground": ground, "jump": jump}


def anchors_from_candidate(analysis: dict[str, torch.Tensor], heading: int, correspondence_path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    candidate, depth = analysis["candidate"], analysis["depth"]
    columns = torch.nonzero(candidate.any(0), as_tuple=False).flatten()
    desired = int(cfg["layout"]["desired_tree_count"])
    if columns.numel() < desired:
        return []
    positions = torch.linspace(0, columns.numel() - 1, desired, device=columns.device).round().long()
    xs = columns[positions]
    scale = int(cfg["projection"]["window_size"]) / int(cfg["projection"]["analysis_size"])
    correspondence = np.load(correspondence_path)["panorama_xy"].astype(np.float32)
    focal = int(cfg["projection"]["window_size"]) / (2 * math.tan(math.radians(float(cfg["projection"]["fov_degrees"])) / 2))
    anchors = []
    for x in xs:
        ys = torch.nonzero(candidate[:, x], as_tuple=False).flatten()
        if not ys.numel():
            continue
        y = ys.max()
        distance = float(depth[y, x])
        px = float((x + 0.5) * scale)
        py = float((y + 0.5) * scale)
        crown = float(np.clip(focal * float(cfg["layout"]["crown_diameter_m"]) / distance, cfg["layout"]["minimum_crown_diameter_pixels"], cfg["layout"]["maximum_crown_diameter_pixels"]))
        crown_center_y = py - focal * float(cfg["layout"]["crown_center_height_m"]) / distance
        radius = crown / 2
        if px - radius < 8 or px + radius > int(cfg["projection"]["window_size"]) - 8:
            continue
        if crown_center_y - radius * .72 < 8 or crown_center_y + radius * .72 > py:
            continue
        map_x = int(np.clip(round(px), 0, correspondence.shape[2] - 1)); map_y = int(np.clip(round(py), 0, correspondence.shape[1] - 1))
        anchors.append({
            "heading": heading, "base_x": round(px, 2), "base_y": round(py, 2), "depth_m": round(distance, 3),
            "panorama_x": round(float(correspondence[0, map_y, map_x]), 2), "panorama_y": round(float(correspondence[1, map_y, map_x]), 2),
            "crown_center_x": round(px, 2), "crown_center_y": round(crown_center_y, 2), "crown_diameter_px": round(crown, 2),
            "trunk_width_px": round(max(2.0, focal * float(cfg["layout"]["trunk_width_m"]) / distance), 2),
        })
    return anchors


def render_layout(rgb: torch.Tensor, anchors: list[dict[str, Any]]) -> torch.Tensor:
    height, width = rgb.shape[1:]
    yy, xx = torch.meshgrid(torch.arange(height, device=rgb.device), torch.arange(width, device=rgb.device), indexing="ij")
    overlay = rgb.float().clone(); alpha = torch.zeros((height, width), device=rgb.device)
    green = torch.tensor([38.0, 166.0, 91.0], device=rgb.device)[:, None, None]
    brown = torch.tensor([120.0, 78.0, 45.0], device=rgb.device)[:, None, None]
    for anchor in anchors:
        radius = anchor["crown_diameter_px"] / 2
        crown_mask = ((xx - anchor["crown_center_x"]) / radius).square() + ((yy - anchor["crown_center_y"]) / (radius * .72)).square() <= 1
        trunk_mask = (xx - anchor["base_x"]).abs() <= anchor["trunk_width_px"] / 2
        trunk_mask &= (yy >= anchor["crown_center_y"] + radius * .35) & (yy <= anchor["base_y"])
        crown_alpha = crown_mask.float() * .38; trunk_alpha = trunk_mask.float() * .42
        overlay = overlay * (1 - crown_alpha[None]) + green * crown_alpha[None]
        overlay = overlay * (1 - trunk_alpha[None]) + brown * trunk_alpha[None]
        alpha = torch.maximum(alpha, torch.maximum(crown_alpha, trunk_alpha))
    return overlay.clamp(0, 255)


def write_report(summary: dict[str, Any], qc: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# stage_45 Tree-first semantic/depth 2.5D layout

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Eligible points: {summary['eligible_points']}
- Analysed windows: {summary['analysed_windows']}
- Layout-ready points: {summary['layout_ready_points']}
- Total tree anchors: {summary['total_tree_anchors']}
- Full-GPU semantic/depth: {summary['full_gpu_verified']}
- Peak reserved VRAM: {summary['peak_vram_gib']:.3f} GiB

The output is a geometry proposal, not a generated photograph. Candidate bases require walkable
semantic support, ground-like depth geometry, low obstacle probability and finite 2-30 m depth.
Large 7 m crowns and sparse trunks implement the high-canopy tree-row requirement. Panorama target
coordinates are carried through the stage_44 spherical correspondence maps.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--agent-results", type=Path)
    parser.add_argument("--windows-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(); log = make_logger(); failed_path = LOG_DIR / "failed_files.csv"
    try:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        agent_path = args.agent_results.resolve() if args.agent_results else resolve(cfg["inputs"]["agent_results"]); windows_root = args.windows_directory.resolve() if args.windows_directory else resolve(cfg["inputs"]["windows"])
        if not agent_path.is_file() or not windows_root.is_dir(): raise FileNotFoundError("Agent results or perspective windows missing")
        agent = pd.read_csv(agent_path, dtype={"point_id": str})
        eligible = agent.loc[agent.policy_state.eq("READY_FOR_LAYOUT")].copy()
        model_paths = {name: Path(value) for name, value in cfg["models"].items()}
        check = {"status": "PASS" if torch.cuda.is_available() and len(eligible) >= 1 and all(path.is_dir() for path in model_paths.values()) else "FAIL", "eligible_points": len(eligible), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "models": {k: str(v) for k, v in model_paths.items()}}
        if args.mode == "check": print(json.dumps(check, ensure_ascii=False, indent=2)); return 0 if check["status"] == "PASS" else 2
        if check["status"] != "PASS": raise RuntimeError(f"Preflight failed: {check}")
        output = args.output_directory.resolve() if args.output_directory else resolve(cfg["outputs"]["directory"])
        if output.exists() and any(output.iterdir()) and not args.overwrite: raise FileExistsError(f"Output exists; use --overwrite: {output}")
        output.mkdir(parents=True, exist_ok=True); device = torch.device("cuda:0")
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        semantic = Mask2FormerForUniversalSegmentation.from_pretrained(model_paths["semantic"], local_files_only=True, dtype=torch.float16).to(device).eval()
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_paths["depth"], local_files_only=True, dtype=torch.float16).to(device).eval()
        if any(parameter.device.type != "cuda" for model in (semantic, depth_model) for parameter in model.parameters()): raise RuntimeError("Non-CUDA model parameter detected")
        labels = {int(key): value for key, value in semantic.config.id2label.items()}; groups = group_matrix(labels, GROUPS, device)
        rows, failures, all_layouts = [], [], []; started = time.perf_counter()
        jobs = [(str(row.point_id), heading) for row in eligible.itertuples(index=False) for heading in range(0, 360, 45)]
        analyses_by_point: dict[str, list[dict[str, Any]]] = {}
        for point_id, heading in tqdm(jobs, desc="Semantic-depth layout analysis", unit="window", dynamic_ncols=True):
            try:
                path = windows_root / "points" / point_id / "windows" / f"heading_{heading:03d}.png"
                with Image.open(path) as image: array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                rgb = torch.from_numpy(array).permute(2, 0, 1).to(device=device, dtype=torch.uint8)
                analysis = analyze_window(rgb, semantic, depth_model, groups, cfg)
                ratio = float(analysis["candidate"].float().mean())
                anchors = anchors_from_candidate(analysis, heading, windows_root / "points" / point_id / "correspondence" / f"heading_{heading:03d}.npz", cfg)
                analyses_by_point.setdefault(point_id, []).append({"heading": heading, "candidate_ratio": ratio, "anchors": anchors, "rgb": rgb, "analysis": analysis})
            except Exception as error:
                log.exception("Point %s heading %s failed", point_id, heading)
                failures.append({"point_id": point_id, "filename": f"heading_{heading:03d}", "error_message": f"{type(error).__name__}: {error}"})
        for point_id, candidates in analyses_by_point.items():
            best = max(candidates, key=lambda item: (len(item["anchors"]), item["candidate_ratio"]))
            ready = len(best["anchors"]) >= int(cfg["layout"]["minimum_tree_count"]) and best["candidate_ratio"] >= float(cfg["quality_gates"]["minimum_candidate_ratio"])
            point_dir = output / "points" / point_id; point_dir.mkdir(parents=True, exist_ok=True)
            layout = {"point_id": point_id, "status": "LAYOUT_READY" if ready else "NO_VALID_LAYOUT", "selected_heading": best["heading"], "candidate_ratio": best["candidate_ratio"], "tree_anchors": best["anchors"] if ready else [], "generation_unlocked": False, "next_step": "image_editor_preflight" if ready else "retain_agent_stop"}
            (point_dir / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
            candidate = F.interpolate(best["analysis"]["candidate"][None, None].float(), size=(512, 512), mode="nearest")[0, 0]
            base = best["rgb"].float(); cyan = torch.tensor([30., 180., 210.], device=device)[:, None, None]
            mask_preview = base * (1 - candidate[None] * .28) + cyan * candidate[None] * .28
            layout_preview = render_layout(mask_preview, layout["tree_anchors"])
            Image.fromarray(layout_preview.byte().permute(1, 2, 0).cpu().numpy()).save(point_dir / "layout_preview.png", compress_level=3)
            rows.append({"point_id": point_id, "status": layout["status"], "selected_heading": best["heading"], "candidate_ratio": best["candidate_ratio"], "tree_anchor_count": len(layout["tree_anchors"]), "generation_unlocked": False})
            all_layouts.extend([{**anchor, "point_id": point_id} for anchor in layout["tree_anchors"]])
        qc = pd.DataFrame(rows); qc.to_csv(output / "layout_qc.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(all_layouts).to_csv(output / "tree_anchors.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(failures, columns=["point_id", "filename", "error_message"]).to_csv(failed_path, index=False, encoding="utf-8-sig")
        peak = torch.cuda.max_memory_reserved(device) / 1024**3
        passed = not failures and len(qc) == len(eligible) and int(qc.status.eq("LAYOUT_READY").sum()) >= 1 and peak <= float(cfg["runtime"]["maximum_peak_vram_gib"])
        summary = {"status": "PASS" if passed else "LAYOUT_QUALITY_FAIL", "generated": datetime.now().astimezone().isoformat(), "runtime_seconds": time.perf_counter() - started, "eligible_points": len(eligible), "analysed_windows": len(jobs) - len(failures), "failure_count": len(failures), "layout_ready_points": int(qc.status.eq("LAYOUT_READY").sum()) if len(qc) else 0, "total_tree_anchors": int(qc.tree_anchor_count.sum()) if len(qc) else 0, "full_gpu_verified": True, "peak_vram_gib": peak}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); write_report(summary, qc, args.report.resolve() if args.report else resolve(cfg["outputs"]["report"]))
        log.info("stage_45 %s: %s", summary["status"], summary); return 0 if passed else 2
    except Exception as error:
        log.exception("stage_45 failed")
        with failed_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0: writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_45", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__": raise SystemExit(main())
