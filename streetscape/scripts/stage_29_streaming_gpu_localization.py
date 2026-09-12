"""Storage-safe, resumable CUDA localization for the citywide planner manifest."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
from tqdm import tqdm
from transformers import (AutoModelForDepthEstimation,
                          AutoModelForZeroShotObjectDetection, AutoProcessor,
                          Mask2FormerForUniversalSegmentation)
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "streetscape/configs/stage_29_streaming_gpu_localization.yaml"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_29"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def normalize_label(value: str) -> str:
    return str(value).lower().replace("(", "").replace(")", "").replace("-", " ").replace("_", " ").strip()


def group_matrix(labels: dict[int, str], groups: dict, device: torch.device) -> torch.Tensor:
    names = [normalize_label(labels[index]) for index in range(len(labels))]
    matrix = torch.zeros((len(groups), len(names)), device=device)
    for group_index, terms in enumerate(groups.values()):
        terms = {normalize_label(term) for term in terms}
        for class_index, name in enumerate(names):
            if name in terms: matrix[group_index, class_index] = 1
    return matrix


def raw_image(path: Path, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    with Image.open(path) as image: array = np.asarray(image.convert("RGB")).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device=device, dtype=torch.uint8)
    return array, tensor


def semantic_input(rgb: torch.Tensor, size: int) -> torch.Tensor:
    value = F.interpolate(rgb[None].half() / 255, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([.485, .456, .406], device=rgb.device, dtype=torch.float16)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=rgb.device, dtype=torch.float16)[None, :, None, None]
    return (value - mean) / std


def depth_input(rgb: torch.Tensor) -> torch.Tensor:
    value = F.interpolate(rgb[None].half() / 255, size=(518, 518), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor([.485, .456, .406], device=rgb.device, dtype=torch.float16)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=rgb.device, dtype=torch.float16)[None, :, None, None]
    return (value - mean) / std


def surface_geometry(depth: torch.Tensor, fov: float, kernel: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = depth.shape[-1]; focal = size / (2 * math.tan(math.radians(fov) / 2))
    smooth = F.avg_pool2d(depth[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]
    y, x = torch.meshgrid(torch.arange(size, device=depth.device), torch.arange(size, device=depth.device), indexing="ij")
    points = torch.stack(((x-(size-1)/2)/focal*smooth, (y-(size-1)/2)/focal*smooth, smooth), 0)
    dx = F.pad(points[:, :, 2:] - points[:, :, :-2], (1, 1, 0, 0), mode="replicate")
    dy = F.pad(points[:, 2:, :] - points[:, :-2, :], (0, 0, 1, 1), mode="replicate")
    normal = F.normalize(torch.cross(dx, dy, dim=0), dim=0, eps=1e-6)
    log_depth = smooth.clamp(.05, 100).log()
    gx = F.pad((log_depth[:, 1:] - log_depth[:, :-1]).abs(), (0, 1), mode="replicate")
    gy = torch.cat(((log_depth[1:] - log_depth[:-1]).abs(), (log_depth[-1:] - log_depth[-2:-1]).abs()), 0)
    jump = torch.maximum(gx, gy); valid = torch.isfinite(normal).all(0) & torch.isfinite(smooth) & (smooth > .05)
    return normal, jump, valid


def semantic_depth_metrics(semantic_model, depth_model, rgb: torch.Tensor, group_map: torch.Tensor, cfg: dict, boxes_json: str = "[]") -> dict:
    psize = int(cfg["processing"]["semantic_probability_size"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = semantic_model(pixel_values=semantic_input(rgb, int(cfg["camera"]["image_size"])),
                                pixel_mask=torch.ones((1, int(cfg["camera"]["image_size"]), int(cfg["camera"]["image_size"])), device=rgb.device, dtype=torch.bool))
        query_class = output.class_queries_logits.float().softmax(-1)[0, :, :-1]
        query_mask = output.masks_queries_logits.float().sigmoid()[0]
        probability = torch.einsum("qc,qhw->chw", query_class, query_mask)
        probability = F.interpolate(probability[None], size=(psize, psize), mode="bilinear", align_corners=False)[0]
        probability = probability / probability.sum(0, keepdim=True).clamp_min(1e-8)
        depth = depth_model(pixel_values=depth_input(rgb)).predicted_depth
        depth = F.interpolate(depth.unsqueeze(1), size=(psize, psize), mode="bicubic", align_corners=False, antialias=True)[0, 0].float().clamp(.05, 80)
    common = torch.einsum("gc,chw->ghw", group_map, probability).clamp(0, 1)
    confidence = probability.max(0).values; normal, jump, valid = surface_geometry(depth, float(cfg["camera"]["fov_degrees"]), int(cfg["processing"]["depth_smoothing_kernel"]))
    ground = normal[1].abs().clamp(0, 1); occlusion = (jump >= float(cfg["processing"]["depth_discontinuity_log_threshold"])) & valid
    walkable, obstacle = common[0], common[3]
    score = walkable * (1-obstacle) * ground * confidence
    candidate = (score >= float(cfg["processing"]["walkable_score_threshold"])) & (ground >= float(cfg["processing"]["ground_normal_abs_y_minimum"])) & valid & ~occlusion
    lower = torch.arange(psize, device=rgb.device)[:, None] >= psize // 2
    boxes = json.loads(boxes_json if isinstance(boxes_json, str) and boxes_json else "[]")
    for box in boxes:
        x1=max(0,int(math.floor(float(box["x1"])*psize))); x2=min(psize,int(math.ceil(float(box["x2"])*psize)))
        y1=max(0,int(math.floor(float(box["y1"])*psize))); y2=min(psize,int(math.ceil(float(box["y2"])*psize)))
        patch=common[1,y1:y2,x1:x2]; box["vegetation_support"]=round(float(patch.mean()) if patch.numel() else 0.0,6)
    return {"mean_confidence": float(confidence.mean()), "sky_ratio": float(common[2].mean()), "vegetation_ratio": float(common[1].mean()),
            "walkable_probability_mean": float(walkable.mean()), "lower_half_walkable_probability": float(walkable[lower.expand_as(walkable)].mean()),
            "obstacle_probability_mean": float(obstacle.mean()), "depth_median_proxy": float(depth.median()),
            "depth_occlusion_ratio": float(occlusion.float().mean()), "ground_score_mean": float(ground[valid].mean()),
            "retained_walkable_ratio": float(candidate.float().mean()), "tree_boxes_json": json.dumps(boxes,separators=(",",":"))}


def object_prompt(processor, taxonomy: list[dict], device: torch.device):
    parts, spans, cursor = [], [], 0
    for item in taxonomy:
        phrase = item["phrase"].strip().lower().rstrip("."); parts.append(phrase+"."); spans.append((cursor, cursor+len(phrase))); cursor += len(phrase)+2
    encoded = processor.tokenizer(" ".join(parts), return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0]; masks = [((offsets[:, 1] > start) & (offsets[:, 0] < end)) for start, end in spans]
    return {key: value.to(device) for key, value in encoded.items()}, torch.stack(masks).to(device)


def object_metrics(model, rgb: torch.Tensor, text_inputs: dict, token_masks: torch.Tensor, cfg: dict) -> dict:
    size = int(cfg["objects"]["input_size"]); image = semantic_input(rgb, size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model(pixel_values=image, pixel_mask=torch.ones((1, size, size), device=rgb.device, dtype=torch.bool), **text_inputs)
    logits = output.logits[0].sigmoid(); masks = token_masks
    if masks.shape[1] < logits.shape[1]: masks = F.pad(masks, (0, logits.shape[1]-masks.shape[1]), value=False)
    class_scores = torch.stack([logits[:, mask].mean(1) for mask in masks], 1); scores, classes = class_scores.max(1)
    keep = scores >= float(cfg["objects"]["initial_box_threshold"]); scores, classes, boxes = scores[keep], classes[keep], output.pred_boxes[0][keep]
    if not boxes.numel(): return {"object_count": 0, "tree_object_count": 0, "shade_object_count": 0, "object_classes": "", "tree_boxes_json": "[]"}
    cx, cy, width, height = boxes.unbind(1); xyxy = torch.stack((cx-width/2, cy-height/2, cx+width/2, cy+height/2), 1).clamp(0, 1)
    valid = width*height >= float(cfg["objects"]["minimum_normalized_box_size"]); scores, classes, xyxy = scores[valid], classes[valid], xyxy[valid]
    if not xyxy.numel(): return {"object_count": 0, "tree_object_count": 0, "shade_object_count": 0, "object_classes": "", "tree_boxes_json": "[]"}
    keep_idx = batched_nms(xyxy, scores, classes, float(cfg["objects"]["nms_iou_threshold"]))[:int(cfg["objects"]["maximum_detections_per_view"])]
    names = [cfg["objects"]["taxonomy"][int(classes[index])]["class_name"] for index in keep_idx]
    boxes_payload = [{"class_name": cfg["objects"]["taxonomy"][int(classes[index])]["class_name"],
                      "score": round(float(scores[index]), 6),
                      "x1": round(float(xyxy[index, 0]), 6), "y1": round(float(xyxy[index, 1]), 6),
                      "x2": round(float(xyxy[index, 2]), 6), "y2": round(float(xyxy[index, 3]), 6)} for index in keep_idx]
    return {"object_count": len(names), "tree_object_count": sum(name in {"street_tree","tree_trunk"} for name in names),
            "shade_object_count": 0,
            "object_classes": ";".join(sorted(set(names))), "tree_boxes_json": json.dumps(boxes_payload, separators=(",", ":"))}


def database(path: Path, overwrite: bool) -> sqlite3.Connection:
    if overwrite and path.exists(): path.unlink()
    connection = sqlite3.connect(path); connection.execute("PRAGMA journal_mode=WAL"); connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""CREATE TABLE IF NOT EXISTS view_results(
        point_id TEXT NOT NULL, heading INTEGER NOT NULL, filename TEXT NOT NULL, month INTEGER, svf_legacy REAL, gvi_legacy REAL, fclass TEXT,
        semantic_depth_done INTEGER DEFAULT 0, object_done INTEGER DEFAULT 0, mean_confidence REAL, sky_ratio REAL, vegetation_ratio REAL,
        walkable_probability_mean REAL, lower_half_walkable_probability REAL, obstacle_probability_mean REAL, depth_median_proxy REAL,
        depth_occlusion_ratio REAL, ground_score_mean REAL, retained_walkable_ratio REAL, object_count INTEGER, tree_object_count INTEGER,
        shade_object_count INTEGER, object_classes TEXT, tree_boxes_json TEXT, error_message TEXT, updated_at TEXT, PRIMARY KEY(point_id,heading))""")
    connection.commit(); return connection


def resolve_jobs(manifest: pd.DataFrame, raw: Path, limit: int) -> pd.DataFrame:
    points = sorted(manifest.point_id.unique(), key=lambda value: int(value))[:limit]; selected = manifest[manifest.point_id.isin(points)].copy()
    requested={(str(row.point_id),int(row.heading)) for row in selected.itertuples(index=False)};index={};duplicates=set()
    for path in raw.glob("*.png"):
        parts=path.stem.split("_")
        if len(parts)<5: continue
        try:key=(parts[0],int(float(parts[3])))
        except ValueError:continue
        if key not in requested:continue
        if key in index:duplicates.add(key)
        else:index[key]=path
    selected["filepath"]=[str(index.get((str(row.point_id),int(row.heading)),"")) if (str(row.point_id),int(row.heading)) not in duplicates else "" for row in selected.itertuples(index=False)]
    return selected.sort_values(["point_id","heading"], key=lambda values: values.astype(int) if values.name in {"point_id","heading"} else values)


def upsert_base(connection: sqlite3.Connection, jobs: pd.DataFrame) -> None:
    for row in jobs.itertuples(index=False):
        connection.execute("INSERT OR IGNORE INTO view_results(point_id,heading,filename,month,svf_legacy,gvi_legacy,fclass) VALUES(?,?,?,?,?,?,?)",
                           (str(row.point_id), int(row.heading), str(row.filepath), int(row.month), float(row.SVF), float(row.GVI), str(row.fclass)))
    connection.commit()


def export_results(connection: sqlite3.Connection, output: Path) -> pd.DataFrame:
    frame = pd.read_sql_query("SELECT * FROM view_results ORDER BY CAST(point_id AS INTEGER), heading", connection)
    frame.to_csv(output/"view_metrics.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    numeric = ["sky_ratio","vegetation_ratio","walkable_probability_mean","lower_half_walkable_probability","retained_walkable_ratio","tree_object_count","shade_object_count"]
    complete = frame[(frame.semantic_depth_done==1)&(frame.object_done==1)]
    points = complete.groupby("point_id")[numeric].mean().reset_index() if len(complete) else pd.DataFrame(columns=["point_id"]+numeric)
    if len(complete):
        counts = complete.groupby("point_id").agg(view_count=("heading","count"),object_view_count=("object_count",lambda x:int((x>0).sum()))).reset_index(); points=points.merge(counts,on="point_id")
    points.to_csv(output/"point_metrics.csv", index=False, encoding="utf-8-sig"); return frame


def main() -> int:
    args = arguments(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); manifest_path=Path(cfg["inputs"]["manifest"]); raw=Path(cfg["inputs"]["raw_direction_directory"])
    manifest = pd.read_csv(manifest_path, dtype={"point_id":str}) if manifest_path.exists() else pd.DataFrame(); limit=args.max_points or int(cfg["processing"]["pilot_point_count"])
    jobs = resolve_jobs(manifest, raw, limit) if len(manifest) else pd.DataFrame(); model_paths={name:Path(value["local_path"]) for name,value in cfg["models"].items()}
    check={"status":"READY" if torch.cuda.is_available() and len(jobs)==limit*12 and jobs.filepath.ne("").all() and all(path.exists() for path in model_paths.values()) else "BLOCKED",
           "point_count":limit,"view_count":len(jobs),"missing_images":int(jobs.filepath.eq("").sum()) if len(jobs) else None,"cuda_available":torch.cuda.is_available(),
           "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"models":{key:str(value) for key,value in model_paths.items()},"cpu_fallback":False,"full_tensor_persistence":False}
    if args.mode=="check": print(json.dumps(check,ensure_ascii=False,indent=2)); return 0 if check["status"]=="READY" else 1
    if check["status"]!="READY": raise RuntimeError(check)
    output=Path(cfg["outputs"]["directory"]); output.mkdir(parents=True,exist_ok=True); connection=database(output/"checkpoint.sqlite",args.overwrite); upsert_base(connection,jobs)
    logger=make_logger(Path(cfg["outputs"]["log"])); device=torch.device(cfg["gpu_policy"]["device"]); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    started=datetime.now().astimezone(); tick=time.perf_counter(); failures=[]; logger.info("START points=%d views=%d gpu=%s",limit,len(jobs),torch.cuda.get_device_name(0))
    semantic_model=Mask2FormerForUniversalSegmentation.from_pretrained(model_paths["semantic"],local_files_only=True,dtype=torch.float16).to(device).eval()
    depth_model=AutoModelForDepthEstimation.from_pretrained(model_paths["depth"],local_files_only=True,dtype=torch.float16).to(device).eval()
    if any(parameter.device.type!="cuda" for model in (semantic_model,depth_model) for parameter in model.parameters()): raise RuntimeError("non-CUDA semantic/depth parameters")
    labels={int(key):value for key,value in semantic_model.config.id2label.items()}; group_map=group_matrix(labels,cfg["semantic_groups"],device)
    pending=pd.read_sql_query("SELECT point_id,heading,filename FROM view_results WHERE semantic_depth_done=0 ORDER BY CAST(point_id AS INTEGER),heading",connection)
    for row in tqdm(pending.itertuples(index=False),total=len(pending),desc="stage_29 semantic+depth",unit="view",dynamic_ncols=True):
        try:
            _,rgb=raw_image(Path(row.filename),device); metrics=semantic_depth_metrics(semantic_model,depth_model,rgb,group_map,cfg); now=datetime.now().astimezone().isoformat()
            columns=list(metrics); connection.execute(f"UPDATE view_results SET semantic_depth_done=1,{','.join(name+'=?' for name in columns)},error_message=NULL,updated_at=? WHERE point_id=? AND heading=?",[*metrics.values(),now,row.point_id,int(row.heading)]); connection.commit()
        except Exception as error:
            message=f"{type(error).__name__}: {error}"; failures.append({"point_id":row.point_id,"heading":row.heading,"stage":"semantic_depth","error_message":message}); connection.execute("UPDATE view_results SET error_message=?,updated_at=? WHERE point_id=? AND heading=?",(message,datetime.now().astimezone().isoformat(),row.point_id,int(row.heading))); connection.commit()
    del semantic_model,depth_model,group_map; gc.collect(); torch.cuda.empty_cache()
    processor=AutoProcessor.from_pretrained(model_paths["objects"],local_files_only=True); object_model=AutoModelForZeroShotObjectDetection.from_pretrained(model_paths["objects"],local_files_only=True,dtype=torch.float16).to(device).eval()
    if any(parameter.device.type!="cuda" for parameter in object_model.parameters()): raise RuntimeError("non-CUDA object model parameters")
    text_inputs,token_masks=object_prompt(processor,cfg["objects"]["taxonomy"],device); pending=pd.read_sql_query("SELECT point_id,heading,filename FROM view_results WHERE object_done=0 ORDER BY CAST(point_id AS INTEGER),heading",connection)
    for row in tqdm(pending.itertuples(index=False),total=len(pending),desc="stage_29 objects",unit="view",dynamic_ncols=True):
        try:
            _,rgb=raw_image(Path(row.filename),device); metrics=object_metrics(object_model,rgb,text_inputs,token_masks,cfg); now=datetime.now().astimezone().isoformat(); columns=list(metrics)
            connection.execute(f"UPDATE view_results SET object_done=1,{','.join(name+'=?' for name in columns)},error_message=NULL,updated_at=? WHERE point_id=? AND heading=?",[*metrics.values(),now,row.point_id,int(row.heading)]); connection.commit()
        except Exception as error:
            message=f"{type(error).__name__}: {error}"; failures.append({"point_id":row.point_id,"heading":row.heading,"stage":"objects","error_message":message}); connection.execute("UPDATE view_results SET error_message=?,updated_at=? WHERE point_id=? AND heading=?",(message,datetime.now().astimezone().isoformat(),row.point_id,int(row.heading))); connection.commit()
    frame=export_results(connection,output); connection.close(); pd.DataFrame(failures,columns=("point_id","heading","stage","error_message")).to_csv(output/"failed_files.csv",index=False,encoding="utf-8-sig")
    database_bytes=(output/"checkpoint.sqlite").stat().st_size; complete=int(((frame.semantic_depth_done==1)&(frame.object_done==1)).sum()); runtime=time.perf_counter()-tick
    summary={**check,"status":"STREAMING_GPU_PILOT_COMPLETE" if complete==len(jobs) and not failures else "STREAMING_GPU_PILOT_INCOMPLETE","started_at":started.isoformat(),"ended_at":datetime.now().astimezone().isoformat(),"runtime_seconds":round(runtime,3),
             "success_view_count":complete,"failure_count":len(failures),"checkpoint_bytes":database_bytes,"bytes_per_completed_view":round(database_bytes/max(1,complete),2),"full_tensor_files_written":0,
             "all_models_cuda":True,"cpu_offload":False,"cpu_fallback":False,"peak_allocated_mib":round(torch.cuda.max_memory_allocated(device)/2**20,2),"peak_reserved_mib":round(torch.cuda.max_memory_reserved(device)/2**20,2),
             "resumable":True,"object_scope":"Frozen street_tree + tree_trunk raw proposals only; artificial shade is planner-review only and is not automatically detected or placed.",
             "next_action":"Add frozen semantic-support and adjacent-view-recurrence filters, then authorize the 6,014-point formal run."}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); logger.info("END %s",summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if summary["status"].endswith("COMPLETE") else 1


if __name__=="__main__": raise SystemExit(main())
