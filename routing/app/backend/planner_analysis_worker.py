"""Persistent CUDA Mask2Former worker for uploaded streetscape images."""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation, Dinov2Model, Mask2FormerForUniversalSegmentation


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models/facebook_mask2former-swin-large-cityscapes-semantic"
DINO_DIR = ROOT / "models/facebook_dinov2-base"
DEPTH_DIR = ROOT / ".local/models/depth-anything-v2-metric-outdoor-small-hf"
DINO_FEATURES = ROOT / "data/training_data/static/dinov2_window_features.npy"
DINO_INDEX = ROOT / "data/training_data/static/dinov2_window_index.csv"
SEMANTIC_FEATURES = ROOT / "data/training_data/static/semantic_features_8975.csv"
LABELS = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)


def load_reference_bank(device: torch.device) -> dict[str, Any]:
    semantic: dict[str, tuple[float, float, float]] = {}
    with SEMANTIC_FEATURES.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            semantic[str(row["point_id"])] = (float(row["SVF"]), float(row["GVI"]), float(row["BVI"]))
    index_rows: list[dict[str, str]] = []
    with DINO_INDEX.open("r", encoding="utf-8-sig", newline="") as stream:
        index_rows = sorted(csv.DictReader(stream), key=lambda row: int(row["row_index"]))
    features = np.load(DINO_FEATURES, mmap_mode="r")
    if features.shape != (len(index_rows), 8, 768):
        raise RuntimeError(f"Unexpected DINOv2 reference shape: {features.shape}")
    feature_gpu = torch.from_numpy(np.asarray(features).copy()).to(device=device, dtype=torch.float32).reshape(-1, 768)
    feature_gpu = F.normalize(feature_gpu, dim=1)
    point_ids: list[str] = []
    azimuths: list[float] = []
    svf: list[float] = []
    gvi: list[float] = []
    bvi: list[float] = []
    for row in index_rows:
        point_id = str(row["point_id"])
        point_svf, point_gvi, point_bvi = semantic[point_id]
        for window in range(8):
            point_ids.append(point_id)
            azimuths.append(float(row[f"azimuth_{window}"]))
            svf.append(point_svf); gvi.append(point_gvi); bvi.append(point_bvi)
    svf_gpu = torch.tensor(svf, device=device, dtype=torch.float32)
    good_mask = svf_gpu < .30
    good_centroid = F.normalize(feature_gpu[good_mask].mean(0), dim=0)
    poor_centroid = F.normalize(feature_gpu[svf_gpu >= .40].mean(0), dim=0)
    return {
        "features": feature_gpu, "point_ids": point_ids, "azimuths": azimuths,
        "svf": svf, "gvi": gvi, "bvi": bvi, "good_mask": good_mask,
        "good_centroid": good_centroid, "poor_centroid": poor_centroid,
        "reference_count": int(feature_gpu.shape[0]),
    }


def load_runtime() -> tuple[Any, Any, Any, dict[str, Any], torch.device, torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).eval().to(device)
    dino = Dinov2Model.from_pretrained(DINO_DIR, local_files_only=True).eval().to(device)
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_DIR, local_files_only=True, dtype=torch.float16
    ).eval().to(device)
    labels = tuple(str(model.config.id2label[i]) for i in range(model.config.num_labels))
    if labels != LABELS:
        raise RuntimeError("Unexpected Mask2Former Cityscapes label mapping")
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
    bank = load_reference_bank(device)
    if any(parameter.device.type != "cuda" for runtime_model in (model, dino, depth_model) for parameter in runtime_model.parameters()):
        raise RuntimeError("All planner visual models must remain on CUDA")
    return model, dino, depth_model, bank, device, mean, std


def retrieve_self_supervised_references(
    rgb: np.ndarray,
    dino: Any,
    bank: dict[str, Any],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, Any]:
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    tensor = F.interpolate(tensor, (256, 256), mode="bicubic", align_corners=False, antialias=True)
    tensor = tensor[:, :, 16:240, 16:240] / 255.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        query = dino(pixel_values=(tensor - mean) / std).pooler_output.float()[0]
    query = F.normalize(query, dim=0)
    similarities = bank["features"] @ query
    overall_values, overall_indices = torch.topk(similarities, k=10)
    good_values, good_indices = torch.topk(torch.where(bank["good_mask"], similarities, torch.full_like(similarities, -2.0)), k=5)
    overall_indices_list = overall_indices.cpu().tolist()
    good_indices_list = good_indices.cpu().tolist()
    good_fraction = float(bank["good_mask"][overall_indices].float().mean().item())
    margin = float(torch.dot(query, bank["good_centroid"]).item() - torch.dot(query, bank["poor_centroid"]).item())
    references = []
    for value, index in zip(good_values.cpu().tolist(), good_indices_list):
        references.append({
            "point_id": bank["point_ids"][index],
            "reference_azimuth_deg": bank["azimuths"][index],
            "cosine_similarity": round(float(value), 4),
            "svf": round(float(bank["svf"][index]), 4),
            "gvi": round(float(bank["gvi"][index]), 4),
            "bvi": round(float(bank["bvi"][index]), 4),
        })
    nearest_index = overall_indices_list[0]
    return {
        "model": "facebook/dinov2-base",
        "learning_mode": "self_supervised_embedding_reference_retrieval",
        "reference_library_size": bank["reference_count"],
        "good_reference_definition": "point_level_SVF_below_0.30",
        "good_neighbor_fraction_top10": round(good_fraction, 4),
        "good_vs_poor_prototype_margin": round(margin, 4),
        "nearest_morphology_point_id": bank["point_ids"][nearest_index],
        "nearest_morphology_azimuth_deg": bank["azimuths"][nearest_index],
        "nearest_morphology_similarity": round(float(overall_values[0].item()), 4),
        "good_shade_references": references,
    }


def segment(path: Path, model: Any, device: torch.device, mean: torch.Tensor, std: torch.Tensor) -> tuple[np.ndarray, np.ndarray, bool]:
    with Image.open(path) as source:
        source.load(); rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    height, width = rgb.shape[:2]
    if min(height, width) < 128 or max(height, width) > 12000:
        raise ValueError("Image dimensions must be between 128 and 12000 pixels")
    panorama = width / height >= 3.2
    input_height = 384
    input_width = max(128, min(1536, int(round(width * input_height / height / 32)) * 32))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
    tensor = F.interpolate(tensor, (input_height, input_width), mode="bilinear", align_corners=False, antialias=True)
    pad = 96 if panorama else 0
    if pad:
        tensor = F.pad(tensor, (pad, pad, 0, 0), mode="circular")
    pixels = (tensor - mean) / std
    mask_valid = torch.ones((1, input_height, input_width + 2 * pad), dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model(pixel_values=pixels, pixel_mask=mask_valid)
    classes = output.class_queries_logits.float().softmax(-1)[..., :-1]
    masks = output.masks_queries_logits.float().sigmoid()
    logits = torch.einsum("bqc,bqhw->bchw", classes, masks)
    logits = F.interpolate(logits, (input_height, input_width + 2 * pad), mode="bilinear", align_corners=False)
    if pad:
        logits = logits[:, :, :, pad:pad + input_width]
    semantic = F.interpolate(logits, (height, width), mode="bilinear", align_corners=False).argmax(1)[0]
    mask = semantic.to(torch.uint8).cpu().numpy()
    return rgb, mask, panorama


def infer_metric_depth(
    rgb: np.ndarray, depth_model: Any, device: torch.device,
    mean: torch.Tensor, std: torch.Tensor,
) -> torch.Tensor:
    """Infer per-direction metric-depth proxy entirely on CUDA."""
    height, width = rgb.shape[:2]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
    tensor = F.interpolate(tensor, (518, 518), mode="bicubic", align_corners=False, antialias=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        predicted = depth_model(pixel_values=(tensor - mean) / std).predicted_depth
    return F.interpolate(
        predicted.unsqueeze(1), (height, width), mode="bicubic",
        align_corners=False, antialias=True,
    )[0, 0].float().clamp(.05, 80)


def _odd(value: int) -> int:
    return max(3, value if value % 2 else value + 1)


def _dilate(binary: torch.Tensor, kernel_y: int, kernel_x: int) -> torch.Tensor:
    return F.max_pool2d(
        binary.float(), (_odd(kernel_y), _odd(kernel_x)), stride=1,
        padding=(_odd(kernel_y) // 2, _odd(kernel_x) // 2),
    ) > .5


def _edge(binary: torch.Tensor) -> torch.Tensor:
    eroded = 1 - F.max_pool2d(1 - binary.float(), 5, stride=1, padding=2)
    return binary & (eroded < .5)


def _smooth_1d(values: torch.Tensor, window: int, circular: bool) -> torch.Tensor:
    """Smooth image columns, wrapping only for true 360-degree panoramas."""
    window = _odd(window)
    pad = window // 2
    mode = "circular" if circular else "replicate"
    padded = F.pad(values[None, None].float(), (pad, pad), mode=mode)
    return F.avg_pool1d(padded, window, stride=1)[0, 0]


def _road_direction_exclusion(road2d: torch.Tensor, panorama: bool) -> tuple[torch.Tensor, list[int]]:
    """Find road vanishing directions and return excluded panorama columns."""
    height, width = road2d.shape
    y = torch.arange(height, device=road2d.device).view(height, 1).expand(height, width)
    road_top = torch.where(road2d, y, height).amin(0)
    road_depth = (height - road_top).clamp(min=0).float()
    road_mass = road2d.float().sum(0)
    score = _smooth_1d(road_depth + .30 * road_mass, max(9, int(width * .035)), panorama)
    if float(score.max().item()) < height * .055:
        return torch.zeros(width, dtype=torch.bool, device=road2d.device), []
    first = int(score.argmax().item())
    centers = [first]
    if panorama:
        columns = torch.arange(width, device=road2d.device)
        distance = torch.minimum((columns - first).abs(), width - (columns - first).abs())
        opposite_arc = (distance >= int(width * .30)) & (distance <= int(width * .70))
        second_score = torch.where(opposite_arc, score, torch.zeros_like(score))
        if float(second_score.max().item()) >= max(height * .055, float(score.max().item()) * .18):
            centers.append(int(second_score.argmax().item()))
    columns = torch.arange(width, device=road2d.device)
    excluded = torch.zeros(width, dtype=torch.bool, device=road2d.device)
    radius = max(18, int(width * (.060 if panorama else .105)))
    for center in centers:
        distance = (columns - center).abs()
        if panorama:
            distance = torch.minimum(distance, width - distance)
        excluded |= distance <= radius
    return excluded, centers


def _road_scene_geometry(
    road2d: torch.Tensor, vehicle2d: torch.Tensor, panorama: bool,
    rgb: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Estimate curb convergence, vanishing point and road/ego-hood scale on CUDA.

    This is an image-space cue, not a metric road-width measurement.  The left
    and right boundaries of the visible road act as curb proxies.  Their
    convergence controls how strongly a perspective canopy envelope should
    taper toward the road end.
    """
    height, width = road2d.shape
    device = road2d.device
    rows = torch.arange(height, device=device, dtype=torch.float32)
    # Fill lane-marking holes, then follow only the road interval connected to
    # the primary road-direction anchor.  Global min/max would incorrectly join
    # separated carriageways and roadside fragments into a full-image width.
    road_clean = _dilate(road2d[None, None], 5, 9)[0, 0]
    _, direction_centres = _road_direction_exclusion(road_clean, panorama)
    direction_anchor = float(direction_centres[0]) if direction_centres else width / 2
    columns = torch.arange(width, device=device).view(1, width).expand(height, width)
    distance = torch.abs(columns.float() - direction_anchor)
    nearest = torch.where(road_clean, distance, torch.full_like(distance, width * 2)).argmin(1)
    has_road = road_clean.any(1)
    before = columns < nearest[:, None]
    after = columns > nearest[:, None]
    left_false = torch.where((~road_clean) & before, columns, torch.full_like(columns, -1)).amax(1)
    right_false = torch.where((~road_clean) & after, columns, torch.full_like(columns, width)).amin(1)
    left = (left_false + 1).float()
    right = (right_false - 1).float()
    visible_width = (right - left).clamp(min=0)
    fit_rows = (
        has_road
        & (rows >= height * .36)
        & (rows <= height * .86)
        & (visible_width >= width * .07)
    )

    def fit_line(values: torch.Tensor) -> tuple[float, float]:
        if int(fit_rows.sum().item()) < 8:
            return 0.0, float(values[has_road].median().item()) if bool(has_road.any()) else width / 2
        y = rows[fit_rows] / max(1.0, float(height - 1))
        x = values[fit_rows] / max(1.0, float(width - 1))
        weights = torch.linspace(.35, 1.0, y.numel(), device=device)
        design = torch.stack((y, torch.ones_like(y)), dim=1)
        weighted_design = design * weights[:, None]
        beta = torch.linalg.lstsq(weighted_design, x * weights).solution
        return float(beta[0].item() * (width - 1) / max(1.0, height - 1)), float(beta[1].item() * (width - 1))

    left_slope, left_intercept = fit_line(left)
    right_slope, right_intercept = fit_line(right)
    denominator = left_slope - right_slope
    if abs(denominator) >= 1e-4:
        vanish_y = (right_intercept - left_intercept) / denominator
        vanish_x = left_slope * vanish_y + left_intercept
    else:
        valid_center = (left + right) * .5
        vanish_x = float(valid_center[fit_rows].median().item()) if bool(fit_rows.any()) else width / 2
        vanish_y = height * .46
    if not (-width * .10 <= vanish_x <= width * 1.10 and height * .12 <= vanish_y <= height * .72):
        vanish_x = direction_anchor
        vanish_y = height * .43
    elif abs(vanish_x - direction_anchor) > width * .12:
        # The depth-score anchor is more stable than a curb intersection when
        # one curb is occluded or leaves the side-view frame.
        vanish_x = direction_anchor
    vanish_x = float(np.clip(vanish_x, 0, width - 1))
    vanish_y = float(np.clip(vanish_y, height * .20, height * .68))

    # Bright low-chroma lane markings provide an independent centre-line cue.
    # They do not replace the semantic road boundaries: their robust row-wise
    # median only adjusts the curb-derived vanishing point when both agree.
    lane_confidence = 0.0
    lane_vanish_x = vanish_x
    if rgb is not None and not panorama:
        image = rgb.to(device=device, dtype=torch.float32)
        luminance = image[..., 0] * .2126 + image[..., 1] * .7152 + image[..., 2] * .0722
        chroma = image.amax(-1) - image.amin(-1)
        lower_fit = fit_rows & (rows >= height * .43)
        road_luminance = luminance[road_clean & lower_fit[:, None]]
        if road_luminance.numel() >= 200:
            bright = torch.quantile(road_luminance, .79)
            road_center = (left + right) * .5
            central_band = torch.abs(columns.float() - road_center[:, None]) <= visible_width[:, None] * .34
            lane = road_clean & central_band & (luminance >= bright) & (chroma <= 62) & lower_fit[:, None]
            lane_count = lane.sum(1)
            lane_rows = lower_fit & (lane_count >= 1) & (lane_count <= torch.maximum(torch.full_like(lane_count, 3), (visible_width * .16).long()))
            if int(lane_rows.sum().item()) >= 8:
                lane_x = torch.where(lane, columns.float(), torch.full_like(columns.float(), float("nan"))).nanmedian(1).values
                y_norm = rows[lane_rows] / max(1.0, float(height - 1))
                x_norm = lane_x[lane_rows] / max(1.0, float(width - 1))
                design = torch.stack((y_norm, torch.ones_like(y_norm)), 1)
                beta = torch.linalg.lstsq(design, x_norm).solution
                lane_slope = float(beta[0].item() * (width - 1) / max(1.0, height - 1))
                lane_intercept = float(beta[1].item() * (width - 1))
                lane_vanish_x = lane_slope * vanish_y + lane_intercept
                agreement = max(0.0, 1.0 - abs(lane_vanish_x - vanish_x) / max(width * .18, 1.0))
                row_coverage = min(1.0, float(lane_rows.sum().item()) / max(height * .30, 1.0))
                lane_confidence = agreement * row_coverage
                if lane_confidence >= .18 and -width * .05 <= lane_vanish_x <= width * 1.05:
                    blend = min(.42, .18 + .35 * lane_confidence)
                    vanish_x = vanish_x * (1 - blend) + lane_vanish_x * blend
    vanish_x = float(np.clip(vanish_x, 0, width - 1))

    def width_at(row_ratio: float) -> float:
        row = int(np.clip(round(height * row_ratio), 0, height - 1))
        radius = max(2, int(height * .018))
        values = visible_width[max(0, row - radius):min(height, row + radius + 1)]
        values = values[values > width * .04]
        return float(values.median().item()) if values.numel() else 0.0

    road_width_near = width_at(.76)
    road_width_far = width_at(.48)
    convergence = max(0.0, min(1.0, (road_width_near - road_width_far) / max(road_width_near, 1.0)))
    curb_divergence = max(0.0, min(1.0, abs(right_slope - left_slope) * height / max(width, 1) * .9))
    centre_offset = abs(vanish_x - (width - 1) / 2) / max(width / 2, 1.0)
    alignment = max(0.0, 1.0 - centre_offset)
    curb_perspective_strength = 0.0 if panorama else max(0.0, min(1.0, max(convergence, curb_divergence) * (.10 + .90 * alignment ** 1.5)))

    # The hood is the same physical reference in all twelve views, whereas the
    # Cityscapes vehicle class often merges it with nearby cars.  Use the
    # calibrated capture-vehicle image width consistently across directions.
    hood_width = width * .30
    hood_source = "same_capture_vehicle_calibrated_width"
    hood_confidence = .55 * (.35 + .65 * alignment)
    width_ratio = road_width_near / max(hood_width, 1.0)
    road_surface_ratio = float(road2d.float().mean().item())
    fragmented_width_measurement = road_surface_ratio >= .28 and width_ratio < .80
    if fragmented_width_measurement:
        hood_confidence = min(hood_confidence, .12)
        width_evidence = "宽阔道路面；局部路沿宽度受遮挡或分割碎片影响"
    elif alignment < .28:
        width_evidence = "侧向视角，宽度证据较弱"
    elif width_ratio >= 1.70:
        width_evidence = "偏机动车空间"
    elif width_ratio <= 1.18:
        width_evidence = "偏步行骑行或窄街空间"
    else:
        width_evidence = "混合道路空间"
    return {
        "vanishing_point_x_ratio": round(vanish_x / width, 4),
        "vanishing_point_y_ratio": round(vanish_y / height, 4),
        "left_curb_slope_dx_dy": round(left_slope, 4),
        "right_curb_slope_dx_dy": round(right_slope, 4),
        "left_curb_intercept_x": round(left_intercept, 2),
        "right_curb_intercept_x": round(right_intercept, 2),
        "curb_perspective_strength": round(curb_perspective_strength, 4),
        "lane_marking_vanishing_x_ratio": round(lane_vanish_x / width, 4),
        "lane_marking_confidence": round(lane_confidence, 4),
        "road_end_exclusion_method": "semantic_curb_pair_plus_bright_lane_marking_convergence_protected_wedge",
        "road_width_px_at_0_76h": round(road_width_near, 2),
        "ego_hood_width_px": round(hood_width, 2),
        "road_to_ego_hood_width_ratio": round(width_ratio, 3),
        "ego_hood_width_source": hood_source,
        "width_evidence_confidence": hood_confidence,
        "road_width_space_evidence": width_evidence,
        "road_surface_ratio": round(road_surface_ratio, 4),
        "fragmented_local_width_measurement": fragmented_width_measurement,
    }


def _road_end_protected_wedge(
    road2d: torch.Tensor, road_geometry: dict[str, Any], panorama: bool,
) -> torch.Tensor:
    """Protect the forward road-end view, including sky above the vanishing point."""
    height, width = road2d.shape
    columns = torch.arange(width, device=road2d.device, dtype=torch.float32).view(1, width)
    rows = torch.arange(height, device=road2d.device, dtype=torch.float32).view(height, 1)
    if panorama:
        excluded, _ = _road_direction_exclusion(road2d, True)
        return excluded.view(1, width).expand(height, width)
    vanish_x = float(road_geometry.get("vanishing_point_x_ratio", .5)) * width
    vanish_y = float(road_geometry.get("vanishing_point_y_ratio", .43)) * height
    lane_confidence = float(road_geometry.get("lane_marking_confidence", 0.0))
    upper_progress = ((vanish_y - rows) / max(vanish_y, 1.0)).clamp(0, 1)
    lower_progress = ((rows - vanish_y) / max(height - vanish_y, 1.0)).clamp(0, 1)
    # The protected area widens upward because the user's forward sight cone
    # includes the sky above the road end; it widens downward following the
    # diverging carriageway. Lane agreement slightly strengthens this cone.
    half_width = width * (
        .135 + .055 * lane_confidence
        + .115 * upper_progress.pow(.78)
        + .235 * lower_progress.pow(.90)
    )
    wedge = torch.abs(columns - vanish_x) <= half_width
    # Include the visible carriageway plus a small curb margin below the
    # horizon so no proposal leaks through fragmented road pixels.
    road_guard = _dilate(road2d[None, None], max(5, int(height * .025)), max(7, int(width * .018)))[0, 0]
    road_guard &= rows >= vanish_y * .82
    return wedge | road_guard


def _depth_sidewalk_facility_geometry(
    semantic: torch.Tensor,
    depth: torch.Tensor,
    road_geometry: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a depth-ordered road-edge cross section on CUDA.

    It recovers narrow potential walking/cycling strips between the carriageway
    curb and side structures even when the semantic sidewalk class is sparse.
    It also separates planting support from attachable facility support.
    """
    height, width = semantic.shape
    device = semantic.device
    road, sidewalk = semantic == 0, semantic == 1
    building, wall, fence = semantic == 2, semantic == 3, semantic == 4
    pole = (semantic == 5) | (semantic == 6) | (semantic == 7)
    vegetation, terrain = semantic == 8, semantic == 9
    dynamic_obstacle = (semantic >= 11) & (semantic <= 18)

    smooth_depth = F.avg_pool2d(depth[None, None], 5, stride=1, padding=2)[0, 0]
    focal = width / (2 * math.tan(math.radians(45)))
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij",
    )
    points = torch.stack(((xx - (width - 1) / 2) / focal * smooth_depth,
                          (yy - (height - 1) / 2) / focal * smooth_depth,
                          smooth_depth), 0)
    dx = F.pad(points[:, :, 2:] - points[:, :, :-2], (1, 1, 0, 0), mode="replicate")
    dy = F.pad(points[:, 2:, :] - points[:, :-2, :], (0, 0, 1, 1), mode="replicate")
    normal = F.normalize(torch.cross(dx, dy, dim=0), dim=0, eps=1e-6)
    ground_score = normal[1].abs().clamp(0, 1)
    log_depth = smooth_depth.clamp(.05, 80).log()
    jump_x = F.pad((log_depth[:, 1:] - log_depth[:, :-1]).abs(), (0, 1), mode="replicate")
    jump_y = torch.cat(((log_depth[1:] - log_depth[:-1]).abs(), (log_depth[-1:] - log_depth[-2:-1]).abs()), 0)
    depth_edge = torch.maximum(jump_x, jump_y) > .16

    road_outer_band = _dilate(road[None, None], int(height * .045), int(width * .17))[0, 0] & ~road
    lower = yy >= height * .43
    side_structure = building | wall | fence | vegetation
    inferred_gap = road_outer_band & lower & ~side_structure & ~dynamic_obstacle & (ground_score >= .32) & ~depth_edge
    walk_corridor = (sidewalk | inferred_gap) & lower & ~dynamic_obstacle

    # Space immediately outside the recovered corridor distinguishes a possible
    # soil/tree strip from a narrow hard corridor that needs attached shade.
    corridor_neighbourhood = _dilate(walk_corridor[None, None], int(height * .035), int(width * .055))[0, 0]
    planting_support = corridor_neighbourhood & (terrain | vegetation) & ~road
    attachable = wall | fence | pole
    facility_anchor = corridor_neighbourhood & attachable

    corridor_pixels = int(walk_corridor.sum().item())
    explicit_pixels = int((sidewalk & lower).sum().item())
    inferred_pixels = int((walk_corridor & ~sidewalk).sum().item())
    planting_ratio = float(planting_support.float().mean().item())
    anchor_ratio = float(facility_anchor.float().mean().item())
    corridor_ratio = corridor_pixels / float(height * width)
    row_widths = walk_corridor.sum(1).float()
    valid_widths = row_widths[row_widths > 0]
    clearance_ratio = float(valid_widths.median().item() / width) if valid_widths.numel() else 0.0

    walk_depth_values = smooth_depth[walk_corridor]
    adjacent_shade = corridor_neighbourhood & (vegetation | building | wall)
    shade_depth_values = smooth_depth[adjacent_shade]
    walk_depth = float(walk_depth_values.median().item()) if walk_depth_values.numel() else float("nan")
    shade_depth = float(shade_depth_values.median().item()) if shade_depth_values.numel() else float("nan")
    distant_shade_only = bool(
        walk_depth_values.numel() and shade_depth_values.numel()
        and shade_depth > walk_depth * 1.22
        and corridor_ratio > .001
    )
    narrow_corridor = clearance_ratio < .075 and corridor_ratio > .001
    tree_space_limited = narrow_corridor and planting_ratio < .012
    facility_feasible = tree_space_limited and anchor_ratio >= .00008
    curb_edge = _edge(road[None, None])[0, 0]
    near_side_structure = _dilate(curb_edge[None, None], int(height * .10), int(width * .10))[0, 0] & side_structure
    near_side_shade = near_side_structure & (vegetation | building)
    vanish_column = int(round(float(road_geometry.get("vanishing_point_x_ratio", .5)) * (width - 1)))
    shade_columns = near_side_shade[int(height * .18):int(height * .72)].any(0)
    structure_columns = near_side_structure[int(height * .22):int(height * .82)].any(0)

    def side_continuity(values: torch.Tensor, start: int, end: int) -> float:
        if end - start < 8:
            return 0.0
        segment = values[start:end].float()
        smooth = F.max_pool1d(segment[None, None], 9, stride=1, padding=4)[0, 0]
        return float(smooth.mean().item())

    left_shade_continuity = side_continuity(shade_columns, 0, max(0, vanish_column))
    right_shade_continuity = side_continuity(shade_columns, min(width, vanish_column + 1), width)
    visible_sides = [value for value in (left_shade_continuity, right_shade_continuity) if value > .01]
    shade_continuity = float(np.mean(visible_sides)) if visible_sides else 0.0
    urban_interface_continuity = side_continuity(structure_columns, 0, max(0, vanish_column)) * .5 + side_continuity(structure_columns, min(width, vanish_column + 1), width) * .5
    near_interface_ratio = float(near_side_structure.float().mean().item())
    return {
        "model": "Depth Anything V2 Metric Outdoor Small",
        "device": str(depth.device),
        "cross_section_method": "curb_to_side_structure_depth_ordering",
        "walk_corridor_ratio": round(corridor_ratio, 5),
        "explicit_sidewalk_ratio": round(explicit_pixels / float(height * width), 5),
        "depth_recovered_walk_corridor_ratio": round(inferred_pixels / float(height * width), 5),
        "median_lateral_clearance_image_ratio": round(clearance_ratio, 4),
        "median_walk_corridor_depth_m_proxy": None if math.isnan(walk_depth) else round(walk_depth, 3),
        "median_adjacent_shade_depth_m_proxy": None if math.isnan(shade_depth) else round(shade_depth, 3),
        "distant_shade_does_not_cover_near_corridor": distant_shade_only,
        "planting_support_ratio": round(planting_ratio, 5),
        "attachable_facility_anchor_ratio": round(anchor_ratio, 5),
        "narrow_corridor": narrow_corridor,
        "tree_planting_space_limited": tree_space_limited,
        "attached_artificial_shade_feasible": facility_feasible,
        "near_road_building_tree_wall_ratio": round(near_interface_ratio, 5),
        "urban_side_interface_continuity": round(urban_interface_continuity, 4),
        "left_side_shade_continuity": round(left_shade_continuity, 4),
        "right_side_shade_continuity": round(right_shade_continuity, 4),
        "directional_side_shade_continuity": round(shade_continuity, 4),
        "directional_side_shade_gap_ratio": round(1.0 - shade_continuity, 4),
        "road_width_primary_evidence": road_geometry.get("road_width_space_evidence"),
    }, walk_corridor[None, None], facility_anchor[None, None], planting_support[None, None]


def _canopy_convex_envelope(
    raw_canopy: torch.Tensor,
    sky2d: torch.Tensor,
    existing_structure: torch.Tensor,
    road_excluded: torch.Tensor,
    road_centers: list[int],
    confidence_ratio: float,
    panorama: bool,
    road_geometry: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a continuous side-of-road canopy envelope on CUDA.

    The target top is a convex panorama-space arch. Existing trees or shade
    structures can lift that arch, but low objects cannot pull it downward.
    """
    height, width = sky2d.shape
    device = sky2d.device
    raw2d = raw_canopy[0, 0] & ~road_excluded.view(1, width)
    raw_support = raw2d.any(0).float()
    support_window = _odd(max(11, int(width * (.055 - .018 * confidence_ratio))))
    side_support = F.max_pool1d(raw_support[None, None], support_window, stride=1, padding=support_window // 2)[0, 0] > .5
    side_support &= ~road_excluded

    columns = torch.arange(width, device=device, dtype=torch.float32)
    if panorama and len(road_centers) >= 2:
        first, second = sorted(road_centers[:2])
        arc = second - first
        side_centers = [(first + arc / 2) % width, (second + (width - arc) / 2) % width]
    elif panorama and road_centers:
        side_centers = [float((road_centers[0] + width // 2) % width)]
    elif road_centers:
        center = float(road_centers[0])
        side_centers = [max(0.0, center - width * .28), min(width - 1.0, center + width * .28)]
    else:
        side_centers = [width * .25, width * .75]
    distance_to_side = torch.full((width,), float(width), device=device)
    for center in side_centers:
        distance = torch.abs(columns - center)
        if panorama:
            distance = torch.minimum(distance, width - distance)
        distance_to_side = torch.minimum(distance_to_side, distance)
    normalized = torch.clamp(distance_to_side / max(1.0, width * .25), 0, 1)
    if panorama:
        target_top = height * (.235 + .115 * normalized.square())
        target_bottom = torch.full_like(target_top, height * .56)
    else:
        # One-point perspective is strongest when the view faces the road end.
        # As the camera turns toward either road side, curb convergence weakens
        # and the target boundary approaches a flatter side elevation.
        vanish_x = float(road_geometry.get("vanishing_point_x_ratio", .5)) * width
        vanish_y = float(road_geometry.get("vanishing_point_y_ratio", .43)) * height
        strength = float(road_geometry.get("curb_perspective_strength", 0.0))
        side_distance = torch.abs(columns - vanish_x) / max(width * .50, 1.0)
        side_distance = side_distance.clamp(0, 1).pow(.82)
        perspective_top = vanish_y - (vanish_y - height * .17) * side_distance
        flat_top = torch.full_like(perspective_top, height * .245)
        target_top = flat_top * (1 - strength) + perspective_top * strength
        perspective_bottom = height * .50 + height * .10 * side_distance
        flat_bottom = torch.full_like(perspective_bottom, height * .56)
        target_bottom = flat_bottom * (1 - strength) + perspective_bottom * strength

    y = torch.arange(height, device=device).view(height, 1).expand(height, width)
    existing_top = torch.where(existing_structure, y, height).amin(0).float()
    existing_high = existing_top < target_top
    structural_reference = torch.where(existing_high, existing_top, target_top)
    structural_reference = _smooth_1d(structural_reference, max(9, int(width * .022)), panorama)
    envelope_top = torch.minimum(target_top, structural_reference).clamp(height * .10, height * .46)
    envelope_bottom = torch.maximum(envelope_top + height * (.09 + .05 * (1 - float(road_geometry.get("curb_perspective_strength", 0.0)))), target_bottom).clamp(max=height * .64)
    canopy = (
        side_support.view(1, width)
        & sky2d
        & (y >= envelope_top.view(1, width))
        & (y <= envelope_bottom.view(1, width))
    )
    return canopy[None, None], envelope_top, side_support


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def suggestion_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    active_space: bool,
    need: str,
    vegetation_ratio: float,
    confidence: int,
    road_geometry: dict[str, Any],
    depth_geometry: dict[str, Any],
    depth_walk_corridor: torch.Tensor,
    facility_anchor: torch.Tensor,
    planting_support: torch.Tensor,
) -> tuple[str, list[dict[str, Any]], dict[str, str], str, str]:
    """Build a CUDA-composited, translucent planning overlay without hiding the street view."""
    height, width = mask.shape
    semantic = torch.from_numpy(mask).to(device=device, dtype=torch.long)[None, None]
    sidewalk = semantic == 1
    road = semantic == 0
    building = semantic == 2
    wall = semantic == 3
    pole = semantic == 5
    vegetation = semantic == 8
    terrain = semantic == 9
    sky = semantic == 10

    rgb_gpu = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    luminance = rgb_gpu[..., 0] * .2126 + rgb_gpu[..., 1] * .7152 + rgb_gpu[..., 2] * .0722
    sidewalk_luminance = luminance[sidewalk[0, 0]]
    confidence_ratio = max(0.0, min(1.0, confidence / 100.0))
    exposure_quantile = .35 + .45 * confidence_ratio
    exposure_threshold = torch.clamp(
        torch.quantile(sidewalk_luminance, exposure_quantile) if sidewalk_luminance.numel() else torch.tensor(90. + 48. * confidence_ratio, device=device),
        min=90., max=150.,
    )
    # Street-view cameras sit near the carriageway centre, so a small Cityscapes
    # sidewalk mask is not allowed to erase the planning evidence.  Use the
    # visible sidewalk when present and a narrow road-edge/terrain corridor as a
    # conservative potential walking/cycling zone otherwise.
    proxy_scale = .052 - .022 * confidence_ratio
    road_edge = _dilate(_edge(road), int(height * (.050 - .016 * confidence_ratio)), int(height * proxy_scale))
    potential_walk = sidewalk | (road_edge & (road | terrain)) | depth_walk_corridor
    exposed_walk = potential_walk & (luminance[None, None] >= exposure_threshold)
    lower = torch.arange(height, device=device).view(1, 1, height, 1) >= int(height * .44)

    panorama = width / height >= 3.2
    road_excluded, road_centers = _road_direction_exclusion(road[0, 0], panorama)
    road_end_protected = _road_end_protected_wedge(road[0, 0], road_geometry, panorama)
    road_geometry["road_end_protected_ratio"] = round(float(road_end_protected.float().mean().item()), 4)
    road_geometry["road_end_protected_shape"] = "curb_and_lane_convergence_2d_wedge"
    shift = max(8, int(height * .17))
    raised_walk = torch.zeros_like(sidewalk)
    raised_walk[:, :, :-shift] = exposed_walk[:, :, shift:]
    raw_canopy = _dilate(raised_walk, int(height * (.145 - .070 * confidence_ratio)), int(height * (.220 - .110 * confidence_ratio))) & sky
    if bool(depth_geometry.get("tree_planting_space_limited", False)):
        raw_canopy &= _dilate(planting_support, int(height * .025), int(width * .025))
    existing_structure = (vegetation | building | wall)[0, 0]
    canopy_target, canopy_envelope_top, canopy_side_support = _canopy_convex_envelope(
        raw_canopy, sky[0, 0], existing_structure, road_excluded, road_centers,
        confidence_ratio, panorama, road_geometry
    )
    canopy_target &= ~road_end_protected[None, None]

    anchor_influence = _dilate(facility_anchor, int(height * .34), int(width * (.17 - .05 * confidence_ratio)))
    facility_review = anchor_influence & (sky | building | wall)
    facility_review &= torch.arange(height, device=device).view(1, 1, height, 1) >= int(height * .20)
    facility_review &= torch.arange(height, device=device).view(1, 1, height, 1) <= int(height * .58)
    facility_review &= ~road_excluded.view(1, 1, 1, width)
    facility_review &= ~road_end_protected[None, None]
    facility_review &= ~_dilate(canopy_target, int(height * .025), int(width * .008))

    mobility_target = (semantic == 11) | (semantic == 12) | (semantic == 17) | (semantic == 18)
    direct_anchor = (
        float(depth_geometry.get("walk_corridor_ratio", 0.0)) >= .001
        or float(mobility_target.float().mean().item()) >= .00005
    )
    structural_anchor = (
        float(road_geometry.get("road_surface_ratio", 0.0)) >= .035
        and (
            float(depth_geometry.get("planting_support_ratio", 0.0)) >= .0003
            or float(depth_geometry.get("urban_side_interface_continuity", 0.0)) >= .08
            or float(road_geometry.get("curb_perspective_strength", 0.0)) >= .08
        )
    )
    spatial_anchor_available = direct_anchor or structural_anchor
    publishable_active_space = active_space and spatial_anchor_available

    # The panorama is a decision canvas, not a semantic-debug view.  Only the
    # two possible intervention footprints are blended into it.  Existing
    # shade, walking proxies and ground classes remain internal evidence.
    layers: list[tuple[str, torch.Tensor, tuple[int, int, int], float, str]] = []
    if publishable_active_space and need != "通常不需要" and not bool(depth_geometry.get("tree_planting_space_limited", False)):
        layers.append(("canopy_target", canopy_target, (42, 157, 91), .24, "树冠目标覆盖区域"))
    if publishable_active_space and need not in {"通常不需要", "季节复核"} and bool(depth_geometry.get("attached_artificial_shade_feasible", False)):
        layers.append(("facility_review", facility_review, (224, 146, 48), .22, "人工遮阳棚候选覆盖区域"))

    image = rgb_gpu
    proposal_class = torch.zeros((height, width), device=device, dtype=torch.uint8)
    regions: list[dict[str, Any]] = []
    for key, selected, color, alpha, label in layers:
        selected2d = selected[0, 0]
        pixels = int(selected2d.sum().item())
        if pixels < max(20, int(mask.size * .00005)):
            continue
        color_tensor = torch.tensor(color, device=device, dtype=torch.float32)
        image[selected2d] = image[selected2d] * (1 - alpha) + color_tensor * alpha
        proposal_class[selected2d] = 1 if key == "canopy_target" else 2
        outline = _edge(selected)[0, 0]
        image[outline] = image[outline] * .18 + color_tensor * .82
        region = {
            "type": key,
            "label": label,
            "pixel_ratio": pixels / float(mask.size),
            "blend_alpha": alpha,
        }
        if key == "canopy_target":
            if panorama:
                excluded_directions = {"road_directions_excluded_deg": [round(360.0 * center / width, 1) for center in road_centers]}
            else:
                focal = width / 2.0
                excluded_directions = {
                    "road_end_local_yaw_deg": [round(math.degrees(math.atan((center - (width - 1) / 2) / focal)), 1) for center in road_centers]
                }
            region.update({
                "boundary_method": "curb_lane_convergence_protected_wedge_plus_side_perspective_envelope",
                "side_support_ratio": round(float(canopy_side_support.float().mean().item()), 4),
                "upper_boundary_is_target_height": True,
                "road_end_protected_ratio": round(float(road_end_protected.float().mean().item()), 4),
                "road_end_proposal_overlap_pixels": int((selected2d & road_end_protected).sum().item()),
                "road_geometry": road_geometry,
                **excluded_directions,
            })
        regions.append(region)

    # Mark only real attachable objects (wall/fence/pole/sign/light), while the
    # orange translucent polygon describes their possible shade coverage.
    if publishable_active_space and need not in {"通常不需要", "季节复核"} and bool(depth_geometry.get("attached_artificial_shade_feasible", False)):
        anchors = facility_anchor[0, 0]
        anchor_outline = _dilate(anchors[None, None], 5, 5)[0, 0] & ~anchors
        anchor_color = torch.tensor((126, 72, 171), device=device, dtype=torch.float32)
        image[anchor_outline] = image[anchor_outline] * .12 + anchor_color * .88
        regions.append({
            "type": "facility_anchor",
            "label": "可依附围墙/杆件/路牌",
            "pixel_ratio": round(float(anchors.float().mean().item()), 6),
            "coverage_relation": "anchor_object_to_translucent_facility_review_region",
        })

    rendered = Image.fromarray(image.clamp(0, 255).to(torch.uint8).cpu().numpy(), "RGB").convert("RGBA")
    annotation = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(annotation)
    guide_font = _font(max(11, int(height * .023)))
    # The upper outline of the blended canopy polygon is also the target growth
    # height curve, so no second line is drawn over the panorama.
    canopy_visible = any(key == "canopy_target" for key, *_ in layers)
    canopy2d = canopy_target[0, 0] if canopy_visible else torch.zeros_like(canopy_target[0, 0])
    # Vertical lines indicate intervention locations, not cardinal guides. Peaks
    # are selected only from the two visible intervention footprints.
    position_mask = canopy2d.clone()
    if any(key == "facility_review" for key, *_ in layers):
        position_mask |= facility_review[0, 0]
    column_score = position_mask.float().sum(0)
    smooth_width = max(9, int(width * .035))
    if smooth_width % 2 == 0:
        smooth_width += 1
    smoothed = F.avg_pool1d(column_score[None, None], smooth_width, stride=1, padding=smooth_width // 2)[0, 0]
    candidate_lines: list[dict[str, Any]] = []
    work = smoothed.clone()
    min_score = max(2.0, float(height) * (.018 + .075 * confidence_ratio))
    if not panorama and road_centers:
        center = road_centers[0]
        road_radius = max(18, int(width * .105))
        side_ranges = ((int(width * .04), max(int(width * .04), center - road_radius)), (min(int(width * .96), center + road_radius), int(width * .96)))
        for side, (start, end) in zip(("左侧", "右侧"), side_ranges):
            if end <= start:
                continue
            value, _ = torch.max(smoothed[start:end], 0)
            if float(value.item()) < min_score:
                continue
            side_weights = smoothed[start:end].clamp(min=0)
            local_columns = torch.arange(start, end, device=device, dtype=torch.float32)
            x = int(round(float((local_columns * side_weights).sum().item() / max(float(side_weights.sum().item()), 1e-6))))
            yaw = math.degrees(math.atan((x - (width - 1) / 2) / (width / 2)))
            candidate_lines.append({"x_ratio": x / width, "relative_yaw_deg": round(yaw, 1), "side": side, "display_label": f"{side}优化", "support_score": round(float(value.item()) / height, 4)})
    else:
        max_lines = max(1, min(4, int(round(5 - 3.2 * confidence_ratio))))
        for _ in range(max_lines):
            value, index = torch.max(work, 0)
            if float(value.item()) < min_score:
                break
            x = int(index.item())
            azimuth = 360.0 * x / width
            candidate_lines.append({"x_ratio": x / width, "azimuth_deg": round(azimuth, 1), "display_label": f"优化方位 {azimuth:.0f}°", "support_score": round(float(value.item()) / height, 4)})
            radius = max(24, int(width * (.10 + .06 * confidence_ratio)))
            work[max(0, x - radius):min(width, x + radius + 1)] = 0
    for item in candidate_lines:
        x = int(round(item["x_ratio"] * width))
        line_top = int(canopy_envelope_top[min(width - 1, x)].item()) if canopy_visible else int(height * .30)
        draw.line((x, line_top, x, int(height * .94)), fill=(224, 76, 61, 235), width=max(2, height // 150))
        label = str(item["display_label"])
        bbox = draw.textbbox((0, 0), label, font=guide_font); label_width = bbox[2] - bbox[0]
        lx = max(5, min(width - label_width - 15, x - label_width // 2))
        draw.rounded_rectangle((lx - 5, line_top + 7, lx + label_width + 7, line_top + 34), radius=4, fill=(117, 34, 28, 195))
        draw.text((lx, line_top + 9), label, font=guide_font, fill=(255, 255, 255, 245))
    rendered = Image.alpha_composite(rendered, annotation).convert("RGB")
    buffer = io.BytesIO(); rendered.save(buffer, "PNG", optimize=True)
    legend = {region["type"]: region["label"] for region in regions}
    if candidate_lines:
        legend["optimization_azimuth_lines"] = "优化方位竖线"
    regions.extend({"type": "optimization_azimuth_line", "label": item["display_label"], **item} for item in candidate_lines)
    mask_buffer = io.BytesIO()
    Image.fromarray(proposal_class.cpu().numpy(), "L").save(mask_buffer, "PNG", optimize=True)
    protected_buffer = io.BytesIO()
    Image.fromarray((road_end_protected.to(torch.uint8) * 255).cpu().numpy(), "L").save(
        protected_buffer, "PNG", optimize=True,
    )
    return (
        base64.b64encode(buffer.getvalue()).decode("ascii"), regions, legend,
        base64.b64encode(mask_buffer.getvalue()).decode("ascii"),
        base64.b64encode(protected_buffer.getvalue()).decode("ascii"),
    )


def space_type_evidence(
    ratios: dict[str, float], road_geometry: dict[str, Any], depth_geometry: dict[str, Any],
    road_context: dict[str, Any],
) -> tuple[str, dict[str, float], bool, bool, list[str]]:
    """Return transparent evidence scores, not a definitive land-use label."""
    road, sidewalk, building = ratios["road"], ratios["sidewalk"], ratios["building"]
    # Cityscapes does not separate electric bicycles from motorcycles; the
    # two-wheeler class is therefore an auxiliary micromobility-presence proxy.
    active_parts = {key: ratios[key] for key in ("person", "bicycle", "rider", "motorcycle")}
    dynamic_active_present = any(.00001 <= value <= (.03 if key == "person" else .02) for key, value in active_parts.items())
    active = min(.012, sum(active_parts.values()))
    width_ratio = float(road_geometry.get("road_to_ego_hood_width_ratio", 1.8))
    width_confidence = float(road_geometry.get("width_evidence_confidence", .45))
    road_surface_ratio = float(road_geometry.get("road_surface_ratio", road))
    fragmented_width = bool(road_geometry.get("fragmented_local_width_measurement", False))
    narrow_cue = max(0.0, min(1.0, (1.28 - width_ratio) / .30)) * width_confidence
    if fragmented_width:
        narrow_cue = 0.0
    wide_cue = max(0.0, min(1.0, (width_ratio - 1.35) / .50)) * width_confidence
    recovered_corridor = float(depth_geometry.get("walk_corridor_ratio", 0.0))
    urban_interface = float(depth_geometry.get("urban_side_interface_continuity", 0.0))
    near_interface = float(depth_geometry.get("near_road_building_tree_wall_ratio", 0.0))
    urban_edge_cue = max(0.0, min(1.0, .65 * urban_interface + 8.0 * near_interface))
    osm_class = str(road_context.get("fclass", "unknown")).lower()
    osm_motor_only = bool(road_context.get("motor_vehicle_only", False)) or osm_class in {"motorway", "motorway_link", "trunk", "trunk_link"}
    osm_active = bool(road_context.get("active_mobility_class", False))
    osm_relevant = bool(road_context.get("thermal_comfort_relevant", False))
    public_raw = .28 + 1.6 * sidewalk + 4.0 * active + 1.0 * building + 4.5 * recovered_corridor + 1.05 * narrow_cue + .65 * urban_edge_cue
    mixed_raw = .34 + .9 * road + 1.0 * sidewalk + 1.0 * building + 2.8 * recovered_corridor + .85 * urban_edge_cue + .20 * (1 - abs(narrow_cue - wide_cue))
    motor_raw = .18 + 1.25 * road + (0.18 if building < .03 else 0) + (0.12 if recovered_corridor < .008 else 0) + .48 * wide_cue + (.32 if road_surface_ratio >= .28 else 0)
    decision_basis = ["visual_road_width_and_surface"]
    if osm_motor_only:
        motor_raw += 2.4; decision_basis.append("osm_motor_vehicle_only_strong_gate")
    elif osm_active:
        public_raw += 1.8; decision_basis.append("osm_active_mobility_class")
    elif osm_relevant:
        mixed_raw += .75; public_raw += .20; decision_basis.append("osm_thermal_comfort_relevant")
    if dynamic_active_present:
        public_raw += 1.15; mixed_raw += .25; decision_basis.append("person_or_two_wheeler_detected")
    if urban_edge_cue >= .20:
        decision_basis.append("near_continuous_building_tree_street_edge")
    total = max(public_raw + mixed_raw + motor_raw, 1e-6)
    scores = {
        "公共步行骑行空间": public_raw / total,
        "城市混合道路空间": mixed_raw / total,
        "机动车专用或快速路": motor_raw / total,
    }
    label = max(scores, key=scores.get)
    if osm_motor_only:
        label = "机动车专用或快速路"
    elif osm_active:
        label = "公共步行骑行空间"
    elif dynamic_active_present and (recovered_corridor >= .001 or urban_edge_cue >= .12):
        label = "公共步行骑行空间"
    elif osm_relevant and label == "机动车专用或快速路":
        label = "城市混合道路空间"
    elif wide_cue >= .45 and urban_edge_cue >= .20 and label == "机动车专用或快速路":
        label = "城市混合道路空间"
    visual_motor_fallback = (
        str(road_context.get("source")) == "no_formal_osm_match"
        and label == "机动车专用或快速路" and wide_cue >= .60
        and road_surface_ratio >= .25 and urban_edge_cue < .10 and not dynamic_active_present
    )
    high_confidence_motor = osm_motor_only or visual_motor_fallback
    return label, {key: round(value, 4) for key, value in scores.items()}, high_confidence_motor, dynamic_active_present, decision_basis


def assess(
    rgb: np.ndarray, mask: np.ndarray, depth: torch.Tensor, panorama: bool, intent: str,
    device: torch.device, device_name: str, elapsed: float,
    capture_month: int | None = None,
    confidence: int = 65,
    self_supervised: dict[str, Any] | None = None,
    road_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = float(mask.size)
    ratios = {LABELS[index]: float((mask == index).sum() / total) for index in range(len(LABELS))}
    sky, vegetation, sidewalk, road = (ratios["sky"], ratios["vegetation"], ratios["sidewalk"], ratios["road"])
    active_mobility = ratios["bicycle"] + ratios["rider"] + ratios["motorcycle"]
    active_users = ratios["person"] + active_mobility
    confidence = max(0, min(100, int(confidence)))
    semantic_gpu = torch.from_numpy(mask).to(device=device, dtype=torch.long)
    road_geometry = _road_scene_geometry(
        semantic_gpu == 0,
        (semantic_gpu == 13) | (semantic_gpu == 14) | (semantic_gpu == 15) | (semantic_gpu == 17),
        panorama,
        torch.from_numpy(rgb).to(device=device),
    )
    depth_geometry, depth_walk_corridor, facility_anchor, planting_support = _depth_sidewalk_facility_geometry(
        semantic_gpu, depth, road_geometry
    )
    road_context = road_context or {
        "source": "no_formal_osm_match", "fclass": "unknown",
        "motor_vehicle_only": False, "thermal_comfort_relevant": False,
        "active_mobility_class": False,
    }
    space_type, space_scores, high_speed_road_suspected, dynamic_active_detected, space_decision_basis = space_type_evidence(
        ratios, road_geometry, depth_geometry, road_context
    )
    public_space_score = min(1.0, space_scores["公共步行骑行空间"] + .55 * space_scores["城市混合道路空间"])
    active_space = not high_speed_road_suspected
    leaf_off_risk = capture_month in {11, 12, 1, 2}
    march_phenology_uncertain = capture_month == 3
    # Shade need is assessed first. Space type is reported afterwards and only
    # a high-confidence motor-only scene can exclude a point.
    good_neighbor_fraction = float((self_supervised or {}).get("good_neighbor_fraction_top10", .5))
    adaptive_directional_limit = .30
    if not panorama and good_neighbor_fraction >= .70:
        adaptive_directional_limit = .34
    elif not panorama and good_neighbor_fraction <= .20:
        adaptive_directional_limit = .27
    good_limit = .15 if panorama else adaptive_directional_limit
    local_limit = .25 if panorama else .40
    medium_limit = .35 if panorama else .50
    side_shade_continuity = float(depth_geometry.get("directional_side_shade_continuity", 0.0))
    directional_shade_gap = (not panorama) and sky > adaptive_directional_limit + .05 and side_shade_continuity < .42
    if leaf_off_risk:
        need = "季节复核"
        action = "当前月份存在落叶或早春树种差异风险；低叶量不能直接解释为缺树，先识别连续树列并在展叶季复核。"
        priority = "现有树列与物候复核"
        decision_stage = "stage_2_phenology_gate"
        decision_status = "暂停优化：落叶季或树种差异复核"
        optimization_eligible = False
    elif sky < good_limit:
        need = "通常不需要"
        action = f"天空视域较低，现有遮挡已较充分；方位图以0.30为基础参考，并结合DINOv2相似案例将本图门槛调整为{adaptive_directional_limit:.2f}。" if not panorama else "SVF较低，现有天空遮挡已较充分；可能位于桥下、建筑遮挡或连续树冠下，优先保持现状。"
        priority = "保持现状"
        decision_stage = "stage_3_svf_need_gate"
        decision_status = "提前结束：SVF较低，无需新增遮荫"
        optimization_eligible = False
    elif sky < local_limit:
        need = "较低"
        action = "SVF已经较低；只检查树冠是否连续覆盖人行道，原则上不做大规模新增。"
        priority = "维护与局部补植"
        decision_stage = "stage_4_optimization"
        decision_status = "进入优化阶段：仅复核局部缺口"
        optimization_eligible = True
    elif sky < medium_limit:
        need = "中等"
        action = "SVF中等；沿可步行空间补齐连续高大乔木，重点连接现有树冠缺口。"
        priority = "树木连续性优化"
        decision_stage = "stage_4_optimization"
        decision_status = "进入优化阶段：树冠连续性优化"
        optimization_eligible = True
    else:
        need = "较高"
        action = "SVF较高；优先评估沿人行道连续种植高大乔木，仅在不适宜种树的位置补充依附既有设施的人工遮阳。"
        priority = "连续乔木优先"
        decision_stage = "stage_4_optimization"
        decision_status = "进入优化阶段：连续乔木优先"
        optimization_eligible = True
    if high_speed_road_suspected:
        action = "遮荫现状已完成评估；该场景更可能属于机动车专用或快速路，规划师可据道路属性决定是否排除。"
        priority = "机动车专用空间候选"
        decision_stage = "stage_3_space_typology"
        decision_status = "空间类型后判定：较可能为机动车专用空间"
        optimization_eligible = False
    elif space_type != "公共步行骑行空间":
        decision_status += "；空间类型为概率性判断，不因人行道像素少而终止"
    if optimization_eligible and not high_speed_road_suspected and bool(depth_geometry.get("distant_shade_does_not_cover_near_corridor", False)):
        action += " 深度横断面显示现有建筑或树木主要位于更远处，不能替代近处窄慢行带上方的连续遮荫，因此仍保留优化。"
    if optimization_eligible and not high_speed_road_suspected and bool(depth_geometry.get("attached_artificial_shade_feasible", False)):
        action += " 当前慢行带较窄且缺少树木种植空间，已标记邻近围墙、杆件或路牌作为人工遮阳附属候选，并给出可能覆盖区域。"
        priority = "既有设施附属人工遮阳补充"
    direct_spatial_anchor = float(depth_geometry.get("walk_corridor_ratio", 0.0)) >= .001 or dynamic_active_detected
    structural_spatial_anchor = (
        float(road_geometry.get("road_surface_ratio", 0.0)) >= .035
        and (
            float(depth_geometry.get("planting_support_ratio", 0.0)) >= .0003
            or float(depth_geometry.get("urban_side_interface_continuity", 0.0)) >= .08
            or float(road_geometry.get("curb_perspective_strength", 0.0)) >= .08
        )
    )
    spatial_anchor_available = direct_spatial_anchor or structural_spatial_anchor
    spatial_anchor_mode = (
        "direct_walk_corridor_or_dynamic_mobility" if direct_spatial_anchor
        else "inferred_roadside_structure" if structural_spatial_anchor
        else "planner_assisted_manual_region_required"
    )
    if not high_speed_road_suspected and optimization_eligible and not direct_spatial_anchor:
        if structural_spatial_anchor:
            decision_status += "；采用道路边界与路侧界面保守定位"
            action += " 当前方位的人行道像素较弱，但道路边界、路侧建筑/树木连续性或种植支撑形成了保守空间锚点。"
        else:
            decision_status += "；自动锚点不足，可由规划师圈定候选区域"
            action += " 当前方位仍有遮荫优化需求，但自动空间锚点不足；允许规划师在道路两侧手动圈定区域，后台继续强制排除道路尽头与交通对象。"
    if self_supervised:
        good_refs = self_supervised.get("good_shade_references", [])
        median_ref_gvi = float(np.median([item["gvi"] for item in good_refs])) if good_refs else vegetation
        median_ref_bvi = float(np.median([item["bvi"] for item in good_refs])) if good_refs else ratios["building"]
        if median_ref_gvi >= vegetation + .05 and median_ref_bvi >= ratios["building"] + .05:
            learned_reason = "相似优质街景同时具有更连续树冠和更强建筑边界遮挡"
        elif median_ref_gvi >= vegetation + .05:
            learned_reason = "相似优质街景的主要优势是更高且更连续的树冠覆盖"
        elif median_ref_bvi >= ratios["building"] + .05:
            learned_reason = "相似优质街景更多依赖建筑或构筑物边界遮挡"
        else:
            learned_reason = "相似优质街景与当前形态接近，重点应检查遮荫连续性而非盲目新增"
        self_supervised = {**self_supervised, "learned_shade_advantage": learned_reason}
    overlay_png, suggestion_regions, overlay_legend, proposal_mask_png, road_end_protected_mask_png = suggestion_overlay(
        rgb, mask, device, active_space, need, vegetation, confidence, road_geometry,
        depth_geometry, depth_walk_corridor, facility_anchor, planting_support,
    )
    vegetation_buffer = io.BytesIO()
    Image.fromarray(((mask == 8) * 255).astype(np.uint8), "L").save(vegetation_buffer, "PNG", optimize=True)
    semantic_buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), "L").save(semantic_buffer, "PNG", optimize=True)
    return {
        "analysis_type": "uploaded_image_preliminary_gpu_semantic_assessment",
        "model": "facebook/mask2former-swin-large-cityscapes-semantic",
        "device": device_name,
        "panorama_detected": panorama,
        "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
        "ratios": {key: ratios[key] for key in ("sky", "vegetation", "sidewalk", "road", "building", "wall", "fence", "pole", "traffic sign", "person", "rider", "bicycle", "motorcycle")},
        "svf": sky,
        "sky_view_metric_name": "panorama_svf_proxy" if panorama else "directional_sky_view_ratio",
        "directional_good_shade_reference_threshold": None if panorama else .30,
        "directional_adaptive_threshold": None if panorama else adaptive_directional_limit,
        "directional_threshold_is_reference_not_absolute": not panorama,
        "directional_side_shade_continuity": round(side_shade_continuity, 4),
        "directional_shade_gap_detected": directional_shade_gap,
        "capture_month": capture_month,
        "leaf_off_risk": leaf_off_risk,
        "march_phenology_uncertain": march_phenology_uncertain,
        "phenology_note": "冬季落叶或混合树种会使GVI低估；当前结果不据低叶量自动新增树木。" if leaf_off_risk else ("3月保留物候不确定提示，但不作为冬季硬排除条件。" if march_phenology_uncertain else "未触发月份物候门槛。"),
        "public_walk_cycle_likelihood": "较低" if high_speed_road_suspected else ("较高" if space_type == "公共步行骑行空间" else "可能"),
        "space_type_assessment": space_type,
        "space_type_evidence_scores": space_scores,
        "space_type_decision_basis": space_decision_basis,
        "osm_road_context": road_context,
        "dynamic_active_mobility_detected": dynamic_active_detected,
        "road_geometry_evidence": road_geometry,
        "depth_cross_section_evidence": depth_geometry,
        "public_space_score": round(public_space_score, 4),
        "space_type_is_probabilistic": True,
        "planning_confidence": confidence,
        "confidence_effect": "higher_values_keep_fewer_stronger_intervention_regions",
        "high_speed_road_suspected": high_speed_road_suspected,
        "shade_optimization_need": need,
        "planning_priority": priority,
        "decision_stage": decision_stage,
        "decision_status": decision_status,
        "optimization_eligible": optimization_eligible,
        "directional_spatial_anchor_available": spatial_anchor_available,
        "directional_spatial_anchor_mode": spatial_anchor_mode,
        "generation_candidate_available": bool(optimization_eligible and not high_speed_road_suspected and not leaf_off_risk),
        "automatic_generation_candidate_available": bool(optimization_eligible and spatial_anchor_available and not high_speed_road_suspected and not leaf_off_risk),
        "recommendation": action,
        "planner_intent": intent,
        "overlay_type": "translucent_planning_suggestion_blend",
        "overlay_legend": overlay_legend,
        "suggestion_regions": suggestion_regions,
        "overlay_png_base64": overlay_png,
        "proposal_mask_png_base64": proposal_mask_png,
        "vegetation_mask_png_base64": base64.b64encode(vegetation_buffer.getvalue()).decode("ascii"),
        "semantic_label_png_base64": base64.b64encode(semantic_buffer.getvalue()).decode("ascii"),
        "road_end_protected_mask_png_base64": road_end_protected_mask_png,
        "self_supervised_reference": self_supervised,
        "inference_seconds": elapsed,
        "quantified_thermal_benefit_available": False,
        "limitation": (
            "方向图天空占比不是完整半球SVF；缺少坐标或气象时，"
            "服务层仅提供南京相似城市形态与典型晴热天气下的迁移参考，"
            "不等同于上传地点实测或坐标级热环境模拟。"
        ),
    }


def fuse_directional_results(
    panorama_path: Path,
    directional_results: list[dict[str, Any]],
    direction_records: list[dict[str, Any]],
    device: torch.device,
) -> str:
    """Reproject source views and their plans through one shared seam model.

    The old implementation drew smooth proposal masks over an already jagged
    panorama, so image and proposal geometry disagreed at direction seams.
    Here both RGB and proposal classes use the same known-heading projection,
    centre-dominant feather weights and final sub-pixel smoothing.
    """
    with Image.open(panorama_path) as source:
        source.load()
        panorama = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    height, width = panorama.shape[:2]
    azimuth = torch.arange(width, device=device, dtype=torch.float32) * (2 * math.pi / width)
    vertical_half = math.atan(math.tan(math.radians(45)) * math.cos(math.radians(15))) * (1 - 1 / height)
    elevation = vertical_half - torch.arange(height, device=device, dtype=torch.float32) * (2 * vertical_half / height)
    delta_template = elevation.unsqueeze(1).expand(-1, width)
    canopy_score = torch.zeros((height, width), device=device)
    facility_score = torch.zeros_like(canopy_score)
    coverage = torch.zeros_like(canopy_score)
    rgb_sum = torch.zeros((3, height, width), device=device)
    rgb_coverage = torch.zeros((height, width), device=device)
    focal = 512 / (2 * math.tan(math.radians(45)))
    paths_by_heading = {int(item["heading"]): Path(item["image_path"]).resolve() for item in direction_records}
    for result in tqdm(directional_results, total=len(directional_results), desc="Fusing direction plans", unit="view"):
        payload = base64.b64decode(result["proposal_mask_png_base64"])
        with Image.open(io.BytesIO(payload)) as mask_image:
            mask_array = np.asarray(mask_image.convert("L"), dtype=np.uint8).copy()
        mask_tensor = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32)[None, None]
        image_path = paths_by_heading[int(result["heading"])]
        with Image.open(image_path) as source_view:
            source_view.load()
            view_array = np.asarray(source_view.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS), dtype=np.uint8).copy()
        view_tensor = torch.from_numpy(view_array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        heading = math.radians(float(result["heading"]))
        delta = torch.remainder(azimuth - heading + math.pi, 2 * math.pi) - math.pi
        delta_grid = delta.unsqueeze(0).expand(height, -1)
        map_x = 255.5 + focal * torch.tan(delta_grid)
        map_y = 255.5 - focal * torch.tan(delta_template) / torch.cos(delta_grid)
        valid = (
            (torch.abs(delta_grid) < math.radians(45))
            & (map_x >= -.5) & (map_x <= 511.5)
            & (map_y >= -.5) & (map_y <= 511.5)
        )
        grid = torch.stack((map_x.clamp(0, 511) * 2 / 511 - 1, map_y.clamp(0, 511) * 2 / 511 - 1), dim=-1)[None]
        sampled_canopy = F.grid_sample((mask_tensor == 1).float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        sampled_facility = F.grid_sample((mask_tensor == 2).float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        sampled_rgb = F.grid_sample(view_tensor, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
        nx = (map_x + .5 - 256) / 256
        ny = (map_y + .5 - 256) / 256
        # A high horizontal exponent strongly prefers the centre of each view,
        # while a narrow feather band removes saw-tooth direction boundaries.
        weight = torch.cos(nx.abs().clamp(0, 1) * math.pi / 2).pow(10) * torch.cos(ny.abs().clamp(0, 1) * math.pi / 2).pow(2) * valid
        canopy_score += sampled_canopy * weight
        facility_score += sampled_facility * weight
        coverage += weight
        rgb_sum += sampled_rgb * weight.unsqueeze(0)
        rgb_coverage += weight
    original = torch.from_numpy(panorama).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    reprojected = rgb_sum / rgb_coverage.clamp(min=1e-5).unsqueeze(0)
    valid_rgb = rgb_coverage > .015
    image_chw = torch.where(valid_rgb.unsqueeze(0), reprojected, original)
    # Shared low-amplitude seam smoothing removes one-pixel stair steps without
    # blurring the whole navigation scene.
    seam_columns = torch.zeros(width, dtype=torch.bool, device=device)
    for heading in paths_by_heading:
        seam = int(round(((heading + 15) % 360) / 360 * width)) % width
        seam_columns[(torch.arange(width, device=device) - seam).abs() <= 2] = True
    smooth_rgb = F.avg_pool2d(F.pad(image_chw[None], (2, 2, 1, 1), mode="reflect"), (3, 5), stride=1)[0]
    image_chw[:, :, seam_columns] = smooth_rgb[:, :, seam_columns]
    image = image_chw.permute(1, 2, 0)

    canopy_probability = canopy_score / coverage.clamp(min=1e-5)
    facility_probability = facility_score / coverage.clamp(min=1e-5)
    canopy_probability = F.avg_pool2d(canopy_probability[None, None], (3, 5), stride=1, padding=(1, 2))[0, 0]
    facility_probability = F.avg_pool2d(facility_probability[None, None], (3, 5), stride=1, padding=(1, 2))[0, 0]
    canopy = (canopy_probability >= facility_probability) & (canopy_probability > .22) & valid_rgb
    facility = (facility_probability > canopy_probability) & (facility_probability > .22) & valid_rgb
    for selected, color, alpha in ((canopy, (42, 157, 91), .24), (facility, (224, 146, 48), .22)):
        color_tensor = torch.tensor(color, device=device, dtype=torch.float32)
        image[selected] = image[selected] * (1 - alpha) + color_tensor * alpha
        outline = _edge(selected[None, None])[0, 0]
        image[outline] = image[outline] * .18 + color_tensor * .82
    buffer = io.BytesIO()
    Image.fromarray(image.clamp(0, 255).to(torch.uint8).cpu().numpy(), "RGB").save(buffer, "PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8"); sys.stdout.reconfigure(encoding="utf-8")
    model, dino, depth_model, reference_bank, device, mean, std = load_runtime()
    print(json.dumps({"ready": True, "gpu": torch.cuda.get_device_name(0), "dinov2_reference_count": reference_bank["reference_count"]}), flush=True)
    cached_path: Path | None = None
    cached_rgb: np.ndarray | None = None
    cached_mask: np.ndarray | None = None
    cached_depth: torch.Tensor | None = None
    cached_panorama = False
    cached_self_supervised: dict[str, Any] | None = None
    for line in sys.stdin:
        request_id = ""
        try:
            message = json.loads(line); request_id = str(message["id"])
            if message["operation"] == "shutdown":
                print(json.dumps({"id": request_id, "ok": True, "result": {}}), flush=True); return 0
            if message["operation"] == "analyze_directions":
                confidence = max(0, min(100, int(message.get("confidence", 65))))
                capture_month_raw = message.get("capture_month")
                capture_month = int(capture_month_raw) if capture_month_raw not in (None, "") else None
                intent = str(message.get("planner_intent", ""))
                road_context = dict(message.get("road_context") or {})
                direction_results: list[dict[str, Any]] = []
                for item in tqdm(message["directions"], total=len(message["directions"]), desc="Analyzing direction views", unit="view"):
                    started = time.perf_counter()
                    path = Path(item["image_path"]).resolve()
                    rgb, mask, panorama = segment(path, model, device, mean, std)
                    depth = infer_metric_depth(rgb, depth_model, device, mean, std)
                    self_supervised = retrieve_self_supervised_references(rgb, dino, reference_bank, device, mean, std)
                    result = assess(rgb, mask, depth, panorama, intent, device, torch.cuda.get_device_name(0), time.perf_counter() - started, capture_month, confidence, self_supervised, road_context)
                    result.update({"heading": int(item["heading"]), "source_filename": path.name})
                    direction_results.append(result)
                fused = fuse_directional_results(
                    Path(message["panorama_path"]).resolve(), direction_results,
                    message["directions"], device,
                )
                need_counts: dict[str, int] = {}
                for result in direction_results:
                    need = str(result["shade_optimization_need"])
                    need_counts[need] = need_counts.get(need, 0) + 1
                gap_directions = [
                    int(result["heading"]) for result in direction_results
                    if bool(result.get("directional_shade_gap_detected"))
                ]
                continuity_values = [float(result.get("directional_side_shade_continuity", 0.0)) for result in direction_results]
                batch_result = {
                    "analysis_type": "twelve_direction_independent_gpu_analysis_with_panorama_decision_fusion",
                    "direction_count": len(direction_results),
                    "directions": direction_results,
                    "fused_panorama_png_base64": fused,
                    "direction_need_counts": need_counts,
                    "side_shade_continuity": {
                        "mean": round(float(np.mean(continuity_values)), 4),
                        "minimum": round(float(np.min(continuity_values)), 4),
                        "local_gap_direction_count": len(gap_directions),
                        "local_gap_headings": gap_directions,
                        "assessment": "存在方向性遮荫缺口" if gap_directions else "道路两侧遮荫较连续",
                        "method": "twelve_direction_near_road_building_tree_continuity_not_mean_svf_only",
                    },
                    "generation_basis": "independent_perspective_views_not_panorama_inference",
                    "external_request_made": False,
                }
                print(json.dumps({"id": request_id, "ok": True, "result": batch_result}, ensure_ascii=False), flush=True)
                continue
            started = time.perf_counter()
            image_path = Path(message["image_path"]).resolve()
            cache_reused = image_path == cached_path and cached_rgb is not None and cached_mask is not None and cached_depth is not None
            if cache_reused:
                rgb, mask, depth, panorama = cached_rgb, cached_mask, cached_depth, cached_panorama
                self_supervised = cached_self_supervised
            else:
                rgb, mask, panorama = segment(image_path, model, device, mean, std)
                depth = infer_metric_depth(rgb, depth_model, device, mean, std)
                self_supervised = retrieve_self_supervised_references(rgb, dino, reference_bank, device, mean, std)
                cached_path, cached_rgb, cached_mask, cached_depth, cached_panorama = image_path, rgb, mask, depth, panorama
                cached_self_supervised = self_supervised
            capture_month = message.get("capture_month")
            capture_month = int(capture_month) if capture_month not in (None, "") else None
            confidence = max(0, min(100, int(message.get("confidence", 65))))
            result = assess(
                rgb, mask, depth, panorama, str(message.get("planner_intent", "")),
                device, torch.cuda.get_device_name(0), time.perf_counter() - started,
                capture_month, confidence, self_supervised, dict(message.get("road_context") or {}),
            )
            result["segmentation_cache_reused"] = cache_reused
            print(json.dumps({"id": request_id, "ok": True, "result": result}, ensure_ascii=False), flush=True)
        except Exception as error:
            print(json.dumps({"id": request_id, "ok": False, "error_message": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
