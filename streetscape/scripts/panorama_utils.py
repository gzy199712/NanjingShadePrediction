"""Shared CUDA panorama projection helpers."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
import torch


HEADINGS = range(0, 360, 30)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Panorama generation requires CUDA")
    return torch.device("cuda:0")


def _camera_to_world(heading_deg: float) -> np.ndarray:
    heading = math.radians(heading_deg)
    return np.array(
        [
            [math.cos(heading), 0.0, math.sin(heading)],
            [-math.sin(heading), 0.0, math.cos(heading)],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def projection_grid(
    rotation_correction_deg: np.ndarray,
    heading: int,
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width, height = config["panorama"]["width"], config["panorama"]["height"]
    source_width = config["camera"]["source_width"]
    source_height = config["camera"]["source_height"]
    field_of_view = math.radians(config["camera"]["fov_degrees"])
    azimuth = torch.arange(width, device=device) * (2 * math.pi / width)
    limit = math.atan(math.tan(field_of_view / 2) * math.cos(math.radians(15))) * (1 - 1 / height)
    elevation = limit - torch.arange(height, device=device) * (2 * limit / height)
    azimuth, elevation = torch.meshgrid(azimuth, elevation, indexing="xy")
    world = torch.stack(
        (
            torch.sin(azimuth) * torch.cos(elevation),
            torch.cos(azimuth) * torch.cos(elevation),
            torch.sin(elevation),
        ),
        dim=-1,
    )
    correction = Rotation.from_rotvec(np.radians(rotation_correction_deg)).as_matrix()
    inverse = torch.as_tensor(
        (correction @ _camera_to_world(heading)).T,
        device=device,
        dtype=torch.float32,
    )
    camera = torch.einsum("ij,hwj->hwi", inverse, world)
    focal = source_width / (2 * math.tan(field_of_view / 2))
    depth = camera[..., 2]
    x = (source_width - 1) / 2 + focal * camera[..., 0] / depth.clamp_min(1e-6)
    y = (source_height - 1) / 2 + focal * camera[..., 1] / depth.clamp_min(1e-6)
    valid = (depth > 0) & (x >= -0.5) & (x <= source_width - 0.5) & (y >= -0.5) & (y <= source_height - 0.5)
    normalized_x = (x + 0.5 - source_width / 2) / (source_width / 2)
    normalized_y = (y + 0.5 - source_height / 2) / (source_height / 2)
    weight = torch.cos(normalized_x.abs().clamp(0, 1) * math.pi / 2) * torch.cos(normalized_y.abs().clamp(0, 1) * math.pi / 2) * valid
    grid = torch.stack(
        (
            x.clamp(0, source_width - 1) * 2 / (source_width - 1) - 1,
            y.clamp(0, source_height - 1) * 2 / (source_height - 1) - 1,
        ),
        dim=-1,
    )
    return grid, valid, weight


def load_point_images(group: pd.DataFrame, source: Path, device: torch.device) -> torch.Tensor:
    indexed = group.set_index("heading")
    images = []
    for heading in HEADINGS:
        path = source / Path(str(indexed.loc[heading, "filepath"])).name
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)


def save_rgb(image: torch.Tensor, path: Path) -> None:
    array = image.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path, compress_level=3)
