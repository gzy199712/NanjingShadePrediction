"""stage_44: CUDA panorama-to-perspective windows with explicit spherical correspondence."""

from __future__ import annotations

import argparse
import csv
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
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/panorama_perspective_windows.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_44"


def resolve(value: str) -> Path:
    return ROOT / value


def make_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_44")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_44.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def camera_to_world(heading: float, device: torch.device) -> torch.Tensor:
    angle = math.radians(heading)
    return torch.tensor(
        [[math.cos(angle), 0.0, math.sin(angle)], [-math.sin(angle), 0.0, math.cos(angle)], [0.0, -1.0, 0.0]],
        device=device, dtype=torch.float32,
    )


def correspondence_grid(cfg: dict[str, Any], heading: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = int(cfg["projection"]["window_size"])
    pano_w = int(cfg["projection"]["panorama_width"])
    pano_h = int(cfg["projection"]["panorama_height"])
    fov = math.radians(float(cfg["projection"]["horizontal_fov_degrees"]))
    focal = size / (2 * math.tan(fov / 2))
    coordinate = torch.arange(size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    ray = torch.stack(((xx - (size - 1) / 2) / focal, (yy - (size - 1) / 2) / focal, torch.ones_like(xx)), dim=-1)
    ray = F.normalize(ray, dim=-1)
    world = torch.einsum("ij,hwj->hwi", camera_to_world(heading, device), ray)
    azimuth = torch.atan2(world[..., 0], world[..., 1]).remainder(2 * math.pi)
    elevation = torch.atan2(world[..., 2], torch.linalg.vector_norm(world[..., :2], dim=-1))
    half_extent = math.radians(float(cfg["projection"]["panorama_vertical_half_extent_degrees"]))
    pano_x = azimuth / (2 * math.pi) * pano_w
    pano_y = (half_extent - elevation) / (2 * half_extent) * pano_h
    wrapped_x = pano_x + 1.0
    grid_x = wrapped_x * 2 / (pano_w + 1) - 1
    grid_y = pano_y.clamp(0, pano_h - 1) * 2 / (pano_h - 1) - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)[None]
    valid = (pano_y >= 0) & (pano_y <= pano_h - 1)
    return grid, torch.stack((pano_x.remainder(pano_w), pano_y), dim=0), valid


def reconstruct(windows: list[torch.Tensor], maps: list[torch.Tensor], valid_masks: list[torch.Tensor], height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = torch.zeros((3, height * width), device=windows[0].device)
    denominator = torch.zeros(height * width, device=windows[0].device)
    for window, mapping, valid in zip(windows, maps, valid_masks):
        x0 = mapping[0].floor()
        y0 = mapping[1].floor()
        fx = mapping[0] - x0
        fy = mapping[1] - y0
        size = window.shape[-1]
        coord = torch.arange(size, device=window.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        radius = torch.maximum((xx - (size - 1) / 2).abs(), (yy - (size - 1) / 2).abs()) / (size / 2)
        base_weight = torch.cos(radius.clamp(0, 1) * math.pi / 2).square() * valid
        for dx, dy, interpolation in (
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            x = (x0.long() + dx).remainder(width)
            y = (y0.long() + dy).clamp(0, height - 1)
            index = (y * width + x).flatten()
            weight = base_weight * interpolation
            numerator.scatter_add_(1, index[None].expand(3, -1), (window * weight).reshape(3, -1))
            denominator.scatter_add_(0, index, weight.flatten())
    reconstruction = (numerator / denominator.clamp_min(1e-8)[None]).reshape(3, height, width)
    return reconstruction, denominator.reshape(height, width)


def write_report(summary: dict[str, Any], qc: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# stage_44 Panorama perspective windows

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Points: {summary['point_count']}
- Windows: {summary['window_count']} ({summary['windows_per_point']} per point)
- CUDA projection: {summary['cuda_projection']}
- Minimum planning-band coverage: {summary['minimum_planning_band_coverage']:.2%}
- Maximum centre-azimuth error: {summary['maximum_center_azimuth_error_degrees']:.4f} degrees
- Maximum round-trip MAE: {summary['maximum_roundtrip_mae']:.3f} RGB levels
- Peak reserved VRAM: {summary['peak_reserved_vram_mib']:.2f} MiB

Eight overlapping 100-degree pinhole windows are centred every 45 degrees. Each window stores an
explicit pixel-to-panorama coordinate map. The extreme top/bottom rows are outside the planning band;
the 10-90% vertical band is the formal geometry-coverage region used by later intervention layout.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--agent-results", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    log = make_logger()
    failed_path = LOG_DIR / "failed_files.csv"
    try:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        agent_path = args.agent_results.resolve() if args.agent_results else resolve(cfg["inputs"]["agent_results"])
        benchmark_path = args.benchmark.resolve() if args.benchmark else resolve(cfg["inputs"]["benchmark"])
        if not agent_path.is_file() or not benchmark_path.is_file():
            raise FileNotFoundError("stage_43 results or benchmark missing")
        agent = pd.read_csv(agent_path, dtype={"point_id": str})
        benchmark = pd.read_csv(benchmark_path, dtype={"point_id": str})
        selected = agent[["point_id", "policy_state", "next_tool"]].merge(benchmark[["point_id", "output_path", "benchmark_role"]], on="point_id", validate="one_to_one")
        missing = selected.loc[~selected.output_path.map(lambda value: Path(value).is_file())]
        check = {"status": "PASS" if torch.cuda.is_available() and len(selected) >= 1 and missing.empty else "FAIL", "points": len(selected), "headings": cfg["projection"]["headings"], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.mode == "check":
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0 if check["status"] == "PASS" else 2
        if check["status"] != "PASS":
            raise RuntimeError(f"Preflight failed: {check}")
        output = args.output_directory.resolve() if args.output_directory else resolve(cfg["outputs"]["directory"])
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
        output.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        grids, maps, valids = {}, {}, {}
        for heading in cfg["projection"]["headings"]:
            grids[heading], maps[heading], valids[heading] = correspondence_grid(cfg, heading, device)
        rows, failures = [], []
        started = time.perf_counter()
        for record in tqdm(selected.to_dict("records"), desc="Projecting panorama windows", unit="point", dynamic_ncols=True):
            point_id = str(record["point_id"])
            try:
                point_dir = output / "points" / point_id
                window_dir = point_dir / "windows"
                map_dir = point_dir / "correspondence"
                window_dir.mkdir(parents=True, exist_ok=True); map_dir.mkdir(parents=True, exist_ok=True)
                array = np.asarray(Image.open(record["output_path"]).convert("RGB"), dtype=np.uint8).copy()
                pano = torch.from_numpy(array).permute(2, 0, 1).to(device=device, dtype=torch.float32)
                padded = torch.cat((pano[:, :, -1:], pano, pano[:, :, :1]), dim=2)[None]
                windows, point_maps, point_valids = [], [], []
                center_errors = []
                for heading in cfg["projection"]["headings"]:
                    window = F.grid_sample(padded, grids[heading], mode="bilinear", padding_mode="border", align_corners=True)[0]
                    window = torch.where(valids[heading][None], window, torch.zeros_like(window))
                    windows.append(window); point_maps.append(maps[heading]); point_valids.append(valids[heading])
                    Image.fromarray(window.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()).save(window_dir / f"heading_{heading:03d}.png", compress_level=3)
                    np.savez_compressed(map_dir / f"heading_{heading:03d}.npz", panorama_xy=maps[heading].half().cpu().numpy(), valid=valids[heading].cpu().numpy(), heading=heading, fov=cfg["projection"]["horizontal_fov_degrees"])
                    center = maps[heading][:, maps[heading].shape[1] // 2, maps[heading].shape[2] // 2]
                    centre_azimuth = float(center[0] / cfg["projection"]["panorama_width"] * 360)
                    center_errors.append(abs((centre_azimuth - heading + 180) % 360 - 180))
                reconstruction, weight = reconstruct(windows, point_maps, point_valids, pano.shape[1], pano.shape[2])
                band = cfg["quality_gates"]["planning_band_y_fraction"]
                y0, y1 = int(pano.shape[1] * band[0]), int(pano.shape[1] * band[1])
                covered = weight > 0
                coverage = float(covered[y0:y1].float().mean())
                mae = float((reconstruction[:, y0:y1] - pano[:, y0:y1]).abs()[:, covered[y0:y1]].mean())
                Image.fromarray(reconstruction.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()).save(point_dir / "roundtrip_reconstruction.png", compress_level=3)
                rows.append({"point_id": point_id, "benchmark_role": record["benchmark_role"], "policy_state": record["policy_state"], "window_count": len(windows), "planning_band_coverage": coverage, "center_azimuth_error_max": max(center_errors), "roundtrip_mae": mae, "north_window_center_x": float(maps[0][0, 256, 256]), "success": True})
            except Exception as error:
                log.exception("Point %s failed", point_id)
                failures.append({"point_id": point_id, "filename": record["output_path"], "error_message": f"{type(error).__name__}: {error}"})
        qc = pd.DataFrame(rows)
        qc.to_csv(output / "window_qc.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(failures, columns=["point_id", "filename", "error_message"]).to_csv(failed_path, index=False, encoding="utf-8-sig")
        gates = cfg["quality_gates"]
        passed = len(qc) == len(selected) and not failures and qc.planning_band_coverage.min() >= gates["minimum_planning_band_coverage"] and qc.center_azimuth_error_max.max() <= gates["maximum_center_azimuth_error_degrees"] and qc.roundtrip_mae.max() <= gates["maximum_roundtrip_mae"]
        summary = {"status": "PASS" if passed else "FAIL", "generated": datetime.now().astimezone().isoformat(), "runtime_seconds": time.perf_counter() - started, "point_count": len(qc), "window_count": int(qc.window_count.sum()) if len(qc) else 0, "windows_per_point": len(cfg["projection"]["headings"]), "failure_count": len(failures), "cuda_projection": True, "minimum_planning_band_coverage": float(qc.planning_band_coverage.min()) if len(qc) else 0.0, "maximum_center_azimuth_error_degrees": float(qc.center_azimuth_error_max.max()) if len(qc) else None, "maximum_roundtrip_mae": float(qc.roundtrip_mae.max()) if len(qc) else None, "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / 2**20}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(summary, qc, args.report.resolve() if args.report else resolve(cfg["outputs"]["report"]))
        log.info("stage_44 %s: %s", summary["status"], summary)
        return 0 if passed else 2
    except Exception as error:
        log.exception("stage_44 failed")
        with failed_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0: writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_44", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
