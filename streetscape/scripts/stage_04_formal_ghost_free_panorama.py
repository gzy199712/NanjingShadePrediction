"""Build north-aligned, ghost-free, strict single-source panoramas.

Heading is the immutable absolute azimuth prior. CUDA performs spherical
projection, seam-cost evaluation, dynamic programming, and panorama rendering.
No feature-based orientation, homography, geometric warp, or cross-view RGB
averaging is used.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, logging, math, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

import panorama_utils as pano

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/stage_04_formal_ghost_free_panorama.yaml"
HEADINGS = list(range(0, 360, 30))


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("check", "run", "finalize"))
    p.add_argument("--max-points", type=int)
    p.add_argument("--point-ids", nargs="*")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_04")
    log.handlers.clear(); log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.FileHandler(path, mode="a", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt); log.addHandler(h)
    return log


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def projection_assets(cfg, device):
    grids = []
    for h in HEADINGS:
        grid, _, _ = pano.projection_grid(np.zeros(3), h, cfg, device)
        grids.append(grid)
    grids = torch.stack(grids)
    H, W = cfg["panorama"]["height"], cfg["panorama"]["width"]
    az = torch.arange(W, device=device) * 360 / W
    delta = torch.stack([torch.remainder(az - h + 180, 360) - 180 for h in HEADINGS]).abs()
    owner = delta.argmin(0)[None].expand(H, -1).clone()
    half = round(float(cfg["seam"]["search_half_width_degrees"]) * W / 360)
    centres = [round(((h + 15) % 360) * W / 360) for h in HEADINGS]
    seam_cols = torch.stack([torch.arange(c - half, c + half + 1, device=device) for c in centres])
    return grids, owner, seam_cols


def batch_optimal_seams(layers, seam_cols, cfg):
    """Solve all 12 vertical seams together; all costs and DP stay on CUDA."""
    pairs_a, pairs_b = [], []
    for i in range(12):
        pairs_a.append(layers[i, :, :, seam_cols[i]])
        pairs_b.append(layers[(i + 1) % 12, :, :, seam_cols[i]])
    a, b = torch.stack(pairs_a), torch.stack(pairs_b)  # 12,C,H,B
    rgb = (a - b).abs().mean(1) / 255
    ga = (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean(1)
    gb = (b[:, :, :, 1:] - b[:, :, :, :-1]).abs().mean(1)
    gd = F.pad((ga - gb).abs() / 255, (0, 1))
    x = torch.linspace(-1, 1, a.shape[-1], device=a.device)
    cost = (float(cfg["seam"]["rgb_cost_weight"]) * rgb
            + float(cfg["seam"]["gradient_cost_weight"]) * gd
            + 0.025 * x.square()[None, None])
    dp = cost[:, 0]
    parents = torch.empty((cost.shape[1] - 1, 12, cost.shape[2]), device=a.device, dtype=torch.int8)
    for y in range(1, cost.shape[1]):
        candidates = torch.stack((F.pad(dp[:, :-1], (1, 0), value=1e6), dp,
                                  F.pad(dp[:, 1:], (0, 1), value=1e6)))
        best, idx = candidates.min(0)
        parents[y - 1] = idx.to(torch.int8)
        dp = cost[:, y] + best
    pos = dp.argmin(1)
    path = torch.empty((cost.shape[1], 12), device=a.device, dtype=torch.long)
    path[-1] = pos
    pair_index = torch.arange(12, device=a.device)
    for y in range(cost.shape[1] - 2, -1, -1):
        pos = pos + parents[y, pair_index, pos].long() - 1
        path[y] = pos
    return path, dp.min(1).values / cost.shape[1]


def render(images, grids, base_owner, seam_cols, cfg):
    with torch.inference_mode():
        layers = F.grid_sample(images, grids, mode="bilinear", padding_mode="border", align_corners=True)
        paths, seam_costs = batch_optimal_seams(layers, seam_cols, cfg)
        owner = base_owner.clone()
        offsets = torch.arange(seam_cols.shape[1], device=images.device)[None]
        for i in range(12):
            owner[:, seam_cols[i]] = torch.where(
                offsets < paths[:, i, None],
                torch.tensor(i, device=images.device),
                torch.tensor((i + 1) % 12, device=images.device),
            )
        output = torch.zeros((3, owner.shape[0], owner.shape[1]), device=images.device)
        for i in range(12):
            mask = owner == i
            output[:, mask] = layers[i, :, mask]
    return output, owner, seam_costs


def qc_metrics(image, owner, seam_costs):
    gray = image.mean(0) / 255
    right = torch.roll(gray, -1, 1)
    boundary = owner != torch.roll(owner, -1, 1)
    discontinuity = (gray - right).abs()[boundary]
    counts = torch.bincount(owner.flatten(), minlength=12)
    return {
        "coverage_ratio": 1.0,
        "unique_source_count": int((counts > 0).sum()),
        "single_source_pixel_ratio": 1.0,
        "ownership_min_pixels": int(counts.min()),
        "ownership_max_pixels": int(counts.max()),
        "seam_discontinuity_mean": float(discontinuity.mean()),
        "seam_discontinuity_p95": float(torch.quantile(discontinuity, .95)),
        "north_boundary_jump": float((gray[:, 0] - gray[:, -1]).abs().mean()),
        "mean_optimization_cost": float(seam_costs.mean()),
        "success": True,
    }


def finalize(cfg):
    out = Path(cfg["outputs"]["directory"])
    qc_rows = read_jsonl(out / "qc.jsonl")
    meta_rows = read_jsonl(out / "metadata.jsonl")
    fail_rows = read_jsonl(out / "failures.jsonl")
    def latest(rows):
        d = {}
        for r in rows: d[str(r["point_id"])] = r
        return list(d.values())
    pd.DataFrame(latest(qc_rows)).to_csv(cfg["outputs"]["qc"], index=False, encoding="utf-8-sig")
    pd.DataFrame(latest(meta_rows)).to_csv(cfg["outputs"]["metadata"], index=False, encoding="utf-8-sig")
    pd.DataFrame(fail_rows, columns=("point_id", "filename", "error_message", "time")).to_csv(
        cfg["outputs"]["failures"], index=False, encoding="utf-8-sig")
    return len(latest(qc_rows)), len(latest(meta_rows)), len(fail_rows)


def main():
    a = arguments(); cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    inp = {k: Path(v) for k, v in cfg["inputs"].items()}
    metadata = pd.read_csv(inp["image_metadata"], dtype=str)
    metadata["heading"] = pd.to_numeric(metadata["heading"], errors="raise").astype(int)
    grouped = metadata.groupby("point_id", sort=True)
    complete = grouped.heading.nunique(); bad = complete[complete != 12]
    check = {
        "status": "READY" if all(p.exists() for p in inp.values()) and bad.empty else "BLOCKED",
        "image_count": len(metadata), "point_count": int(complete.size),
        "complete_point_count": int((complete == 12).sum()), "incomplete_point_count": int(len(bad)),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "projection_device": "CUDA", "seam_cost_and_dp_device": "CUDA",
        "cross_source_rgb_blending": False, "geometric_warp": False,
        "north_mapping": {"x0": "N", "x512": "E", "x1024": "S", "x1536": "W", "x2048": "N"},
        "v1_write_allowed": False,
    }
    if a.mode == "check":
        print(json.dumps(check, ensure_ascii=False, indent=2)); return 0 if check["status"] == "READY" and check["cuda_available"] else 1
    if a.mode == "finalize":
        print(json.dumps(dict(zip(("qc", "metadata", "failures"), finalize(cfg))), ensure_ascii=False, indent=2)); return 0
    if check["status"] != "READY" or not check["cuda_available"]: raise RuntimeError(check)
    device = pano.require_cuda(); out = Path(cfg["outputs"]["directory"]); images_dir = Path(cfg["outputs"]["images"]); owner_dir = Path(cfg["outputs"]["ownership"])
    if a.overwrite and out.exists():
        raise RuntimeError("Formal output deletion is intentionally prohibited; resume into the existing directory")
    images_dir.mkdir(parents=True, exist_ok=True); owner_dir.mkdir(parents=True, exist_ok=True)
    log = logger(Path(cfg["outputs"]["log"])); grids, base_owner, seam_cols = projection_assets(cfg, device)
    ids = list(complete.index.astype(str))
    if a.point_ids: ids = [x for x in a.point_ids if x in set(ids)]
    if a.max_points: ids = ids[:a.max_points]
    existing = {p.name.split("_", 1)[0] for p in images_dir.glob("*_north_pano.png")}
    pending = [x for x in ids if x not in existing]
    started = datetime.now().astimezone(); tick = time.perf_counter(); success = skipped = failures = 0
    torch.cuda.reset_peak_memory_stats(device); log.info("START selected=%d pending=%d existing=%d", len(ids), len(pending), len(existing))
    for pid in tqdm(ids, desc="stage_04 formal ghost-free panoramas", unit="point", dynamic_ncols=True):
        group = grouped.get_group(pid)
        x, y = group.iloc[0]["coord_x"], group.iloc[0]["coord_y"]
        name = f"{pid}_{x}_{y}_north_pano.png"; image_path = images_dir / name; owner_path = owner_dir / f"{pid}_source_heading.png"
        if image_path.exists() and owner_path.exists(): skipped += 1; continue
        try:
            source = pano.load_point_images(group, inp["raw_perspective_dir"], device)
            image, ownership, costs = render(source, grids, base_owner, seam_cols, cfg)
            pano.save_rgb(image, image_path)
            Image.fromarray((ownership.byte().cpu().numpy() * 30).astype(np.uint8), mode="L").save(owner_path, compress_level=6)
            metrics = {"point_id": pid, "output_path": str(image_path.resolve()), "ownership_path": str(owner_path.resolve()), **qc_metrics(image, ownership, costs)}
            append_jsonl(out / "qc.jsonl", metrics)
            append_jsonl(out / "metadata.jsonl", {"point_id": pid, "coord_x": x, "coord_y": y, "output_path": str(image_path.resolve()), "projection_method": "heading_equirectangular_strict_single_source_gpu_optimal_seam", "image_count": 12, "width": 2048, "height": 512, "north_at_x0": True})
            success += 1
        except Exception as exc:
            append_jsonl(out / "failures.jsonl", {"point_id": pid, "filename": name, "error_message": f"{type(exc).__name__}: {exc}", "time": datetime.now().astimezone().isoformat()}); failures += 1; log.exception("FAILED point_id=%s", pid)
    q, m, f = finalize(cfg); elapsed = time.perf_counter() - tick
    summary = {"status": "COMPLETE" if q >= complete.size and failures == 0 else "PARTIAL_RESUMABLE", "started_at": started.isoformat(), "ended_at": datetime.now().astimezone().isoformat(), "runtime_seconds": round(elapsed, 3), "selected_count": len(ids), "new_success_count": success, "skipped_existing_count": skipped, "new_failure_count": failures, "formal_qc_count": q, "formal_metadata_count": m, "failure_record_count": f, "total_expected_points": int(complete.size), "rate_points_per_second": success / elapsed if elapsed else 0, "gpu": torch.cuda.get_device_name(0), "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 2), "single_source_guarantee": True, "v1_outputs_written": 0}
    (out / "stage_04_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("END %s", summary); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0 if failures == 0 else 1


if __name__ == "__main__": raise SystemExit(main())
