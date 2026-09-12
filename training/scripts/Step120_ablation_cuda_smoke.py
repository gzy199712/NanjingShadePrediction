"""Step120: implement and CUDA-smoke-test the frozen ablation matrix."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve(); PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from training.src.ablation_model import AblationSolarDirectionalMultitaskModel, AblationSpec  # noqa: E402
from training.src.data_loading import ManifestDataset, TrainingPaths, load_formal_data, load_specs, load_yaml  # noqa: E402
from training.src.losses import MultitaskSmoothL1Loss  # noqa: E402
from training.src.model import SolarDirectionalMultitaskModel  # noqa: E402
from training.src.reproducibility import seed_everything  # noqa: E402
from training.src.trainer import TrainingScalers, prepare_batch  # noqa: E402


SCRIPT_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/multidate_ablation_smoke.yaml")
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(root: Path, value: str) -> Path:
    path = Path(value); return (path if path.is_absolute() else root / path).resolve()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else ["variant_id", "error_message"]
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def logger_for(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True); logger = logging.getLogger("step120"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def ablation_spec(value: dict[str, Any]) -> AblationSpec:
    return AblationSpec(
        variant_id=str(value["id"]), zero_dynamic_fields=tuple(value.get("zero_dynamic_fields", [])),
        use_global_dinov2=bool(value.get("use_global_dinov2", True)),
        use_directional_dinov2=bool(value.get("use_directional_dinov2", True)),
        use_semantic_structure=bool(value.get("use_semantic_structure", True)),
        use_cross_attention=bool(value.get("use_cross_attention", True)),
        use_azimuth_encoding=bool(value.get("use_azimuth_encoding", True)),
        connect_shade_to_tmrt=bool(value.get("connect_shade_to_tmrt", True)),
    )


def scaler_from_checkpoint() -> TrainingScalers:
    path = PROJECT_ROOT / "training/checkpoints/multidate_full/seed_42/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return TrainingScalers.from_state_dict(checkpoint["feature_scalers"])


def branch_gradient_summary(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "directional": ("directional_projection", "direction_encoding", "cross_attention"),
        "semantic": ("semantic_encoder",), "global_dino": ("global_dino_encoder",),
        "dynamic": ("dynamic_encoder",), "shade": ("shade_hidden", "shade_output"), "tmrt": ("tmrt_head",),
    }
    summary: dict[str, float] = {}
    for group, prefixes in groups.items():
        values = [float(parameter.grad.detach().norm()) for name, parameter in model.named_parameters() if name.startswith(prefixes) and parameter.grad is not None]
        summary[f"gradient_{group}"] = float(np.linalg.norm(values)) if values else 0.0
    return summary


def check_inactive_gradients(spec: AblationSpec, gradients: dict[str, float]) -> bool:
    expected_zero: list[str] = []
    if not spec.use_directional_dinov2: expected_zero.append("gradient_directional")
    if not spec.use_semantic_structure: expected_zero.append("gradient_semantic")
    if not spec.use_global_dinov2: expected_zero.append("gradient_global_dino")
    return all(abs(gradients[field]) <= 1e-12 for field in expected_zero)


def full_equivalence(config: dict[str, Any], specs: Any, batch: dict[str, Any], device: torch.device, scalers: TrainingScalers) -> dict[str, float]:
    seed_everything(42)
    base = SolarDirectionalMultitaskModel(len(specs.semantic_fields), len(specs.dino_mean_fields), len(specs.dynamic_fields), specs.window_azimuth, config["model"], 0.0).to(device)
    candidate = AblationSolarDirectionalMultitaskModel(len(specs.semantic_fields), len(specs.dino_mean_fields), specs.dynamic_fields, specs.window_azimuth, config["model"], 0.0, AblationSpec("full_model")).to(device)
    candidate.load_state_dict(base.state_dict(), strict=True); base.eval(); candidate.eval(); prepared = prepare_batch(batch, device, scalers.to(device), True)
    with torch.inference_mode():
        first = base(prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"])
        second = candidate(prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"])
    result = {key: float((first[key] - second[key]).abs().max()) for key in ("shade_prediction", "tmrt_standardized", "attention_weights")}
    del base, candidate; torch.cuda.empty_cache(); return result


def run_variant(config: dict[str, Any], specs: Any, data: Any, manifest: pd.DataFrame, scalers: TrainingScalers, variant: dict[str, Any], device: torch.device) -> dict[str, Any]:
    smoke = config["smoke"]; spec = ablation_spec(variant); seed_everything(int(smoke["seed"]))
    dataset = ManifestDataset(manifest, data.semantic, data.dino_mean, data.dino_window, data.dynamic, data.labels, specs)
    loader = DataLoader(dataset, batch_size=int(smoke["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(int(smoke["seed"])))
    model = AblationSolarDirectionalMultitaskModel(len(specs.semantic_fields), len(specs.dino_mean_fields), specs.dynamic_fields, specs.window_azimuth, config["model"], float(config["training"]["dropout"]), spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(smoke["learning_rate"]), weight_decay=float(smoke["weight_decay"])); criterion = MultitaskSmoothL1Loss(float(config["training"]["lambda_shade"]), float(config["training"]["lambda_tmrt"])); amp = torch.amp.GradScaler("cuda", enabled=True); gpu_scalers = scalers.to(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); losses: list[float] = []; gradient_norms: list[float] = []; branch = {}
    started = time.perf_counter()
    for index, batch in tqdm(enumerate(loader), total=int(smoke["batches_per_variant"]), desc=spec.variant_id, unit="batch", dynamic_ncols=True, leave=False):
        if index >= int(smoke["batches_per_variant"]): break
        prepared = prepare_batch(batch, device, gpu_scalers, True); optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            output = model(prepared["semantic_features"], prepared["dino_mean"], prepared["dino_windows"], prepared["dynamic_features"])
            result = criterion(output["shade_prediction"], prepared["shade_target"], output["tmrt_standardized"], prepared["tmrt_target_standardized"])
        amp.scale(result.total).backward(); amp.unscale_(optimizer); gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(smoke["gradient_clip_norm"])))
        if index == 0: branch = branch_gradient_summary(model)
        amp.step(optimizer); amp.update(); losses.append(float(result.total.detach())); gradient_norms.append(gradient_norm)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    finite = bool(np.isfinite(losses).all() and np.isfinite(gradient_norms).all())
    row = {"variant_id": spec.variant_id, "batches": len(losses), "records": len(losses) * int(smoke["batch_size"]), "mean_loss": float(np.mean(losses)), "maximum_gradient_norm": float(np.max(gradient_norms)), "finite": finite, "inactive_gradient_check": check_inactive_gradients(spec, branch), "elapsed_seconds": elapsed, "peak_allocated_mb": torch.cuda.max_memory_allocated()/1024**2, "peak_reserved_mb": torch.cuda.max_memory_reserved()/1024**2, **branch}
    del model, optimizer, criterion, amp, loader, dataset; torch.cuda.empty_cache(); return row


def main() -> int:
    args = parse_args(); config_path = args.config.resolve(); config = load_yaml(config_path)
    project_value = Path(config["project"]["root"]); root = (config_path.parent / project_value).resolve() if not project_value.is_absolute() else project_value
    smoke = config["smoke"]; metrics_dir = project_path(root, smoke["metrics_dir"]); report_dir = project_path(root, smoke["report_dir"]); record_dir = project_path(root, smoke["record_dir"])
    for directory in (metrics_dir, report_dir, record_dir): directory.mkdir(parents=True, exist_ok=True)
    logger = logger_for(record_dir / "step120.log"); started = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    try:
        if args.mode == "run" and not args.approved_by_user: raise PermissionError("--approved-by-user is required")
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU fallback forbidden")
        variants = config["variants"]
        if len(variants) != 11 or len({value["id"] for value in variants}) != 11: raise RuntimeError("Frozen matrix must contain 11 unique variants")
        if args.mode == "check":
            report = {"status":"PASS", "variant_count":len(variants), "variant_ids":[value["id"] for value in variants], "cuda":torch.cuda.get_device_name(0), "no_training_performed":True}
            atomic_json(record_dir / "preflight.json", report); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0
        paths = TrainingPaths.from_config(config_path, config); data = load_formal_data(paths); specs = load_specs(paths)
        train = data.manifest[data.manifest["spatiotemporal_partition"].eq("train")].iloc[:int(smoke["sample_records"])].copy().reset_index(drop=True)
        if len(train) != int(smoke["sample_records"]) or set(train["weather_split"]) != {"train"}: raise RuntimeError("Smoke subset mismatch")
        scalers = scaler_from_checkpoint(); preview_loader = DataLoader(ManifestDataset(train.iloc[:256], data.semantic, data.dino_mean, data.dino_window, data.dynamic, data.labels, specs), batch_size=256, shuffle=False)
        equivalence = full_equivalence(config, specs, next(iter(preview_loader)), torch.device("cuda:0"), scalers)
        rows = [run_variant(config, specs, data, train, scalers, variant, torch.device("cuda:0")) for variant in tqdm(variants, desc="Ablation CUDA smoke", unit="variant", dynamic_ncols=True)]
        all_pass = all(row["finite"] and row["inactive_gradient_check"] for row in rows) and max(equivalence.values()) <= 1e-7
        atomic_csv(metrics_dir / "variant_smoke_metrics.csv", rows)
        implementation = [{**variant, "spec": ablation_spec(variant).__dict__} for variant in variants]
        atomic_json(metrics_dir / "implementation_manifest.json", implementation)
        actual_seconds = [2424.4419592, 2727.7848286, 2722.8042275, 3536.7940175, 4151.3364815]
        mean_hours = float(np.mean(actual_seconds))/3600; new_runs = 10 * len(config["training"]["formal_seeds"])
        ended = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
        summary = {"step":120, "status":"PASS" if all_pass else "FAIL", "started_at":started.isoformat(), "ended_at":ended.isoformat(), "elapsed_seconds":(ended-started).total_seconds(), "variant_count":len(rows), "all_variants_finite":all(row["finite"] for row in rows), "inactive_gradient_checks_pass":all(row["inactive_gradient_check"] for row in rows), "full_model_equivalence_max_abs":equivalence, "gpu":torch.cuda.get_device_name(0), "formal_training_started":False, "formal_baseline_reused":True, "baseline_seeds":[42,52,62], "new_ablation_training_runs":new_runs, "estimated_new_gpu_hours":mean_hours*new_runs, "recommended_execution":"Three staged groups with frozen Test withheld until all training is complete."}
        atomic_json(report_dir / "summary.json", summary); atomic_csv(record_dir / "failed_files.csv", []); logger.info("Step120 ended | status=%s", summary["status"]); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0 if all_pass else 2
    except Exception as error:
        logger.exception("Step120 failed"); atomic_json(record_dir / "failure.json", {"status":"FAIL", "error":str(error), "traceback":traceback.format_exc()}); return 1


if __name__ == "__main__": raise SystemExit(main())
