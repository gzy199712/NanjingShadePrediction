"""Step121: CUDA-only formal multi-date ablation training.

Model selection is restricted to Validation points x Validation weather. The
three frozen Test views are not read by this step. Step117 seeds 42/52/62 are
reused as the full-model baseline and are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.scripts.Step117_train_multidate_model import (  # noqa: E402
    fingerprint,
    settings_for,
    setup_logger,
    split_frames,
)
from training.src.ablation_model import (  # noqa: E402
    AblationSolarDirectionalMultitaskModel,
    AblationSpec,
)
from training.src.data_loading import (  # noqa: E402
    TrainingPaths,
    load_formal_data,
    load_specs,
    load_yaml,
)
from training.src.logging_utils import atomic_json  # noqa: E402
from training.src.reproducibility import collect_environment, collect_git_state  # noqa: E402
from training.src.trainer import fit_train_scalers, save_scalers, train_seed  # noqa: E402


SCRIPT_VERSION = "1.0.0"
GROUP_ORDER = ("visual_semantic", "weather", "fusion_multitask")
EXPECTED_VARIANTS = 10
EXPECTED_SEEDS = [42, 52, 62]


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "training/configs/multidate_ablation_formal.yaml",
    )
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--groups", nargs="+", choices=GROUP_ORDER)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def ablation_spec(value: dict[str, Any]) -> AblationSpec:
    return AblationSpec(
        variant_id=str(value["id"]),
        zero_dynamic_fields=tuple(value.get("zero_dynamic_fields", [])),
        use_global_dinov2=bool(value.get("use_global_dinov2", True)),
        use_directional_dinov2=bool(value.get("use_directional_dinov2", True)),
        use_semantic_structure=bool(value.get("use_semantic_structure", True)),
        use_cross_attention=bool(value.get("use_cross_attention", True)),
        use_azimuth_encoding=bool(value.get("use_azimuth_encoding", True)),
        connect_shade_to_tmrt=bool(value.get("connect_shade_to_tmrt", True)),
    )


def selected_variants(config: dict[str, Any], groups: list[str] | None) -> list[tuple[str, dict[str, Any]]]:
    selected = groups or list(GROUP_ORDER)
    return [
        (group, variant)
        for group in GROUP_ORDER
        if group in selected
        for variant in config["groups"][group]
    ]


def verify_dependencies(config: dict[str, Any]) -> dict[str, Any]:
    if str(config["runtime"]["device"]).lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Step121 requires CUDA; CPU fallback is forbidden")
    smoke_path = PROJECT_ROOT / "training/reports/ablation_preflight/summary.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("formal_training_started") is not False:
        raise RuntimeError("Step120 clean PASS is missing")
    if [int(value) for value in config["training"]["seeds"]] != EXPECTED_SEEDS:
        raise RuntimeError(f"Frozen formal seeds must be {EXPECTED_SEEDS}")
    baseline: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        path = PROJECT_ROOT / f"training/checkpoints/multidate_full/seed_{seed}/best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        baseline.append(
            {
                "seed": seed,
                "path": str(path),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_validation_total_loss": float(checkpoint["best_validation_total_loss"]),
            }
        )
    variants = selected_variants(config, None)
    if len(variants) != EXPECTED_VARIANTS or len({item[1]["id"] for item in variants}) != EXPECTED_VARIANTS:
        raise RuntimeError("Formal ablation matrix must contain 10 unique variants")
    return {"smoke": smoke, "baseline": baseline}


def check_phase(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    dependencies = verify_dependencies(config)
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths)
    train, validation = split_frames(data.manifest, config)
    scalers, provenance = fit_train_scalers(train, data, load_specs(paths))
    del scalers, data
    return {
        "step": 121,
        "status": "PASS",
        "checked_at": now_text(),
        "ready_for_formal_training": True,
        "gpu": torch.cuda.get_device_name(0),
        "groups": list(GROUP_ORDER),
        "variant_count": EXPECTED_VARIANTS,
        "seeds": EXPECTED_SEEDS,
        "new_training_runs": EXPECTED_VARIANTS * len(EXPECTED_SEEDS),
        "baseline_reused": dependencies["baseline"],
        "train_records": len(train),
        "validation_records": len(validation),
        "test_records_used": 0,
        "scaler_provenance": provenance,
        "config_fingerprint": fingerprint(config),
        "no_training_performed": True,
    }


def output_policy(variants: list[tuple[str, dict[str, Any]]], seeds: list[int], config: dict[str, Any], overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    checkpoint_root = project_path(config["training"]["checkpoint_root"])
    metrics_root = project_path(config["training"]["metrics_root"])
    existing = []
    for _, variant in variants:
        for seed in seeds:
            existing.extend(
                path for path in (
                    checkpoint_root / variant["id"] / f"seed_{seed}/last.pt",
                    metrics_root / variant["id"] / f"seed_{seed}/training_history.csv",
                ) if path.exists()
            )
    if existing and not (overwrite or resume):
        raise FileExistsError("Step121 outputs exist; use --resume or --overwrite")
    if overwrite:
        for _, variant in variants:
            for root in (checkpoint_root, metrics_root):
                path = root / variant["id"]
                if path.exists():
                    shutil.rmtree(path)


def run_training(config_path: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.approved_by_user:
        raise PermissionError("--approved-by-user is required")
    check_path = PROJECT_ROOT / "training/reports/ablation_training/step121_check.json"
    check = json.loads(check_path.read_text(encoding="utf-8"))
    if not check.get("ready_for_formal_training") or check["config_fingerprint"] != fingerprint(config):
        raise RuntimeError("Step121 check is missing, failed, or stale")
    dependencies = verify_dependencies(config)
    variants = selected_variants(config, args.groups)
    seeds = EXPECTED_SEEDS
    output_policy(variants, seeds, config, args.overwrite, args.resume)
    paths = TrainingPaths.from_config(config_path, config)
    data = load_formal_data(paths)
    specs = load_specs(paths)
    train, validation = split_frames(data.manifest, config)
    scalers, scaler_provenance = fit_train_scalers(train, data, specs)
    save_scalers(project_path(config["training"]["scaler_path"]), scalers, scaler_provenance)
    step117_check = json.loads((PROJECT_ROOT / "training/reports/multidate_training/step117_check.json").read_text(encoding="utf-8"))
    input_fingerprints = step117_check["input_fingerprints"]
    environment = collect_environment()
    git_state = collect_git_state(PROJECT_ROOT, 50).as_dict()
    checkpoint_root = project_path(config["training"]["checkpoint_root"])
    metrics_root = project_path(config["training"]["metrics_root"])
    log_root = project_path(config["training"]["log_root"])
    status_path = project_path(config["training"]["status_path"])
    summary_path = project_path(config["training"]["summary_path"])
    status: dict[str, Any] = {
        "step": 121, "state": "running", "started_at": now_text(), "updated_at": now_text(),
        "active_group": None, "active_variant": None, "active_seed": None,
        "completed_runs": [], "requested_groups": args.groups or list(GROUP_ORDER),
        "total_runs": len(variants) * len(seeds), "test_records_used": 0,
    }
    atomic_json(status_path, status)
    results: list[dict[str, Any]] = []
    for group, variant in tqdm(variants, desc="Formal ablation variants", unit="variant", dynamic_ncols=True):
        spec = ablation_spec(variant)
        for seed in tqdm(seeds, desc=spec.variant_id, unit="seed", dynamic_ncols=True, leave=False):
            checkpoint_dir = checkpoint_root / spec.variant_id / f"seed_{seed}"
            metrics_dir = metrics_root / spec.variant_id / f"seed_{seed}"
            last_path = checkpoint_dir / "last.pt"
            if args.resume and last_path.exists():
                saved = torch.load(last_path, map_location="cpu", weights_only=False)
                if saved.get("completed"):
                    result = {
                        "group": group, "variant_id": spec.variant_id, "seed": seed,
                        "completed": True, "resumed_completed": True,
                        "epochs_completed": int(saved["epoch"]), "best_epoch": int(saved["best_epoch"]),
                        "best_validation_total_loss": float(saved["best_validation_total_loss"]),
                        "best_checkpoint": str(checkpoint_dir / "best.pt"), "last_checkpoint": str(last_path),
                        "elapsed_seconds": 0.0,
                    }
                    results.append(result)
                    status["completed_runs"].append({"variant_id": spec.variant_id, "seed": seed})
                    atomic_json(status_path, status)
                    continue
            status.update({"active_group": group, "active_variant": spec.variant_id, "active_seed": seed, "updated_at": now_text()})
            atomic_json(status_path, status)
            active_config = copy.deepcopy(config)
            active_config["active_ablation"] = {"group": group, **variant}
            variant_fingerprint = fingerprint(active_config)

            def model_factory(dropout: float, current_spec: AblationSpec = spec) -> AblationSolarDirectionalMultitaskModel:
                return AblationSolarDirectionalMultitaskModel(
                    len(specs.semantic_fields), len(specs.dino_mean_fields), specs.dynamic_fields,
                    specs.window_azimuth, config["model"], dropout, current_spec,
                )

            def update_run(run_status: dict[str, Any]) -> None:
                status["run_status"] = run_status
                status["updated_at"] = now_text()
                atomic_json(status_path, status)

            logger = setup_logger(log_root / spec.variant_id / f"seed_{seed}/training.log", f"step121_{spec.variant_id}_{seed}", append=args.resume)
            result = train_seed(
                data=data, specs=specs, train_manifest=train, validation_manifest=validation,
                scalers_cpu=scalers, settings=settings_for(config, seed), model_config=config["model"],
                checkpoint_dir=checkpoint_dir, metrics_dir=metrics_dir, config=active_config,
                config_fingerprint=variant_fingerprint, input_fingerprints=input_fingerprints,
                git_state=git_state, environment=environment, resume=args.resume and last_path.exists(),
                logger=logger, status_callback=update_run, model_factory=model_factory,
            )
            result.update({"group": group, "variant_id": spec.variant_id})
            results.append(result)
            status["completed_runs"].append({"variant_id": spec.variant_id, "seed": seed})
            status["updated_at"] = now_text()
            atomic_json(status_path, status)
    summary = {
        "step": 121, "status": "COMPLETE", "started_at": status["started_at"], "ended_at": now_text(),
        "script": SCRIPT_PATH.name, "script_version": SCRIPT_VERSION,
        "groups": args.groups or list(GROUP_ORDER), "variant_count": len(variants), "seeds": seeds,
        "completed_run_count": len(results), "baseline_reused": dependencies["baseline"],
        "train_records": len(train), "validation_records": len(validation), "test_records_used": 0,
        "scaler_provenance": scaler_provenance, "environment": environment, "git": git_state, "results": results,
    }
    atomic_json(summary_path, summary)
    status.update({"state": "complete", "active_group": None, "active_variant": None, "active_seed": None, "updated_at": now_text(), "summary_path": str(summary_path)})
    atomic_json(status_path, status)
    return summary


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    report_dir = PROJECT_ROOT / "training/reports/ablation_training"
    report_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(PROJECT_ROOT / "training/logs/multidate_ablation/step121.log", "step121", append=args.resume)
    try:
        if args.mode == "check":
            report = check_phase(config_path, config)
            output = report_dir / "step121_check.json"
            if output.exists() and not args.overwrite:
                raise FileExistsError("Step121 check exists; use --overwrite")
            atomic_json(output, report)
            log.info("Step121 check PASS")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            summary = run_training(config_path, config, args)
            log.info("Step121 complete with %d runs", len(summary["results"]))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        log.exception("Step121 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
