"""stage_37: preflight the multimodal planner-agent and structure-first AIGC mainline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/planner_agent_foundation.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_37"


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_37")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "stage_37.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(value: str) -> Path:
    return ROOT / value


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise NotADirectoryError(path)


def evenly_spaced(frame: pd.DataFrame, count: int, sort_fields: list[str]) -> pd.DataFrame:
    available = [field for field in sort_fields if field in frame.columns]
    ordered = frame.sort_values(available, kind="stable") if available else frame.sort_values("point_id")
    if len(ordered) <= count:
        return ordered
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return ordered.iloc[indices]


def build_benchmark(config: dict[str, Any], decisions: pd.DataFrame, panorama: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    sort_fields = list(config["benchmark"]["sort_fields"])
    for class_name, requested in tqdm(
        config["benchmark"]["strata"].items(),
        desc="Selecting planner-agent strata",
        unit="stratum",
        dynamic_ncols=True,
    ):
        members = decisions.loc[decisions["final_decision_class"] == class_name].copy()
        if members.empty:
            continue
        if requested == "all":
            chosen = members
        else:
            chosen = evenly_spaced(members, min(int(requested), len(members)), sort_fields)
        selected.append(chosen)
    if not selected:
        raise RuntimeError("No benchmark points selected")
    result = pd.concat(selected, ignore_index=True).drop_duplicates("point_id")
    panorama_columns = panorama[["point_id", "output_path", "width", "height", "north_at_x0"]].copy()
    result = result.merge(panorama_columns, on="point_id", how="left", validate="one_to_one")
    result["panorama_exists"] = result["output_path"].map(lambda value: Path(str(value)).is_file())
    result["benchmark_role"] = result["final_decision_class"]
    columns = [
        "point_id", "benchmark_role", "final_decision_stage", "final_decision_class",
        "optimization_eligible_final", "planner_review_required_final", "month", "SVF", "GVI",
        "semantic_sky_ratio", "semantic_vegetation_ratio", "retained_walkable_ratio",
        "existing_tree_evidence", "walkable_direction_count", "tree_evidence_direction_count",
        "shade_pred_mean", "tmrt_pred_mean", "utci_pred_mean", "output_path", "width", "height",
        "north_at_x0", "panorama_exists",
    ]
    return result[[column for column in columns if column in result.columns]].sort_values(
        ["benchmark_role", "point_id"], kind="stable"
    )


def hardware_qc(config: dict[str, Any]) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    gpu_total_gib = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda else 0.0
    )
    disk = shutil.disk_usage(ROOT)
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("transformers", "accelerate", "bitsandbytes", "flash_attn", "diffusers", "jsonschema")
    }
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_total_gib": round(gpu_total_gib, 3),
        "disk_free_gib": round(disk.free / 1024**3, 3),
        "full_gpu_required": bool(config["runtime"]["require_full_gpu"]),
        "cpu_offload_allowed": bool(config["runtime"]["allow_cpu_offload"]),
        "packages": modules,
    }


def assess_models(config: dict[str, Any], hardware: dict[str, Any]) -> list[dict[str, Any]]:
    memory = float(hardware["gpu_total_gib"])
    rows: list[dict[str, Any]] = []
    for candidate in config["model_candidates"]:
        row = dict(candidate)
        if candidate["role"] == "image_editor_candidate":
            official = float(candidate["official_vram_gib"])
            row["bf16_full_gpu_feasible"] = memory >= official
            row["decision"] = (
                "BF16_FULL_GPU_CANDIDATE" if memory >= official
                else "BF16_REJECTED_REQUIRE_VERIFIED_FULL_GPU_QUANTIZATION_OR_LARGER_GPU"
            )
        elif candidate["role"] == "multimodal_reasoner_primary":
            row["bf16_full_gpu_feasible"] = False if memory <= 8.5 else None
            row["decision"] = "QUANTIZED_SINGLE_POINT_SMOKE_REQUIRED"
        else:
            row["bf16_full_gpu_feasible"] = None
            row["decision"] = "SINGLE_POINT_FULL_GPU_SMOKE_REQUIRED"
        rows.append(row)
    return rows


def validate_contract(config: dict[str, Any], hardware: dict[str, Any], benchmark: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    if not hardware["cuda_available"]:
        failures.append("CUDA is unavailable")
    if hardware["disk_free_gib"] < float(config["runtime"]["minimum_free_disk_gib"]):
        failures.append("Free disk is below the frozen minimum")
    if not benchmark["panorama_exists"].all():
        failures.append("One or more benchmark panoramas are missing")
    if not benchmark["north_at_x0"].fillna(False).all():
        failures.append("One or more benchmark panoramas are not north-aligned at x=0")
    return failures


def write_report(
    config: dict[str, Any], hardware: dict[str, Any], models: list[dict[str, Any]],
    benchmark: pd.DataFrame, failures: list[str], report_path: Path,
) -> None:
    counts = benchmark["benchmark_role"].value_counts().sort_index()
    model_lines = "\n".join(
        f"- `{row['model_id']}` — {row['role']}: **{row['decision']}**"
        for row in models
    )
    strata_lines = "\n".join(f"- `{name}`: {int(count)}" for name, count in counts.items())
    status = "PASS" if not failures else "FAIL"
    body = f"""# stage_37 Planner Agent foundation preflight

Status: **{status}**  
Generated: {datetime.now().astimezone().isoformat()}

## Scope

This step freezes the new planner-facing mainline: multimodal scene reasoning, structured QA,
multimodal retrieval, tool-constrained planning agent, structure-first AIGC and automatic acceptance.
The existing route-planning system and citywide decisions remain unchanged.

## Hardware

- GPU: {hardware['gpu_name']}
- VRAM: {hardware['gpu_total_gib']:.3f} GiB
- CUDA: {hardware['cuda_available']} ({hardware['cuda_runtime']})
- Free disk: {hardware['disk_free_gib']:.3f} GiB
- Full-GPU only: {hardware['full_gpu_required']}
- CPU offload allowed: {hardware['cpu_offload_allowed']}

## Model gates

{model_lines}

FLUX.2 klein 4B is not approved in BF16 on an 8 GiB GPU because the official model card
states approximately 13 GiB VRAM. A later step may test a pinned quantized checkpoint only if
all model tensors remain on CUDA; otherwise this candidate must be replaced or moved to larger hardware.

## Frozen benchmark

- Points: {len(benchmark)}
- Missing panoramas: {int((~benchmark['panorama_exists']).sum())}
- North-alignment failures: {int((~benchmark['north_at_x0'].fillna(False)).sum())}

{strata_lines}

## Contract failures

{chr(10).join(f'- {failure}' for failure in failures) if failures else '- None'}

## Next gate

stage_38 may download and benchmark Qwen3-VL-2B and Qwen3-VL-4B only after dependency,
revision, license and disk checks. Passing means valid schema-constrained single-point output,
zero CPU offload/fallback, measured peak VRAM, and unchanged source imagery.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = configure_logging()
    started = datetime.now().astimezone()
    failures_file = LOG_DIR / "failed_files.csv"
    try:
        config = load_config()
        decisions_path = resolve(config["inputs"]["decisions"])
        panorama_path = resolve(config["inputs"]["panorama_metadata"])
        schema_path = resolve(config["inputs"]["response_schema"])
        require_file(decisions_path)
        require_file(panorama_path)
        require_file(schema_path)
        require_dir(resolve(config["inputs"]["panorama_dir"]))
        json.loads(schema_path.read_text(encoding="utf-8"))

        logger.info("Reading formal citywide decisions")
        decisions = pd.read_csv(decisions_path, low_memory=False)
        panorama = pd.read_csv(panorama_path)
        benchmark = build_benchmark(config, decisions, panorama)
        hardware = hardware_qc(config)
        models = assess_models(config, hardware)
        failures = validate_contract(config, hardware, benchmark)
        summary = {
            "status": "PASS" if not failures else "FAIL",
            "started": started.isoformat(),
            "finished": datetime.now().astimezone().isoformat(),
            "citywide_points": int(len(decisions)),
            "benchmark_points": int(len(benchmark)),
            "benchmark_strata": benchmark["benchmark_role"].value_counts().to_dict(),
            "hardware": hardware,
            "model_assessments": models,
            "failures": failures,
        }
        if args.mode == "check":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if not failures else 2

        output_dir = resolve(config["outputs"]["directory"])
        report_path = resolve(config["outputs"]["report"])
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        benchmark.to_csv(output_dir / "planner_agent_benchmark_manifest.csv", index=False, encoding="utf-8-sig")
        (output_dir / "hardware_qc.json").write_text(
            json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "model_gate.json").write_text(
            json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_report(config, hardware, models, benchmark, failures, report_path)
        with failures_file.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            writer.writeheader()
            for failure in failures:
                writer.writerow({"point_id": "", "filename": "stage_37", "error_message": failure})
        logger.info("stage_37 complete: %s, benchmark=%d", summary["status"], len(benchmark))
        return 0 if not failures else 2
    except Exception as error:
        logger.exception("stage_37 failed")
        with failures_file.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "point_id": "", "filename": "stage_37",
                "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
