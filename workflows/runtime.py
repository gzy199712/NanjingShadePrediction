"""Shared fail-fast checks and error recording for official workflows."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHOR_LOG_ROOT = PROJECT_ROOT / "author_workspace" / "logs" / "workflows"
FAILED_FIELDS = ("stage", "input", "error_message", "timestamp")


class PreflightError(RuntimeError):
    """Raised when an official workflow cannot start safely."""


def make_logger(name: str) -> logging.Logger:
    log_dir = AUTHOR_LOG_ROOT / name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"workflow.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def record_failure(workflow: str, stage: str, source: object, error: BaseException) -> None:
    path = AUTHOR_LOG_ROOT / workflow / "failed_files.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FAILED_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "stage": stage,
                "input": str(source),
                "error_message": f"{type(error).__name__}: {error}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )


def require_paths(paths: Iterable[Path], *, kind: str | None = None) -> None:
    missing: list[str] = []
    for path in paths:
        valid = path.exists()
        if kind == "file":
            valid = path.is_file()
        elif kind == "dir":
            valid = path.is_dir()
        if not valid:
            missing.append(str(path))
    if missing:
        raise PreflightError("missing required path(s): " + "; ".join(missing))


def require_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK):
        raise PreflightError(f"output directory is not writable: {path}")


def require_cuda() -> None:
    """Require both an NVIDIA driver and a CUDA-enabled PyTorch runtime."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise PreflightError("nvidia-smi not found; GPU workflow will not fall back to CPU")
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        message = probe.stderr.strip() or probe.stdout.strip() or "CUDA PyTorch probe failed"
        raise PreflightError(message)


@dataclass(frozen=True)
class Stage:
    name: str
    command: Sequence[str]


def run_stages(workflow: str, stages: Sequence[Stage], logger: logging.Logger) -> int:
    started = time.time()
    completed = 0
    logger.info("workflow=%s start stages=%d", workflow, len(stages))
    for stage in stages:
        logger.info("stage=%s command=%s", stage.name, subprocess.list2cmdline(list(stage.command)))
        try:
            result = subprocess.run(
                list(stage.command), cwd=PROJECT_ROOT, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(f"process returned {result.returncode}")
            completed += 1
        except Exception as error:
            logger.exception("stage=%s failed", stage.name)
            record_failure(workflow, stage.name, stage.command, error)
            return 1
    logger.info(
        "workflow=%s complete stages=%d elapsed_s=%.1f",
        workflow,
        completed,
        time.time() - started,
    )
    return 0
