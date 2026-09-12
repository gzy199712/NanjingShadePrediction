"""Git, environment, seed, and hardware provenance."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import sklearn
import torch


@dataclass
class GitState:
    repository_exists: bool
    branch: str | None
    commit_hash: str | None
    clean: bool
    status_entries: list[str]
    large_untracked_files: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        *args,
    ]
    return subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _lfs_status(root: Path, relative: str) -> bool:
    result = _git(root, "check-attr", "filter", "--", relative)
    return "filter: lfs" in result.stdout


def collect_git_state(root: Path, large_threshold_mb: float) -> GitState:
    repository_exists = _git(root, "rev-parse", "--is-inside-work-tree").returncode == 0
    if not repository_exists:
        return GitState(False, None, None, False, [], [])
    branch_result = _git(root, "branch", "--show-current")
    branch = branch_result.stdout.strip() or None
    commit_result = _git(root, "rev-parse", "HEAD")
    commit_hash = (
        commit_result.stdout.strip() if commit_result.returncode == 0 else None
    )
    status_result = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    status_entries = [
        line for line in status_result.stdout.splitlines() if line.strip()
    ]
    large_files: list[dict[str, Any]] = []
    threshold = int(large_threshold_mb * 1024 * 1024)
    for entry in status_entries:
        if not entry.startswith("?? "):
            continue
        relative = entry[3:]
        path = root / relative
        if path.is_file() and path.stat().st_size >= threshold:
            large_files.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "git_lfs": _lfs_status(root, relative),
                }
            )
    return GitState(
        repository_exists=True,
        branch=branch,
        commit_hash=commit_hash,
        clean=not status_entries,
        status_entries=status_entries,
        large_untracked_files=large_files,
    )


def collect_environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    gpu: dict[str, Any] = {
        "available": cuda,
        "name": None,
        "total_memory_bytes": 0,
        "free_memory_bytes": 0,
        "mixed_precision_fp16": False,
        "mixed_precision_bf16": False,
    }
    if cuda:
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(total_memory),
                "free_memory_bytes": int(free_memory),
                "mixed_precision_fp16": True,
                "mixed_precision_bf16": bool(torch.cuda.is_bf16_supported()),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    process = psutil.Process()
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda": gpu,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "system_memory_bytes": psutil.virtual_memory().total,
        "process_rss_bytes": process.memory_info().rss,
    }

