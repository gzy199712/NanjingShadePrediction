"""Logging and atomic report-writing helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def configure_logging(log_path: Path, overwrite: bool) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and not overwrite:
        raise FileExistsError(
            f"log exists; pass --overwrite to replace Phase A outputs: {log_path}"
        )
    logger = logging.getLogger("training_preflight")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    frame = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)

