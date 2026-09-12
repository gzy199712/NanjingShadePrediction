"""Validation records and numeric comparison helpers for Phase F."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


@dataclass
class PhaseFCheck:
    category: str
    check: str
    status: str
    severity: str
    expected: Any
    actual: Any
    details: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhaseFRecorder:
    def __init__(self) -> None:
        self.rows: list[PhaseFCheck] = []

    def add(
        self,
        category: str,
        check: str,
        passed: bool,
        *,
        expected: Any,
        actual: Any,
        details: str = "",
        blocker: bool = True,
    ) -> None:
        self.rows.append(
            PhaseFCheck(
                category=category,
                check=check,
                status="PASS" if passed else ("FAIL" if blocker else "WARN"),
                severity="none" if passed else ("blocker" if blocker else "warning"),
                expected=expected,
                actual=actual,
                details=details,
            )
        )

    @property
    def blocker_count(self) -> int:
        return sum(row.severity == "blocker" for row in self.rows)

    @property
    def warning_count(self) -> int:
        return sum(row.severity == "warning" for row in self.rows)

    def as_rows(self) -> list[dict[str, Any]]:
        return [row.as_dict() for row in self.rows]


def sha256_file(path: Path, chunk_mb: int, description: str) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    chunk_size = chunk_mb * 1024 * 1024
    with path.open("rb") as handle, tqdm(
        total=total,
        desc=description,
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            progress.update(len(chunk))
    return digest.hexdigest()


def numeric_comparison(
    reference: np.ndarray,
    reproduced: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    reproduced = np.asarray(reproduced, dtype=np.float64)
    if reference.shape != reproduced.shape:
        raise ValueError(
            f"comparison shape mismatch: {reference.shape} != {reproduced.shape}"
        )
    difference = np.abs(reproduced - reference)
    flat_reference = reference.reshape(-1)
    flat_reproduced = reproduced.reshape(-1)
    denominator = float(
        np.linalg.norm(flat_reference) * np.linalg.norm(flat_reproduced)
    )
    cosine = (
        float(np.dot(flat_reference, flat_reproduced) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {
        "value_count": int(reference.size),
        "mae_difference": float(difference.mean()),
        "max_absolute_difference": float(difference.max(initial=0.0)),
        "cosine_similarity": cosine,
        "exact_value_count": int(np.equal(reference, reproduced).sum()),
        "within_tolerance_count": int((difference <= tolerance).sum()),
        "outside_tolerance_count": int((difference > tolerance).sum()),
        "tolerance": tolerance,
        "finite_reference": bool(np.isfinite(reference).all()),
        "finite_reproduced": bool(np.isfinite(reproduced).all()),
    }
