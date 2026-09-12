"""Reusable QC records and numeric validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class QCIssue:
    category: str
    check: str
    status: str
    severity: str
    expected: Any
    actual: Any
    details: str = ""
    sample_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditRecorder:
    """Collect PASS/WARN/FAIL checks without mutating source data."""

    def __init__(self) -> None:
        self.rows: list[QCIssue] = []

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
        sample_id: str = "",
    ) -> None:
        status = "PASS" if passed else ("FAIL" if blocker else "WARN")
        severity = "none" if passed else ("blocker" if blocker else "warning")
        self.rows.append(
            QCIssue(
                category=category,
                check=check,
                status=status,
                severity=severity,
                expected=expected,
                actual=actual,
                details=details,
                sample_id=sample_id,
            )
        )

    @property
    def blocker_count(self) -> int:
        return sum(row.status == "FAIL" for row in self.rows)

    @property
    def warning_count(self) -> int:
        return sum(row.status == "WARN" for row in self.rows)

    def category(self, name: str) -> list[dict[str, Any]]:
        return [row.as_dict() for row in self.rows if row.category == name]

    def failures(self) -> list[dict[str, Any]]:
        return [row.as_dict() for row in self.rows if row.status != "PASS"]


def finite_counts(values: np.ndarray) -> tuple[int, int]:
    return int(np.isnan(values).sum()), int(np.isinf(values).sum())


def numeric_profile(
    frame: pd.DataFrame,
    fields: Iterable[str],
    near_zero_variance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            quantiles = np.quantile(finite, [0.01, 0.05, 0.5, 0.95, 0.99])
            minimum = float(finite.min())
            maximum = float(finite.max())
            mean = float(finite.mean())
            std = float(finite.std(ddof=0))
        else:
            quantiles = np.full(5, np.nan)
            minimum = maximum = mean = std = float("nan")
        rows.append(
            {
                "field": field,
                "dtype": str(frame[field].dtype),
                "row_count": len(values),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
                "minimum": minimum,
                "p01": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "median": float(quantiles[2]),
                "p95": float(quantiles[3]),
                "p99": float(quantiles[4]),
                "maximum": maximum,
                "mean": mean,
                "std": std,
                "constant": bool(np.isfinite(std) and std == 0.0),
                "near_zero_variance": bool(
                    np.isfinite(std) and std * std <= near_zero_variance
                ),
            }
        )
    return rows


def index_array(
    series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    integer_mask = np.isfinite(numeric) & (numeric == np.floor(numeric))
    safe = np.zeros(len(numeric), dtype=np.int64)
    safe[integer_mask] = numeric[integer_mask].astype(np.int64)
    return safe, integer_mask

