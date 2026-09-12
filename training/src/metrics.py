"""Validation metrics used for model selection without accessing Test data."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(
    truth: np.ndarray, prediction: np.ndarray, prefix: str
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"metric shape mismatch: {truth.shape} != {prediction.shape}")
    residual = prediction - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    bias = float(np.mean(residual))
    denominator = float(np.sum(np.square(truth - truth.mean())))
    r2 = (
        float(1.0 - np.sum(np.square(residual)) / denominator)
        if denominator > 0
        else float("nan")
    )
    correlation = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if len(truth) > 1 and truth.std() > 0 and prediction.std() > 0
        else float("nan")
    )
    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2": r2,
        f"{prefix}_bias": bias,
        f"{prefix}_pearson": correlation,
    }

