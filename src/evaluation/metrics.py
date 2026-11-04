"""Regression and probabilistic forecast metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import EvaluationSpec, canonicalize_prediction_columns


def pinball_metric(
    actual: np.ndarray,
    prediction: np.ndarray,
    quantile: float,
) -> float:
    """Return mean pinball loss for one quantile."""
    errors = actual - prediction
    return float(np.maximum(quantile * errors, (quantile - 1) * errors).mean())


def calculate_regression_metrics(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """Calculate point, quantile, interval, and bias metrics."""
    actual = predictions["actual"].to_numpy(dtype=float)
    p10 = predictions["p10"].to_numpy(dtype=float)
    p50 = predictions["p50"].to_numpy(dtype=float)
    p90 = predictions["p90"].to_numpy(dtype=float)
    denominator = np.abs(actual).sum()
    return {
        "mae": float(np.abs(actual - p50).mean()),
        "rmse": float(np.sqrt(np.mean((actual - p50) ** 2))),
        "wmape": float(np.abs(actual - p50).sum() / denominator)
        if denominator
        else np.nan,
        "bias": float((p50 - actual).mean()),
        "pinball_p10": pinball_metric(actual, p10, 0.1),
        "pinball_p50": pinball_metric(actual, p50, 0.5),
        "pinball_p90": pinball_metric(actual, p90, 0.9),
        "coverage_p10_p90": float(((actual >= p10) & (actual <= p90)).mean()),
        "mean_interval_width": float((p90 - p10).mean()),
    }


def evaluate_models(
    prediction_frames: dict[str, pd.DataFrame],
    spec: EvaluationSpec,
) -> pd.DataFrame:
    """Calculate metrics for each available model on one pipeline's test data."""
    rows: list[dict[str, Any]] = []
    for model, frame in prediction_frames.items():
        canonical = canonicalize_prediction_columns(frame, spec)
        rows.append({"pipeline": spec.pipeline, "model": model, **calculate_regression_metrics(canonical)})
    return pd.DataFrame(rows)


def save_metrics_csv(metrics: pd.DataFrame, output_path: Path) -> None:
    """Write metrics as a CSV table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)


def save_metrics_latex(
    metrics: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
) -> None:
    """Write a booktabs LaTeX table suitable for ``\\input``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(metrics.columns)
    body = metrics.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.4f}",
        column_format="ll" + "r" * max(0, len(columns) - 2),
        caption=caption,
        label=label,
    )
    output_path.write_text(body, encoding="utf-8")
