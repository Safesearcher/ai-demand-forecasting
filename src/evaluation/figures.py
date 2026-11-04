"""Publication-quality figures for Phase 4 evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EvaluationSpec, canonicalize_prediction_columns


FIGURE_DPI = 300


def plot_loss_curves(curve_paths: list[Path], output_dir: Path) -> list[Path]:
    """Plot train and validation loss curves from Phase 3 CSV artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for curve_path in curve_paths:
        curves = pd.read_csv(curve_path)
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(curves["epoch"], curves["train_loss"], label="Train")
        axis.plot(curves["epoch"], curves["validation_loss"], label="Validation")
        axis.set(
            xlabel="Epoch",
            ylabel="Pinball loss",
            title=curve_path.stem.replace("_", " "),
        )
        axis.legend()
        figure.tight_layout()
        path = output_dir / f"{curve_path.stem}.png"
        figure.savefig(path, dpi=FIGURE_DPI)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_prediction_bands(
    predictions: pd.DataFrame,
    spec: EvaluationSpec,
    output_dir: Path,
    n_series: int = 3,
) -> list[Path]:
    """Plot actuals and P10-P90 bands for representative test series."""
    if n_series < 1:
        raise ValueError("n_series must be positive.")
    frame = canonicalize_prediction_columns(predictions, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    series_keys = frame[list(spec.series_id_columns)].drop_duplicates().head(n_series)
    paths = []
    for index, key_row in series_keys.iterrows():
        mask = np.ones(len(frame), dtype=bool)
        for column in spec.series_id_columns:
            mask &= frame[column].to_numpy() == key_row[column]
        group = frame.loc[mask].sort_values("date")
        label = "_".join(str(key_row[column]) for column in spec.series_id_columns)
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(group["date"], group["actual"], label="Actual", color="black")
        axis.plot(group["date"], group["p50"], label="P50", color="tab:blue")
        axis.fill_between(
            group["date"],
            group["p10"].to_numpy(),
            group["p90"].to_numpy(),
            alpha=0.25,
            label="P10-P90",
        )
        axis.set(title=f"{spec.pipeline} series {label}", xlabel="Date", ylabel="Sales")
        axis.legend()
        figure.tight_layout()
        path = output_dir / f"prediction_band_{index}_{label}.png"
        figure.savefig(path, dpi=FIGURE_DPI)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_coverage_calibration(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot observed P10-P90 coverage against the nominal 80 percent."""
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(metrics["model"], metrics["coverage_p10_p90"])
    axis.axhline(0.8, color="black", linestyle="--", label="Nominal 80%")
    axis.set(xlabel="Model", ylabel="Empirical coverage", title="Prediction interval calibration")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_residual_histogram(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot P50 residual distributions by model."""
    figure, axis = plt.subplots(figsize=(7, 4))
    for model, group in predictions.groupby("model"):
        axis.hist(group["p50"] - group["actual"], bins=40, alpha=0.5, label=model)
    axis.set(xlabel="P50 minus actual", ylabel="Frequency", title="Residual distribution")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot WMAPE and interval coverage for available models."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(metrics["model"], metrics["wmape"])
    axes[0].set(title="WMAPE", ylabel="WMAPE")
    axes[1].bar(metrics["model"], metrics["coverage_p10_p90"])
    axes[1].axhline(0.8, color="black", linestyle="--")
    axes[1].set(title="P10-P90 coverage", ylabel="Coverage")
    figure.tight_layout()
    _save_figure(figure, output_path)


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
