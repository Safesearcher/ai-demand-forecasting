"""Classification metrics and figures for simulated stockout alerts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def calculate_classification_metrics(
    labeled_predictions: pd.DataFrame,
) -> dict[str, float | int]:
    """Calculate alert metrics; recall matters most for costly missed stockouts."""
    actual = labeled_predictions["stockout"].to_numpy(dtype=int)
    alert = labeled_predictions["alert"].to_numpy(dtype=int)
    score = labeled_predictions["alert_score"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(actual, alert, labels=[0, 1]).ravel()
    positive_rate = float(actual.mean()) if len(actual) else np.nan
    if len(np.unique(actual)) < 2:
        auc = np.nan
        pr_auc = np.nan
    else:
        auc = float(roc_auc_score(actual, score))
        pr_auc = float(average_precision_score(actual, score))
    return {
        "precision": float(precision_score(actual, alert, zero_division=np.nan)),
        "recall": float(recall_score(actual, alert, zero_division=np.nan)),
        "f1": float(f1_score(actual, alert, zero_division=np.nan)),
        "auc_roc": auc,
        "pr_auc": pr_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "positive_rate": positive_rate,
    }


def evaluate_stockout_alerts(
    labeled_by_model: dict[str, pd.DataFrame],
    pipeline: str,
) -> pd.DataFrame:
    """Evaluate every available model using the same simulated-stock labels."""
    rows = []
    for model, frame in labeled_by_model.items():
        rows.append({"pipeline": pipeline, "model": model, **calculate_classification_metrics(frame)})
    return pd.DataFrame(rows)


def save_classification_latex(
    metrics: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
) -> None:
    """Save classification metrics as a booktabs LaTeX table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_latex(
        output_path,
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.4f}",
        column_format="ll" + "r" * max(0, len(metrics.columns) - 2),
        caption=caption,
        label=label,
    )


def plot_roc_curves(
    labeled_by_model: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Plot ROC curves for models with both stockout classes present."""
    figure, axis = plt.subplots(figsize=(6, 5))
    for model, frame in labeled_by_model.items():
        actual = frame["stockout"].to_numpy(dtype=int)
        if len(np.unique(actual)) < 2:
            continue
        fpr, tpr, _ = roc_curve(actual, frame["alert_score"])
        axis.plot(fpr, tpr, label=f"{model} (AUC={roc_auc_score(actual, frame['alert_score']):.3f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Stockout alert ROC")
    axis.legend()
    _save_figure(figure, output_path)


def plot_precision_recall_curves(
    labeled_by_model: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Plot precision-recall curves for models with positive examples."""
    figure, axis = plt.subplots(figsize=(6, 5))
    for model, frame in labeled_by_model.items():
        actual = frame["stockout"].to_numpy(dtype=int)
        if actual.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(actual, frame["alert_score"])
        axis.plot(recall, precision, label=model)
    axis.set(xlabel="Recall", ylabel="Precision", title="Stockout alert precision-recall")
    axis.legend()
    _save_figure(figure, output_path)


def plot_confusion_matrices(
    labeled_by_model: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Plot one confusion matrix per model."""
    models = list(labeled_by_model)
    figure, axes = plt.subplots(1, max(1, len(models)), figsize=(5 * max(1, len(models)), 4))
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        frame = labeled_by_model[model]
        matrix = confusion_matrix(frame["stockout"], frame["alert"], labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No alert", "Alert"],
            yticklabels=["No stockout", "Stockout"],
            title=model,
            xlabel="Predicted",
            ylabel="Actual",
        )
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    _save_figure(figure, output_path)


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
