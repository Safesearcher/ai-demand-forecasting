"""Subgroup error, calibration, bias, and residual analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EvaluationSpec, canonicalize_data_columns, canonicalize_prediction_columns


def add_volume_tiers(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    spec: EvaluationSpec,
) -> pd.DataFrame:
    """Assign low/medium/high tiers from training-period series mean sales."""
    train = canonicalize_data_columns(training_data, spec)
    result = canonicalize_data_columns(evaluation_data, spec)
    means = train.groupby(list(spec.series_id_columns), as_index=False)["actual"].mean()
    means = means.rename(columns={"actual": "_series_mean"})
    thresholds = means["_series_mean"].quantile([1 / 3, 2 / 3]).to_numpy()
    tier_column = "item_volume_tier" if spec.pipeline == "B" else "volume_tier"
    means[tier_column] = pd.cut(
        means["_series_mean"],
        bins=[-np.inf, *thresholds, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return result.merge(
        means[[*spec.series_id_columns, tier_column]],
        on=list(spec.series_id_columns),
        how="left",
        validate="many_to_one",
    )


def calculate_subgroup_errors(
    predictions: pd.DataFrame,
    spec: EvaluationSpec,
    subgroup_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Calculate WMAPE, bias, coverage, and alert recall per subgroup."""
    frame = canonicalize_prediction_columns(predictions, spec)
    rows = []
    for subgroup in subgroup_columns:
        if subgroup not in frame.columns:
            raise ValueError(f"Subgroup column {subgroup!r} is missing from predictions.")
        for value, group in frame.groupby(subgroup, dropna=False):
            actual = group["actual"].to_numpy(dtype=float)
            p50 = group["p50"].to_numpy(dtype=float)
            denominator = np.abs(actual).sum()
            coverage = ((actual >= group["p10"]) & (actual <= group["p90"])).mean()
            recall = np.nan
            if "stockout" in group and group["stockout"].sum() > 0:
                recall = (group.loc[group["stockout"] == 1, "alert"] == 1).mean()
            rows.append(
                {
                    "subgroup": subgroup,
                    "value": str(value),
                    "rows": len(group),
                    "wmape": np.abs(actual - p50).sum() / denominator
                    if denominator
                    else np.nan,
                    "bias": (p50 - actual).mean(),
                    "coverage": coverage,
                    "stockout_alert_recall": recall,
                    "coverage_flag": bool(coverage < 0.70 or coverage > 0.90),
                }
            )
    return pd.DataFrame(rows)


def calculate_bias_gaps(subgroup_metrics: pd.DataFrame) -> pd.DataFrame:
    """Report best/worst WMAPE, gap, and directional bias per subgroup dimension."""
    rows = []
    for subgroup, group in subgroup_metrics.groupby("subgroup"):
        valid = group.dropna(subset=["wmape"])
        if valid.empty:
            continue
        best = valid.loc[valid["wmape"].idxmin()]
        worst = valid.loc[valid["wmape"].idxmax()]
        rows.append(
            {
                "subgroup": subgroup,
                "best_value": best["value"],
                "worst_value": worst["value"],
                "best_wmape": best["wmape"],
                "worst_wmape": worst["wmape"],
                "wmape_gap": worst["wmape"] - best["wmape"],
                "best_bias": best["bias"],
                "worst_bias": worst["bias"],
                "worst_direction": "over" if worst["bias"] > 0 else "under",
            }
        )
    return pd.DataFrame(rows)


def find_worst_series(
    predictions: pd.DataFrame,
    spec: EvaluationSpec,
    limit: int = 10,
) -> pd.DataFrame:
    """Return series with highest P50 WMAPE."""
    frame = canonicalize_prediction_columns(predictions, spec)
    rows = []
    for keys, group in frame.groupby(list(spec.series_id_columns)):
        actual = group["actual"].to_numpy(dtype=float)
        denominator = np.abs(actual).sum()
        rows.append(
            {
                **dict(zip(spec.series_id_columns, keys if isinstance(keys, tuple) else (keys,))),
                "wmape": np.abs(actual - group["p50"]).sum() / denominator
                if denominator
                else np.nan,
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("wmape", ascending=False).head(limit)


def find_worst_dates(predictions: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return dates with highest aggregate P50 absolute error."""
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    grouped = frame.groupby("date").apply(
        lambda group: pd.Series(
            {
                "wmape": np.abs(group["actual"] - group["p50"]).sum()
                / np.abs(group["actual"]).sum()
                if np.abs(group["actual"]).sum()
                else np.nan,
                "bias": (group["p50"] - group["actual"]).mean(),
            }
        ),
        include_groups=False,
    )
    return grouped.reset_index().sort_values("wmape", ascending=False).head(limit)


def plot_subgroup_errors(subgroup_metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save WMAPE bar charts for every subgroup dimension."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for subgroup, group in subgroup_metrics.groupby("subgroup"):
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(group["value"], group["wmape"])
        axis.set(xlabel=subgroup, ylabel="WMAPE", title=f"WMAPE by {subgroup}")
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        path = output_dir / f"subgroup_wmape_{subgroup}.png"
        figure.savefig(path, dpi=300)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_residuals_over_time(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot daily mean signed residuals over the evaluation window."""
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["residual"] = frame["p50"] - frame["actual"]
    daily = frame.groupby("date", as_index=False)["residual"].mean()
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(daily["date"], daily["residual"])
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Date", ylabel="P50 minus actual", title="Residuals over time")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
