"""Origin-known simulated stock and stockout-alert labeling."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import (
    EvaluationSpec,
    canonicalize_data_columns,
    canonicalize_prediction_columns,
)


def compute_origin_known_stock(
    raw_sales: pd.DataFrame,
    forecast_rows: pd.DataFrame,
    spec: EvaluationSpec,
    horizon: int,
    window: int = 28,
) -> pd.Series:
    """Compute trailing stock basis from sales available no later than t-h."""
    if horizon < 1 or window < 1:
        raise ValueError("horizon and window must be positive.")
    raw = canonicalize_data_columns(raw_sales, spec)
    forecast = canonicalize_prediction_columns(forecast_rows, spec)
    raw = raw.sort_values([*spec.series_id_columns, "date"]).copy()
    grouped = raw.groupby(list(spec.series_id_columns), sort=False)["actual"]
    raw["_origin_mean"] = (
        grouped.shift(horizon)
        .groupby(
            [raw[column] for column in spec.series_id_columns],
            sort=False,
        )
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=list(range(len(spec.series_id_columns))), drop=True)
    )
    lookup = raw[[*spec.series_id_columns, "date", "_origin_mean"]]
    joined = forecast[[*spec.series_id_columns, "date"]].merge(
        lookup,
        on=[*spec.series_id_columns, "date"],
        how="left",
        validate="one_to_one",
    )
    if joined["_origin_mean"].isna().any():
        missing = int(joined["_origin_mean"].isna().sum())
        raise ValueError(
            f"Could not compute origin-known stock for {missing} forecast rows."
        )
    return pd.Series(joined["_origin_mean"].to_numpy(dtype=float), index=forecast.index)


def choose_stock_multiplier(
    validation_predictions: Mapping[str, pd.DataFrame] | None,
    validation_sales: pd.DataFrame,
    spec: EvaluationSpec,
    horizon: int,
    target_positive_range: tuple[float, float] = (0.10, 0.20),
) -> float:
    """Choose one simulated-stock multiplier using validation data only."""
    if not validation_predictions:
        raise FileNotFoundError(
            "Validation prediction artifacts are required to choose stock multiplier k; "
            "test predictions cannot be used for this choice."
        )
    low, high = target_positive_range
    if not 0 <= low <= high <= 1:
        raise ValueError("target_positive_range must be within [0, 1].")
    reference = next(iter(validation_predictions.values()))
    validation = canonicalize_prediction_columns(reference, spec)
    sales = canonicalize_data_columns(validation_sales, spec)
    joined = validation[
        [*spec.series_id_columns, "date", "actual"]
    ].merge(
        sales[[*spec.series_id_columns, "date", "actual"]],
        on=[*spec.series_id_columns, "date"],
        how="left",
        suffixes=("_prediction", "_source"),
        validate="one_to_one",
    )
    if joined["actual_source"].isna().any():
        raise ValueError("Validation predictions do not match source sales rows.")
    stock_basis = compute_origin_known_stock(sales, validation, spec, horizon)
    actual = joined["actual_source"].to_numpy(dtype=float)
    basis = stock_basis.to_numpy(dtype=float)
    candidates = np.linspace(0.1, 3.0, 117)
    rates = np.array([(actual > multiplier * basis).mean() for multiplier in candidates])
    in_range = (rates >= low) & (rates <= high)
    if in_range.any():
        midpoint = (low + high) / 2
        selected = np.flatnonzero(in_range)[
            np.argmin(np.abs(rates[in_range] - midpoint))
        ]
        return float(candidates[selected])
    return float(candidates[np.argmin(np.minimum(np.abs(rates - low), np.abs(rates - high)))])


def apply_stockout_labels(
    predictions: pd.DataFrame,
    stock_level: pd.Series,
    multiplier: float,
) -> pd.DataFrame:
    """Add simulated stockout labels, alerts, and continuous alert scores."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")
    result = predictions.copy().reset_index(drop=True)
    stock = pd.Series(stock_level).reset_index(drop=True).astype(float)
    if len(result) != len(stock):
        raise ValueError("stock_level must have one value per prediction row.")
    result["stock_level"] = multiplier * stock
    result["stockout"] = (result["actual"] > result["stock_level"]).astype(int)
    result["alert"] = (result["p90"] > result["stock_level"]).astype(int)
    denominator = result["stock_level"].to_numpy(dtype=float)
    numerator = result["p90"].to_numpy(dtype=float)
    result["alert_score"] = np.divide(
        numerator,
        denominator,
        out=np.full(len(result), np.inf),
        where=denominator > 0,
    )
    return result
