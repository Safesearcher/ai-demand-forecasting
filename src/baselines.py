"""Simple probabilistic baselines for demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .data import PipelineSpec
except ImportError:
    from data import PipelineSpec


def _canonical_frame(frame: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Return a copy using the canonical Date and Sales column names."""
    result = frame.copy()
    if spec.name == "store_item":
        result = result.rename(columns={"date": "Date", "sales": "Sales"})
    result["Date"] = pd.to_datetime(result["Date"])
    return result


def _prediction_frame(
    forecast: pd.DataFrame,
    predictions: np.ndarray,
    spec: PipelineSpec,
    model_name: str,
) -> pd.DataFrame:
    if predictions.ndim != 2 or predictions.shape != (len(forecast), 3):
        raise ValueError("predictions must have shape [len(forecast), 3].")
    result = forecast[[*spec.series_id_cols, "Date", "Sales"]].copy()
    result = result.rename(columns={"Date": "date", "Sales": "actual"})
    result[["p10", "p50", "p90"]] = np.clip(predictions, 0, None)
    result["p10"] = np.minimum(result["p10"], result["p50"])
    result["p90"] = np.maximum(result["p90"], result["p50"])
    result["model"] = model_name
    return result[
        ["date", *spec.series_id_cols, "actual", "p10", "p50", "p90", "model"]
    ]


def format_prediction_frame(
    forecast: pd.DataFrame,
    predictions: np.ndarray,
    spec: PipelineSpec,
    model_name: str,
) -> pd.DataFrame:
    """Format original-unit quantile predictions for result Parquet files."""
    canonical = _canonical_frame(forecast, spec)
    return _prediction_frame(canonical, predictions, spec, model_name)


def seasonal_naive(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    spec: PipelineSpec,
    season_length: int = 7,
) -> pd.DataFrame:
    """Use the value from the same series exactly ``season_length`` days earlier."""
    if season_length < 1:
        raise ValueError("season_length must be positive.")
    history = _canonical_frame(history, spec)
    forecast = _canonical_frame(forecast, spec)
    lookup = history[[*spec.series_id_cols, "Date", "Sales"]].rename(
        columns={"Date": "_source_date", "Sales": "_prediction"}
    )
    forecast = forecast.copy()
    forecast["_source_date"] = forecast["Date"] - pd.Timedelta(days=season_length)
    merged = forecast.merge(
        lookup,
        on=[*spec.series_id_cols, "_source_date"],
        how="left",
        validate="one_to_one",
    )
    predictions = merged["_prediction"].fillna(0).to_numpy()[:, None]
    return _prediction_frame(
        forecast,
        np.repeat(predictions, 3, axis=1),
        spec,
        f"seasonal_naive_{season_length}",
    )


def moving_average_28(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    spec: PipelineSpec,
    window: int = 28,
) -> pd.DataFrame:
    """Use the trailing mean of prior observations from the same series."""
    if window < 1:
        raise ValueError("window must be positive.")
    history = _canonical_frame(history, spec)
    forecast = _canonical_frame(forecast, spec)
    combined = pd.concat(
        [
            history[[*spec.series_id_cols, "Date", "Sales"]],
            forecast[[*spec.series_id_cols, "Date"]].assign(Sales=np.nan),
        ],
        ignore_index=True,
    ).sort_values([*spec.series_id_cols, "Date"])
    grouped = combined.groupby(list(spec.series_id_cols), sort=False)["Sales"]
    combined["_prediction"] = grouped.transform(
        lambda values: values.shift(1).rolling(window, min_periods=1).mean()
    )
    predictions = (
        forecast[[*spec.series_id_cols, "Date"]]
        .merge(
            combined[[*spec.series_id_cols, "Date", "_prediction"]],
            on=[*spec.series_id_cols, "Date"],
            how="left",
            validate="one_to_one",
        )["_prediction"]
        .fillna(0)
        .to_numpy()[:, None]
    )
    return _prediction_frame(
        forecast,
        np.repeat(predictions, 3, axis=1),
        spec,
        f"moving_average_{window}",
    )
