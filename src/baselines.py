"""Simple probabilistic baselines for demand forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data import PipelineSpec, get_pipeline_spec
except ImportError:
    from data import PipelineSpec, get_pipeline_spec


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


def run_baselines_for_pipeline(
    pipeline: str,
    data_dir: Path,
    results_dir: Path,
) -> list[Path]:
    """Execute seasonal naive and moving average baselines and save prediction parquets."""
    spec = get_pipeline_spec(pipeline)
    clean_path = data_dir / spec.clean_filename
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {clean_path}. "
            f"Run data cleaning first to generate {spec.clean_filename}."
        )

    df = pd.read_parquet(clean_path)
    df = _canonical_frame(df, spec)
    df = df.sort_values([*spec.series_id_cols, "Date"]).reset_index(drop=True)

    max_date = df["Date"].max()
    holdout_start = max_date - pd.Timedelta(days=spec.holdout_days - 1)

    history = df[df["Date"] < holdout_start].copy()
    forecast = df[df["Date"] >= holdout_start].copy()

    results_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # 1. Seasonal Naive (7-day lag)
    sn7_preds = seasonal_naive(history, forecast, spec, season_length=7)
    if spec.name == "rossmann" and "Open" in forecast.columns:
        closed_mask = (forecast["Open"].to_numpy() == 0)
        sn7_preds.loc[closed_mask, ["p10", "p50", "p90"]] = 0.0

    sn7_path = results_dir / f"predictions_{pipeline.upper()}_seasonal_naive_7.parquet"
    sn7_preds.to_parquet(sn7_path, index=False)
    saved_paths.append(sn7_path)

    # 2. Moving Average (28-day window)
    ma28_preds = moving_average_28(history, forecast, spec, window=28)
    if spec.name == "rossmann" and "Open" in forecast.columns:
        closed_mask = (forecast["Open"].to_numpy() == 0)
        ma28_preds.loc[closed_mask, ["p10", "p50", "p90"]] = 0.0

    ma28_path = results_dir / f"predictions_{pipeline.upper()}_moving_average_28.parquet"
    ma28_preds.to_parquet(ma28_path, index=False)
    saved_paths.append(ma28_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline forecasting models.")
    parser.add_argument(
        "--pipeline",
        choices=("A", "B", "all"),
        required=True,
        help="Pipeline identifier: 'A' (Rossmann), 'B' (Store Item), or 'all'",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing clean processed parquet files",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save baseline predictions",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir if not args.data_dir.is_absolute() else args.data_dir
    results_dir = root / args.results_dir if not args.results_dir.is_absolute() else args.results_dir

    pipelines = ("A", "B") if args.pipeline == "all" else (args.pipeline,)
    for pipe in pipelines:
        print(f"Generating baseline predictions for Pipeline {pipe}...")
        outputs = run_baselines_for_pipeline(pipe, data_dir, results_dir)
        for output in outputs:
            print(f"  Saved: {output.name}")


if __name__ == "__main__":
    main()
