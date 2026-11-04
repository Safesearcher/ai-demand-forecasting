"""Data loading and leakage-safe feature preparation for Phase 3 models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # Keep feature preparation usable before ML dependencies install.
    torch = None
    DataLoader = Any
    Dataset = object


QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class PipelineSpec:
    """Names and split rules for one independent forecasting pipeline."""

    name: str
    clean_filename: str
    date_col: str
    target_col: str
    series_id_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...]
    holdout_days: int


@dataclass
class PreparedData:
    """Prepared train, validation, and untouched Deliverable 1 holdout data."""

    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame
    categorical_columns: list[str]
    numeric_columns: list[str]
    categorical_sizes: dict[str, int]
    encoders: dict[str, dict[Any, int]]
    scaler: StandardScaler
    spec: PipelineSpec


class ForecastDataset(Dataset):
    """Torch dataset returning categorical IDs, numeric features, and log target."""

    def __init__(
        self,
        frame: pd.DataFrame,
        spec: PipelineSpec,
        categorical_columns: list[str],
        numeric_columns: list[str],
    ) -> None:
        _require_torch()
        self.categorical_ids = torch.as_tensor(
            frame[categorical_columns].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
        self.numeric_features = torch.as_tensor(
            frame[numeric_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.target = torch.as_tensor(
            np.log1p(frame[spec.target_col].to_numpy(dtype=np.float32)),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.categorical_ids[index],
            self.numeric_features[index],
            self.target[index],
        )


def get_pipeline_spec(pipeline: str) -> PipelineSpec:
    """Return the canonical schema and holdout rule for pipeline A or B."""
    normalized = pipeline.upper()
    if normalized == "A":
        return PipelineSpec(
            name="rossmann",
            clean_filename="rossmann_clean.parquet",
            date_col="Date",
            target_col="Sales",
            series_id_cols=("Store",),
            categorical_cols=("StateHoliday", "StoreType", "Assortment", "PromoInterval"),
            holdout_days=42,
        )
    if normalized == "B":
        return PipelineSpec(
            name="store_item",
            clean_filename="store_item_clean.parquet",
            date_col="Date",
            target_col="Sales",
            series_id_cols=("store", "item"),
            categorical_cols=(),
            holdout_days=90,
        )
    raise ValueError(f"Unsupported pipeline {pipeline!r}; expected 'A' or 'B'.")


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for ForecastDataset and DataLoader operations. "
            "Install the pinned dependencies from requirements.txt."
        )


def _temporal_features(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    result = frame.copy()
    dates = result[date_col]
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    result["month"] = dates.dt.month
    result["year"] = dates.dt.year
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    return result


def _history_features(
    frame: pd.DataFrame,
    spec: PipelineSpec,
    horizon: int,
) -> pd.DataFrame:
    """Build lag and rolling features using observations no later than t-h."""
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    result = frame.copy()
    if spec.name == "store_item":
        result = result.rename(columns={"date": "Date", "sales": "Sales"})
        result["Date"] = pd.to_datetime(result["Date"])
    result = _temporal_features(result, "Date")
    result = result.sort_values([*spec.series_id_cols, "Date"]).reset_index(drop=True)

    grouped = result.groupby(list(spec.series_id_cols), sort=False)["Sales"]
    for lag in (1, 7, 14, 28):
        if lag >= horizon:
            result[f"Sales_lag_{lag}"] = grouped.shift(lag)

    for window in (7, 14, 28):
        shifted = grouped.shift(horizon)
        result[f"Sales_roll_mean_{window}"] = (
            shifted.groupby(
                [result[column] for column in spec.series_id_cols],
                sort=False,
            )
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(spec.series_id_cols))), drop=True)
        )
        result[f"Sales_roll_std_{window}"] = (
            shifted.groupby(
                [result[column] for column in spec.series_id_cols],
                sort=False,
            )
            .rolling(window, min_periods=1)
            .std()
            .fillna(0)
            .reset_index(level=list(range(len(spec.series_id_cols))), drop=True)
        )

    if spec.name == "rossmann":
        for lag in (7, 14):
            if lag >= horizon:
                customer_grouped = result.groupby(
                    list(spec.series_id_cols), sort=False
                )["Customers"]
                result[f"Customers_lag_{lag}"] = customer_grouped.shift(lag)
        customer_grouped = result.groupby(list(spec.series_id_cols), sort=False)[
            "Customers"
        ]
        for window in (7, 14):
            shifted = customer_grouped.shift(horizon)
            result[f"Customers_roll_mean_{window}"] = (
                shifted.groupby(
                    [result[column] for column in spec.series_id_cols],
                    sort=False,
                )
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=list(range(len(spec.series_id_cols))), drop=True)
            )
            result[f"Customers_roll_std_{window}"] = (
                shifted.groupby(
                    [result[column] for column in spec.series_id_cols],
                    sort=False,
                )
                .rolling(window, min_periods=1)
                .std()
                .fillna(0)
                .reset_index(level=list(range(len(spec.series_id_cols))), drop=True)
            )
        result = result.drop(columns=["Customers"], errors="ignore")

    return result


def rebuild_horizon_features(
    frame: pd.DataFrame,
    spec: PipelineSpec,
    horizon: int,
) -> pd.DataFrame:
    """Public wrapper for rebuilding features without changing source files."""
    return _history_features(frame, spec, horizon)


def _encode_categories(
    train: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    columns: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, dict[Any, int]]]:
    encoders: dict[str, dict[Any, int]] = {}
    train_result = train.copy()
    other_results = [frame.copy() for frame in other_frames]
    for column in columns:
        values = pd.Series(train_result[column].astype(str).unique())
        mapping = {value: index + 1 for index, value in enumerate(sorted(values))}
        mapping["__unknown__"] = 0
        encoders[column] = mapping
        train_result[column] = (
            train_result[column].astype(str).map(mapping).fillna(0).astype(np.int64)
        )
        for index, frame in enumerate(other_results):
            other_results[index][column] = (
                frame[column]
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype(np.int64)
            )
    return train_result, other_results, encoders


def _candidate_numeric_columns(
    frame: pd.DataFrame,
    spec: PipelineSpec,
    horizon: int,
) -> list[str]:
    excluded = {
        spec.date_col,
        spec.target_col,
        "Date",
        "Sales",
        *spec.series_id_cols,
        *spec.categorical_cols,
        "Open",
        "is_closed",
    }
    columns = [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return [
        column
        for column in columns
        if not column.startswith("Sales_lag_")
        or int(column.rsplit("_", 1)[-1]) >= horizon
    ]


def prepare_data(
    pipeline: str,
    horizon: int,
    data_dir: Path,
    config: dict[str, Any] | None = None,
) -> PreparedData:
    """Load, rebuild, split, encode, and scale one pipeline without leakage."""
    del config
    spec = get_pipeline_spec(pipeline)
    source = pd.read_parquet(data_dir / spec.clean_filename)
    combined = _history_features(source, spec, horizon)
    combined = combined.sort_values("Date").reset_index(drop=True)

    max_date = combined["Date"].max()
    holdout_start = max_date - pd.Timedelta(days=spec.holdout_days - 1)
    holdout = combined[combined["Date"] >= holdout_start].copy()
    development = combined[combined["Date"] < holdout_start].copy()
    validation_start = development["Date"].max() - pd.Timedelta(
        days=spec.holdout_days - 1
    )
    validation = development[development["Date"] >= validation_start].copy()
    train = development[development["Date"] < validation_start].copy()

    if spec.name == "rossmann":
        train = train[train["Open"] == 1].copy()
        validation = validation[validation["Open"] == 1].copy()

    categorical_columns = [*spec.series_id_cols, *spec.categorical_cols]
    train, [validation, holdout], encoders = _encode_categories(
        train, [validation, holdout], categorical_columns
    )
    numeric_columns = _candidate_numeric_columns(train, spec, horizon)
    scaler = StandardScaler()
    train[numeric_columns] = scaler.fit_transform(train[numeric_columns].fillna(0))
    validation[numeric_columns] = scaler.transform(
        validation[numeric_columns].fillna(0)
    )
    holdout[numeric_columns] = scaler.transform(holdout[numeric_columns].fillna(0))

    return PreparedData(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        holdout=holdout.reset_index(drop=True),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        categorical_sizes={
            column: len(mapping) for column, mapping in encoders.items()
        },
        encoders=encoders,
        scaler=scaler,
        spec=spec,
    )


def create_dataloader(
    frame: pd.DataFrame,
    prepared: PreparedData,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create a deterministic-platform-safe loader; Windows uses zero workers."""
    _require_torch()
    if num_workers != 0:
        raise ValueError("num_workers must be 0 for this Windows project.")
    dataset = ForecastDataset(
        frame,
        prepared.spec,
        prepared.categorical_columns,
        prepared.numeric_columns,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
