"""Shared pipeline configuration and schema normalization for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EvaluationSpec:
    """Evaluation inputs and subgroup definitions for one pipeline."""

    pipeline: str
    prediction_path: Path
    processed_path: Path
    date_column: str
    target_column: str
    series_id_columns: tuple[str, ...]
    subgroup_columns: tuple[str, ...]
    horizon: int
    holdout_days: int
    raw_date_column: str
    raw_target_column: str


def get_evaluation_spec(pipeline: str, root: Path) -> EvaluationSpec:
    """Return file paths and canonical evaluation settings for pipeline A or B."""
    normalized = pipeline.upper()
    if normalized == "A":
        return EvaluationSpec(
            pipeline="A",
            prediction_path=root / "results",
            processed_path=root / "data" / "processed" / "rossmann_clean.parquet",
            date_column="date",
            target_column="actual",
            series_id_columns=("Store",),
            subgroup_columns=(
                "StoreType",
                "Assortment",
                "Promo",
                "StateHoliday",
                "SchoolHoliday",
                "DayOfWeek",
                "volume_tier",
            ),
            horizon=7,
            holdout_days=42,
            raw_date_column="Date",
            raw_target_column="Sales",
        )
    if normalized == "B":
        return EvaluationSpec(
            pipeline="B",
            prediction_path=root / "results",
            processed_path=root / "data" / "processed" / "store_item_clean.parquet",
            date_column="date",
            target_column="actual",
            series_id_columns=("store", "item"),
            subgroup_columns=("store", "item_volume_tier", "day_of_week", "month"),
            horizon=7,
            holdout_days=90,
            raw_date_column="date",
            raw_target_column="sales",
        )
    raise ValueError(f"Unsupported pipeline {pipeline!r}; expected 'A' or 'B'.")


def _column_mapping(frame: pd.DataFrame, spec: EvaluationSpec) -> dict[str, str]:
    """Build a mapping for known case/schema aliases without guessing columns."""
    candidates: dict[str, tuple[str, ...]] = {
        "date": (spec.date_column, spec.raw_date_column, "Date", "date"),
        "actual": (spec.target_column, spec.raw_target_column, "Sales", "sales"),
    }
    for column in spec.series_id_columns:
        candidates[column] = (column, column.lower(), column.upper())
    mapping: dict[str, str] = {}
    for canonical, aliases in candidates.items():
        present = [alias for alias in aliases if alias in frame.columns]
        if present:
            mapping[present[0]] = canonical
    return mapping


def canonicalize_prediction_columns(
    frame: pd.DataFrame,
    spec: EvaluationSpec,
) -> pd.DataFrame:
    """Normalize prediction columns to date, actual, quantiles, and series IDs."""
    result = frame.rename(columns=_column_mapping(frame, spec)).copy()
    required = {
        "date",
        "actual",
        "p10",
        "p50",
        "p90",
        *spec.series_id_columns,
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(
            f"{spec.pipeline} prediction frame is missing required columns: {missing}"
        )
    result["date"] = pd.to_datetime(result["date"])
    return result


def canonicalize_data_columns(
    frame: pd.DataFrame,
    spec: EvaluationSpec,
) -> pd.DataFrame:
    """Normalize raw/processed source data for evaluation joins and split logic."""
    mapping = _column_mapping(frame, spec)
    result = frame.rename(columns=mapping).copy()
    if "date" not in result or "actual" not in result:
        raise ValueError(
            f"{spec.pipeline} data must contain date and target source columns."
        )
    result["date"] = pd.to_datetime(result["date"])
    result["actual"] = result["actual"].astype(float)
    return result


def canonicalize_subgroup_columns(
    frame: pd.DataFrame,
    spec: EvaluationSpec,
) -> pd.DataFrame:
    """Add requested subgroup columns using the source's real names."""
    result = frame.copy()
    aliases = {
        "day_of_week": ("day_of_week", "DayOfWeek"),
        "month": ("month",),
    }
    for subgroup in spec.subgroup_columns:
        if subgroup in result.columns:
            continue
        candidates = aliases.get(subgroup, (subgroup,))
        present = next((name for name in candidates if name in result.columns), None)
        if present is not None:
            result[subgroup] = result[present]
        elif subgroup == "day_of_week":
            result[subgroup] = result["date"].dt.dayofweek
        elif subgroup == "month":
            result[subgroup] = result["date"].dt.month
    return result
