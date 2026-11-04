"""Chronological validation and test split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import EvaluationSpec, canonicalize_data_columns


@dataclass(frozen=True)
class SplitFrames:
    """Chronological development, validation, and final test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def build_chronological_splits(
    raw_data: pd.DataFrame,
    spec: EvaluationSpec,
) -> SplitFrames:
    """Split data into train, validation, and untouched D1 test periods."""
    data = canonicalize_data_columns(raw_data, spec).sort_values("date").reset_index(
        drop=True
    )
    test_start = data["date"].max() - pd.Timedelta(days=spec.holdout_days - 1)
    development = data[data["date"] < test_start].copy()
    test = data[data["date"] >= test_start].copy()
    validation_start = development["date"].max() - pd.Timedelta(
        days=spec.holdout_days - 1
    )
    train = development[development["date"] < validation_start].copy()
    validation = development[development["date"] >= validation_start].copy()
    return SplitFrames(train=train, validation=validation, test=test)


def summarize_splits(
    splits: SplitFrames,
    spec: EvaluationSpec,
) -> pd.DataFrame:
    """Return row counts and date ranges for each chronological partition."""
    rows = []
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        rows.append(
            {
                "pipeline": spec.pipeline,
                "split": name,
                "rows": len(frame),
                "date_min": frame["date"].min(),
                "date_max": frame["date"].max(),
            }
        )
    return pd.DataFrame(rows)


def save_split_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Append or create a split summary CSV for the evaluated pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def expanding_window_splits(
    train_data: pd.DataFrame,
    spec: EvaluationSpec,
    n_folds: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window folds without using future observations."""
    if n_folds < 1:
        raise ValueError("n_folds must be positive.")
    data = canonicalize_data_columns(train_data, spec).sort_values("date")
    unique_dates = pd.Index(data["date"].drop_duplicates().sort_values())
    if len(unique_dates) < n_folds + 1:
        raise ValueError("Not enough dates for the requested expanding-window folds.")
    block_size = len(unique_dates) // (n_folds + 1)
    folds = []
    for fold in range(n_folds):
        train_end = block_size * (fold + 1)
        valid_end = block_size * (fold + 2)
        train_dates = unique_dates[:train_end]
        valid_dates = unique_dates[train_end:valid_end]
        folds.append(
            (
                data[data["date"].isin(train_dates)].copy(),
                data[data["date"].isin(valid_dates)].copy(),
            )
        )
    return folds


def load_and_split(
    spec: EvaluationSpec,
    output_dir: Path,
) -> SplitFrames:
    """Load a configured source file, split it, and save its summary."""
    splits = build_chronological_splits(pd.read_parquet(spec.processed_path), spec)
    summary = summarize_splits(splits, spec)
    save_split_summary(summary, output_dir / "split_summary.csv")
    return splits
