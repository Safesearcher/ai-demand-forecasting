"""
03_feature_engineering.py
==========================
Leakage-safe feature engineering for both pipelines.

LAG WARM-UP POLICY
-------------------
Lag features create NaN rows at the START of each series. These rows are
dropped from the training set after feature computation (first 28 rows per
series, corresponding to the maximum lag used). The number of rows dropped
is logged and reported.

NOTE ON LAGS AND FORECAST HORIZON
-----------------------------------
Sales_lag_1 and other short lags are valid ONLY for 1-day-ahead forecasts.
In the modeling phase, lags will be recomputed relative to the forecast
origin: for a h-step-ahead forecast, only lag >= h features are used
(direct multi-horizon forecasting). This script produces lag features for
EDA and single-step baseline purposes only.

SCALING NOTE
------------
StandardScaler is fit on training data only and applied to validation.
LightGBM does NOT require feature scaling — scaling is provided exclusively
for linear / baseline models (Ridge, Lasso, linear SVR).
The target column (Sales) is NOT log-transformed in this script.
If log1p(Sales) is needed for a specific model, apply it inside that model's
training loop, NOT here, to keep the stored features in original units.

OUTPUTS
-------
  data/processed/rossmann_train_features.parquet
  data/processed/rossmann_val_features.parquet
  data/processed/store_item_train_features.parquet
  data/processed/store_item_val_features.parquet
  data/processed/feature_config.json
  models/scalers/{pipeline}/standard_scaler.pkl
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Dict, Any, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# Temporal / cyclical features
# ─────────────────────────────────────────────────────────────────────────────

def create_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create temporal and cyclical features from the date column."""
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.dayofweek          # 0=Monday, 6=Sunday
    df["day_of_month"] = df[date_col].dt.day
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Cyclical encoding (preserves periodicity for distance-based models)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    print("  [temporal] day_of_week, day_of_month, week_of_year, month, year, is_weekend")
    print("  [temporal] cyclical: day_of_week_sin/cos, month_sin/cos")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Lag / rolling features
# ─────────────────────────────────────────────────────────────────────────────

def create_lag_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    lags: List[int] = [1, 7, 14, 28],
    windows: List[int] = [7, 14, 28],
) -> pd.DataFrame:
    """
    Create lag and rolling features per group series.

    BUG FIX: sort by group_cols + Date before computing (previous version
    sorted only by "Date", breaking the per-store order).

    NOTE: Sales_lag_1 is valid for 1-step-ahead only. The modeling phase will
    select lag >= h for h-step-ahead direct forecasting.
    """
    df = df.copy()
    # Sort is critical — must be by group AND date
    df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)

    # Lag features
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df.groupby(group_cols)[value_col].shift(lag)

    print(f"  [lags] {[f'{value_col}_lag_{l}' for l in lags]}")

    # Rolling features (shifted by 1 to prevent any same-day leakage)
    for w in windows:
        df[f"{value_col}_roll_mean_{w}"] = (
            df.groupby(group_cols)[value_col]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"{value_col}_roll_std_{w}"] = (
            df.groupby(group_cols)[value_col]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).std().fillna(0))
        )
    print(f"  [rolling] windows={windows}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Time-based split
# ─────────────────────────────────────────────────────────────────────────────

def time_based_split(
    df: pd.DataFrame,
    date_col: str,
    holdout_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict time-based train / holdout split."""
    max_date = df[date_col].max()
    split_date = max_date - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= split_date].copy()
    val = df[df[date_col] > split_date].copy()
    print(f"  [split] max_date={max_date.date()}, split={split_date.date()}, "
          f"train={len(train):,}, val={len(val):,}")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Drop lag warm-up rows
# ─────────────────────────────────────────────────────────────────────────────

def drop_lag_warmup_rows(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    max_lag: int = 28,
) -> Tuple[pd.DataFrame, int]:
    """
    Drop the first max_lag rows per series (NaN-heavy lag warm-up period).
    Only applied to the TRAINING set to preserve holdout integrity.

    Returns (df_without_warmup, n_rows_dropped).
    """
    df = df.copy().sort_values(group_cols + [date_col])
    df["_row_rank"] = df.groupby(group_cols).cumcount()
    warmup_mask = df["_row_rank"] < max_lag
    n_dropped = int(warmup_mask.sum())
    df = df[~warmup_mask].drop(columns=["_row_rank"])
    print(f"  [warmup] Dropped first {max_lag} rows per series = {n_dropped:,} rows total")
    return df.reset_index(drop=True), n_dropped


# ─────────────────────────────────────────────────────────────────────────────
# Categorical encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_categorical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cat_cols: List[str],
    method: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Encode categorical features. Encoders fit on training only."""
    train_df, val_df = train_df.copy(), val_df.copy()
    encoding_info = {}

    for col in cat_cols:
        if col not in train_df.columns:
            continue
        if method == "label":
            le = LabelEncoder()
            train_df[col] = train_df[col].astype(str)
            le.fit(list(train_df[col]) + ["__unknown__"])
            train_df[col] = le.transform(train_df[col])
            val_df[col] = val_df[col].astype(str)
            unseen = ~val_df[col].isin(le.classes_)
            val_df.loc[unseen, col] = "__unknown__"
            val_df[col] = le.transform(val_df[col])
            encoding_info[col] = {"method": "label", "n_classes": len(le.classes_)}
        elif method == "onehot":
            td = pd.get_dummies(train_df[col], prefix=col, dtype=int)
            vd = pd.get_dummies(val_df[col], prefix=col, dtype=int)
            all_c = sorted(set(td.columns) | set(vd.columns))
            for c in all_c:
                if c not in td.columns: td[c] = 0
                if c not in vd.columns: vd[c] = 0
            train_df = pd.concat([train_df.drop(col, axis=1), td[all_c]], axis=1)
            val_df = pd.concat([val_df.drop(col, axis=1), vd[all_c]], axis=1)
            encoding_info[col] = {"method": "onehot", "columns": all_c}

    print(f"  [encoding] {method} encoding for: {cat_cols}")
    return train_df, val_df, encoding_info


# ─────────────────────────────────────────────────────────────────────────────
# Scaling  (for linear / baseline models only)
# ─────────────────────────────────────────────────────────────────────────────

def scale_numerical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_cols: List[str],
    scaler_dir: Path,
    pipeline_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fit StandardScaler on training data only; transform both splits.

    NOTE: LightGBM does NOT require scaling. These scaled columns are
    intended for linear / baseline models only. The unscaled parquets are
    the canonical outputs; scaling is applied in-memory here for reference.
    """
    train_df, val_df = train_df.copy(), val_df.copy()
    existing_cols = [c for c in num_cols if c in train_df.columns and c in val_df.columns]

    scaler = StandardScaler()
    # Fit on TRAINING only
    train_df[existing_cols] = scaler.fit_transform(train_df[existing_cols].fillna(0))
    val_df[existing_cols] = scaler.transform(val_df[existing_cols].fillna(0))

    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / "standard_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    # Also save a copy with pipeline prefix at models/scalers/ root
    root_path = scaler_dir.parent / f"{pipeline_name}_standard_scaler.pkl"
    joblib.dump(scaler, root_path)
    print(f"  [scaling] StandardScaler fit on training; saved to {scaler_path}")
    print(f"  [scaling] NOTE: LightGBM does NOT require scaling (for linear baselines only)")

    return train_df, val_df, {
        "scaler_type": "standard",
        "note": "Fit on training split only. LightGBM does not need scaling.",
        "scaled_columns": existing_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A: Rossmann
# ─────────────────────────────────────────────────────────────────────────────

def feature_engineering_rossmann(
    df: pd.DataFrame, scaler_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    print("\n" + "=" * 60)
    print("PIPELINE A: Rossmann Feature Engineering")
    print("=" * 60)

    # Temporal features
    df = create_temporal_features(df, "Date")

    # Lag / rolling for Sales
    df = create_lag_rolling_features(df, ["Store"], "Sales",
                                      lags=[1, 7, 14, 28], windows=[7, 14, 28])

    # Customers: same-day value is UNKNOWN in advance → drop it;
    # keep only lagged/rolling customer features.
    # NOTE: same-day Customers has very high correlation with Sales in the raw
    # data (r ≈ 0.82), which is why it must be excluded — it would be a
    # target-leaking feature in production.
    print("\n  [customers] Dropping same-day Customers (unknown in advance, r≈0.82 with Sales)")
    cust_df = create_lag_rolling_features(
        df[["Store", "Date", "Customers"]].copy(),
        ["Store"], "Customers", lags=[7, 14], windows=[7, 14]
    )
    cust_feat_cols = [c for c in cust_df.columns
                      if c.startswith("Customers_") and c != "Customers"]
    df = df.drop(columns=["Customers"], errors="ignore")
    df = df.merge(cust_df[["Store", "Date"] + cust_feat_cols], on=["Store", "Date"], how="left")

    # Time-based split
    HOLDOUT_DAYS = 42
    train_df, val_df = time_based_split(df, "Date", HOLDOUT_DAYS)

    # Drop lag warm-up rows from TRAINING only (not holdout)
    MAX_LAG = 28
    train_df, n_warmup_dropped = drop_lag_warmup_rows(train_df, ["Store"], "Date", MAX_LAG)

    # Categorical encoding (fit on training)
    cat_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
    train_df, val_df, encoding_info = encode_categorical_features(
        train_df, val_df, cat_cols, method="label"
    )

    # Scaling (for linear baselines only)
    num_cols = [
        "CompetitionDistance",
        "day_of_week", "day_of_month", "week_of_year", "month", "year",
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_28",
        "Sales_roll_mean_7", "Sales_roll_std_7",
        "Sales_roll_mean_14", "Sales_roll_std_14",
        "Sales_roll_mean_28", "Sales_roll_std_28",
        "Customers_lag_7", "Customers_lag_14",
        "Customers_roll_mean_7", "Customers_roll_std_7",
        "Customers_roll_mean_14", "Customers_roll_std_14",
    ]
    train_df, val_df, scaling_info = scale_numerical_features(
        train_df, val_df, num_cols, scaler_dir / "rossmann", "rossmann"
    )

    feature_config = {
        "pipeline": "Rossmann",
        "holdout_days": HOLDOUT_DAYS,
        "lag_warmup_rows_dropped_from_train": n_warmup_dropped,
        "max_lag_used": MAX_LAG,
        "target_column": "Sales",
        "target_log_transform": False,
        "target_log_transform_note": "Sales stored in original units. Apply log1p inside model training if needed.",
        "lag_horizon_note": (
            "Sales_lag_1 is valid for 1-step-ahead only. "
            "For direct multi-horizon forecasting, the modeling phase recomputes "
            "lags relative to the forecast origin: lag >= h for h-step-ahead."
        ),
        "customers_note": "Same-day Customers dropped (r≈0.82 with Sales — unknown in advance).",
        "temporal_features": [
            "day_of_week", "day_of_month", "week_of_year", "month", "year", "is_weekend",
            "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        ],
        "lag_features": [f"Sales_lag_{l}" for l in [1, 7, 14, 28]] +
                        [f"Customers_lag_{l}" for l in [7, 14]],
        "rolling_features": (
            [f"Sales_roll_mean_{w}" for w in [7, 14, 28]] +
            [f"Sales_roll_std_{w}" for w in [7, 14, 28]] +
            [f"Customers_roll_mean_{w}" for w in [7, 14]] +
            [f"Customers_roll_std_{w}" for w in [7, 14]]
        ),
        "categorical_encoding": encoding_info,
        "numerical_scaling": scaling_info,
    }

    print("\n  Rossmann feature engineering complete")
    print(f"    train rows: {len(train_df):,} | val rows: {len(val_df):,}")
    return train_df, val_df, feature_config


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B: Store Item
# ─────────────────────────────────────────────────────────────────────────────

def feature_engineering_store_item(
    df: pd.DataFrame, scaler_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    print("\n" + "=" * 60)
    print("PIPELINE B: Store Item Feature Engineering")
    print("=" * 60)

    # Standardise column names
    df = df.rename(columns={"date": "Date", "sales": "Sales"})

    # Temporal features
    df = create_temporal_features(df, "Date")

    # Lag / rolling for Sales
    df = create_lag_rolling_features(df, ["store", "item"], "Sales",
                                      lags=[1, 7, 14, 28], windows=[7, 14, 28])

    # Time-based split
    HOLDOUT_DAYS = 90
    train_df, val_df = time_based_split(df, "Date", HOLDOUT_DAYS)

    # Drop lag warm-up rows from TRAINING only
    MAX_LAG = 28
    train_df, n_warmup_dropped = drop_lag_warmup_rows(train_df, ["store", "item"], "Date", MAX_LAG)

    # Scaling (for linear baselines only)
    num_cols = [
        "day_of_week", "day_of_month", "week_of_year", "month", "year",
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_28",
        "Sales_roll_mean_7", "Sales_roll_std_7",
        "Sales_roll_mean_14", "Sales_roll_std_14",
        "Sales_roll_mean_28", "Sales_roll_std_28",
    ]
    train_df, val_df, scaling_info = scale_numerical_features(
        train_df, val_df, num_cols, scaler_dir / "store_item", "store_item"
    )

    feature_config = {
        "pipeline": "Store Item",
        "holdout_days": HOLDOUT_DAYS,
        "lag_warmup_rows_dropped_from_train": n_warmup_dropped,
        "max_lag_used": MAX_LAG,
        "target_column": "Sales",
        "target_log_transform": False,
        "target_log_transform_note": "Sales stored in original units. Apply log1p inside model training if needed.",
        "lag_horizon_note": (
            "Sales_lag_1 is valid for 1-step-ahead only. "
            "For direct multi-horizon forecasting, the modeling phase recomputes "
            "lags relative to the forecast origin: lag >= h for h-step-ahead."
        ),
        "temporal_features": [
            "day_of_week", "day_of_month", "week_of_year", "month", "year", "is_weekend",
            "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        ],
        "lag_features": [f"Sales_lag_{l}" for l in [1, 7, 14, 28]],
        "rolling_features": (
            [f"Sales_roll_mean_{w}" for w in [7, 14, 28]] +
            [f"Sales_roll_std_{w}" for w in [7, 14, 28]]
        ),
        "categorical_encoding": {},
        "numerical_scaling": scaling_info,
    }

    print("\n  Store Item feature engineering complete")
    print(f"    train rows: {len(train_df):,} | val rows: {len(val_df):,}")
    return train_df, val_df, feature_config


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    scaler_dir = project_root / "models" / "scalers"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned datasets...")
    rossmann_df = pd.read_parquet(processed_dir / "rossmann_clean.parquet")
    store_item_df = pd.read_parquet(processed_dir / "store_item_clean.parquet")

    rossmann_train, rossmann_val, rossmann_cfg = feature_engineering_rossmann(rossmann_df, scaler_dir)
    si_train, si_val, si_cfg = feature_engineering_store_item(store_item_df, scaler_dir)

    print("\n" + "=" * 60)
    print("Exporting feature-engineered datasets...")
    print("=" * 60)

    rossmann_train.to_parquet(processed_dir / "rossmann_train_features.parquet", index=False)
    rossmann_val.to_parquet(processed_dir / "rossmann_val_features.parquet", index=False)
    # Aliases for backward compatibility
    rossmann_train.to_parquet(processed_dir / "train_rossmann.parquet", index=False)
    rossmann_val.to_parquet(processed_dir / "val_rossmann.parquet", index=False)
    print("  Saved Rossmann train/val feature parquets")

    si_train.to_parquet(processed_dir / "store_item_train_features.parquet", index=False)
    si_val.to_parquet(processed_dir / "store_item_val_features.parquet", index=False)
    si_train.to_parquet(processed_dir / "train_store_item.parquet", index=False)
    si_val.to_parquet(processed_dir / "val_store_item.parquet", index=False)
    print("  Saved Store Item train/val feature parquets")

    # Feature config JSON
    feature_config = {"rossmann": rossmann_cfg, "store_item": si_cfg}
    cfg_path = processed_dir / "feature_config.json"
    with open(cfg_path, "w") as f:
        json.dump(feature_config, f, indent=2)
    print(f"  Saved: feature_config.json")

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
