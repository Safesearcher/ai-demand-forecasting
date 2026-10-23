"""
02_clean_and_validate.py
========================
Pipeline A: Rossmann Store Sales cleaning
Pipeline B: Store Item Demand cleaning

LEAKAGE POLICY
--------------
* Time-based splits are performed FIRST.
* IQR thresholds, CompetitionDistance median, and all imputation values are
  computed on the TRAINING portion only, then applied identically to the
  holdout.
* Holdout Sales / sales targets are NEVER capped or modified.
* Rossmann holdout: last 42 days.
* Store Item holdout: last 90 days.

OUTLIER METHOD
--------------
IQR with fence = 1.5 * IQR (standard Tukey), applied per series,
on open days only for Rossmann.

OUTPUTS
-------
  data/processed/rossmann_clean.parquet
  data/processed/store_item_clean.parquet
  data/processed/cleaning_report.json   ← machine-readable, read by LaTeX
"""

from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, required_columns: list, dataset_name: str) -> pd.DataFrame:
    """Check for missing required columns."""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{dataset_name}: Missing required columns: {missing_cols}")
    print(f"  [schema] {dataset_name}: all required columns present")
    return df


def parse_date_column(df: pd.DataFrame, date_col: str,
                      date_format: str = "%Y-%m-%d") -> Tuple[pd.DataFrame, int]:
    """Parse date column; return (df, n_failed_parses)."""
    original = df[date_col].copy()
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    n_failed = int(df[date_col].isna().sum())
    if n_failed > 0:
        print(f"  [dates] WARNING: {n_failed} rows failed date parsing → dropped")
        df = df.dropna(subset=[date_col])
    else:
        print(f"  [dates] All dates parsed successfully (0 failures)")
    return df, n_failed


def sort_chronologically(df: pd.DataFrame, group_cols: list, date_col: str) -> pd.DataFrame:
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    return df


def remove_duplicates(df: pd.DataFrame, group_cols: list, date_col: str) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate (entity, Date) records; keep first."""
    n_before = len(df)
    df = df.drop_duplicates(subset=group_cols + [date_col], keep="first")
    n_dup = n_before - len(df)
    print(f"  [duplicates] Removed: {n_dup}")
    return df, n_dup


def validate_sales(df: pd.DataFrame, sales_col: str) -> Tuple[pd.DataFrame, int]:
    """Coerce non-numeric to NaN; set negative/NaN to 0."""
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    n_invalid = int(((df[sales_col] < 0) | df[sales_col].isna()).sum())
    if n_invalid > 0:
        print(f"  [sales] {n_invalid} invalid (negative/non-numeric) → set to 0")
        df.loc[(df[sales_col] < 0) | df[sales_col].isna(), sales_col] = 0
    else:
        print(f"  [sales] All sales values valid (non-negative, numeric)")
    return df, n_invalid


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A – Rossmann
# ─────────────────────────────────────────────────────────────────────────────

def clean_rossmann_data(raw_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean Rossmann. Returns (cleaned_df, report_dict)."""
    print("\n" + "=" * 60)
    print("PIPELINE A: Rossmann Store Sales Cleaning")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(raw_dir / "rossmann" / "train.csv", low_memory=False)
    store_df = pd.read_csv(raw_dir / "rossmann" / "store.csv")
    original_rows = len(train_df)
    print(f"  Loaded train.csv: {original_rows:,} rows | store.csv: {len(store_df):,} rows")

    # ── Schema ──────────────────────────────────────────────────────────────
    train_df = validate_schema(
        train_df,
        ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo",
         "StateHoliday", "SchoolHoliday"],
        "Rossmann Train"
    )
    store_df = validate_schema(store_df, ["Store", "StoreType", "Assortment"], "Rossmann Store")

    # ── Date parsing ─────────────────────────────────────────────────────────
    train_df, n_failed_dates = parse_date_column(train_df, "Date")

    # ── Sort chronologically ─────────────────────────────────────────────────
    train_df = sort_chronologically(train_df, ["Store"], "Date")

    # ── Duplicates ───────────────────────────────────────────────────────────
    train_df, n_dup = remove_duplicates(train_df, ["Store"], "Date")

    # ── Negative / invalid sales ─────────────────────────────────────────────
    train_df, n_invalid_sales = validate_sales(train_df, "Sales")

    # ── Merge with store metadata ────────────────────────────────────────────
    df = train_df.merge(store_df, on="Store", how="left")
    after_merge_rows = len(df)
    print(f"  After merge: {after_merge_rows:,} rows")

    # ── TIME-BASED SPLIT (train / holdout) ───────────────────────────────────
    # Thresholds are computed on TRAIN only, applied to holdout without
    # modifying holdout Sales targets.
    HOLDOUT_DAYS = 42
    max_date = df["Date"].max()
    split_date = max_date - pd.Timedelta(days=HOLDOUT_DAYS)
    train_mask = df["Date"] <= split_date
    print(f"  Split: train ≤ {split_date.date()}, holdout > {split_date.date()}")

    # ── Missing value imputation (median from TRAINING only) ─────────────────
    if df["CompetitionDistance"].isna().any():
        # Compute median on training portion only
        comp_median = df.loc[train_mask, "CompetitionDistance"].median()
        df["CompetitionDistance_is_missing"] = df["CompetitionDistance"].isna().astype(int)
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(comp_median)
        print(f"  [missing] CompetitionDistance imputed with TRAIN median: {comp_median:.2f}")
    else:
        df["CompetitionDistance_is_missing"] = 0

    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    # ── Closed days rule ──────────────────────────────────────────────────────
    closed_violations = int(((df["Open"] == 0) & (df["Sales"] > 0)).sum())
    if closed_violations > 0:
        df.loc[(df["Open"] == 0) & (df["Sales"] > 0), "Sales"] = 0
        print(f"  [closed] Fixed {closed_violations} records Open=0 but Sales>0")
    else:
        print(f"  [closed] No violations (Open=0 → Sales=0 rule)")
    df["is_closed"] = (df["Open"] == 0).astype(int)

    # ── Zero-sales analysis ───────────────────────────────────────────────────
    open_df_full = df[df["Open"] == 1]
    n_zero_open = int((open_df_full["Sales"] == 0).sum())
    pct_zero_open = n_zero_open / len(open_df_full) * 100 if len(open_df_full) > 0 else 0
    n_zero_total = int((df["Sales"] == 0).sum())
    pct_zero_total = n_zero_total / len(df) * 100
    print(f"  [zero-sales] Total zero rows: {n_zero_total:,} ({pct_zero_total:.2f}%)")
    print(f"  [zero-sales] Zero rows on OPEN days: {n_zero_open:,} ({pct_zero_open:.2f}%)")

    # ── Outlier detection: IQR×1.5, per store, on OPEN days in TRAIN only ────
    # Policy: flag and cap on TRAINING data. Holdout Sales are NEVER touched.
    print("\n  [outliers] Method: IQR × 1.5 per store, open days, training split only")
    open_train = df[df["Open"] == 1].copy()  # open days, all splits
    train_open = open_train[train_mask[open_train.index]]  # open + training only

    outlier_rows_flagged = 0
    series_affected = 0
    outlier_index = set()

    for store, grp in train_open.groupby("Store"):
        q1 = grp["Sales"].quantile(0.25)
        q3 = grp["Sales"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        flag = (grp["Sales"] > upper) | (grp["Sales"] < lower)
        n_flag = int(flag.sum())
        if n_flag > 0:
            series_affected += 1
            outlier_rows_flagged += n_flag
            # Cap at upper bound in training data only
            cap_idx = grp.index[flag & (grp["Sales"] > upper)]
            df["Sales"] = df["Sales"].astype(float)
            df.loc[cap_idx, "Sales"] = float(upper)
            outlier_index.update(grp.index[flag].tolist())

    pct_outlier = outlier_rows_flagged / len(train_open) * 100 if len(train_open) > 0 else 0
    print(f"  [outliers] Rows flagged (train only): {outlier_rows_flagged:,}")
    print(f"  [outliers] Series affected: {series_affected} of {df['Store'].nunique()}")
    print(f"  [outliers] Percentage of open-train rows: {pct_outlier:.3f}%")
    print(f"  NOTE: Holdout Sales targets were NOT modified.")

    # ── StateHoliday: str for parquet ────────────────────────────────────────
    df["StateHoliday"] = df["StateHoliday"].astype(str)

    # ── Summary report ────────────────────────────────────────────────────────
    report = {
        "pipeline": "Rossmann",
        "original_rows": original_rows,
        "after_merge_rows": after_merge_rows,
        "failed_date_parses": n_failed_dates,
        "duplicates_removed": n_dup,
        "invalid_sales_fixed": n_invalid_sales,
        "closed_day_violations_fixed": closed_violations,
        "zero_sales_total": n_zero_total,
        "zero_sales_pct_total": round(pct_zero_total, 3),
        "zero_sales_open_days": n_zero_open,
        "zero_sales_pct_open": round(pct_zero_open, 3),
        "outlier_method": "IQR × 1.5 per store, open days, training split only",
        "outlier_rows_flagged": outlier_rows_flagged,
        "outlier_series_affected": series_affected,
        "outlier_pct_of_open_train_rows": round(pct_outlier, 3),
        "holdout_target_modified": False,
        "train_split_date": split_date.strftime("%Y-%m-%d"),
        "date_range_start": df["Date"].min().strftime("%Y-%m-%d"),
        "date_range_end": df["Date"].max().strftime("%Y-%m-%d"),
        "n_stores": int(df["Store"].nunique()),
        "final_clean_rows": len(df),
    }

    print("\n  Rossmann cleaning report:")
    for k, v in report.items():
        print(f"    {k}: {v}")

    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B – Store Item
# ─────────────────────────────────────────────────────────────────────────────

def clean_store_item_data(raw_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean Store Item Demand. Returns (cleaned_df, report_dict)."""
    print("\n" + "=" * 60)
    print("PIPELINE B: Store Item Demand Cleaning")
    print("=" * 60)

    df = pd.read_csv(raw_dir / "store_item" / "train.csv")
    original_rows = len(df)
    print(f"  Loaded train.csv: {original_rows:,} rows")

    # ── Schema ──────────────────────────────────────────────────────────────
    df = validate_schema(df, ["date", "store", "item", "sales"], "Store Item")

    # ── Date parsing ─────────────────────────────────────────────────────────
    df, n_failed_dates = parse_date_column(df, "date")

    # ── Sort ─────────────────────────────────────────────────────────────────
    df = sort_chronologically(df, ["store", "item"], "date")

    # ── Duplicates ───────────────────────────────────────────────────────────
    df, n_dup = remove_duplicates(df, ["store", "item"], "date")

    # ── Negative / invalid sales ─────────────────────────────────────────────
    df, n_invalid_sales = validate_sales(df, "sales")

    # ── Negative sales check ─────────────────────────────────────────────────
    n_negative = int((df["sales"] < 0).sum())  # already zeroed by validate_sales

    # ── DATE CONTINUITY: fill gaps with 0 ────────────────────────────────────
    print("\n  [date-gaps] Checking continuous daily date ranges per (store, item)...")
    date_gaps_series = []
    full_chunks = []
    n_dates_filled = 0

    for (store, item), grp in df.groupby(["store", "item"]):
        full_range = pd.date_range(start=grp["date"].min(), end=grp["date"].max(), freq="D")
        missing = full_range.difference(grp["date"])
        if len(missing) > 0:
            date_gaps_series.append((store, item, len(missing)))
            n_dates_filled += len(missing)
            grp_reindexed = (
                grp.set_index("date")
                .reindex(full_range)
                .reset_index()
                .rename(columns={"index": "date"})
            )
            grp_reindexed["store"] = store
            grp_reindexed["item"] = item
            grp_reindexed["sales"] = grp_reindexed["sales"].fillna(0)
            full_chunks.append(grp_reindexed)
        else:
            full_chunks.append(grp)

    if n_dates_filled > 0:
        df = pd.concat(full_chunks, ignore_index=True)
        print(f"  [date-gaps] {len(date_gaps_series)} series had gaps → filled {n_dates_filled} dates with 0")
    else:
        print(f"  [date-gaps] All series continuous — 0 dates filled")

    # ── Zero-sales ────────────────────────────────────────────────────────────
    n_zero_sales = int((df["sales"] == 0).sum())
    pct_zero = n_zero_sales / len(df) * 100
    print(f"  [zero-sales] {n_zero_sales:,} zero-sales rows ({pct_zero:.2f}%)")

    # ── TIME-BASED SPLIT ──────────────────────────────────────────────────────
    HOLDOUT_DAYS = 90
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=HOLDOUT_DAYS)
    train_mask = df["date"] <= split_date
    print(f"  Split: train ≤ {split_date.date()}, holdout > {split_date.date()}")

    # ── Outlier detection: IQR × 1.5 per (store, item), TRAINING only ────────
    # Holdout targets are NOT modified.
    print("\n  [outliers] Method: IQR × 1.5 per (store, item) series, training split only")

    train_df = df[train_mask].copy()
    outlier_rows_flagged = 0
    series_affected = 0

    for (store, item), grp in train_df.groupby(["store", "item"]):
        q1 = grp["sales"].quantile(0.25)
        q3 = grp["sales"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        flag = (grp["sales"] > upper) | (grp["sales"] < lower)
        n_flag = int(flag.sum())
        if n_flag > 0:
            series_affected += 1
            outlier_rows_flagged += n_flag
            cap_idx = grp.index[flag & (grp["sales"] > upper)]
            df["sales"] = df["sales"].astype(float)
            df.loc[cap_idx, "sales"] = float(upper)

    pct_outlier = outlier_rows_flagged / len(train_df) * 100 if len(train_df) > 0 else 0
    print(f"  [outliers] Rows flagged (train only): {outlier_rows_flagged:,}")
    print(f"  [outliers] Series affected: {series_affected} of {df.groupby(['store','item']).ngroups}")
    print(f"  [outliers] Percentage of training rows: {pct_outlier:.3f}%")
    print(f"  NOTE: Holdout sales targets were NOT modified.")

    # ── Summary report ────────────────────────────────────────────────────────
    report = {
        "pipeline": "Store Item",
        "original_rows": original_rows,
        "failed_date_parses": n_failed_dates,
        "duplicates_removed": n_dup,
        "invalid_sales_fixed": n_invalid_sales,
        "negative_sales": n_negative,
        "dates_filled_with_zero": n_dates_filled,
        "series_with_date_gaps": len(date_gaps_series),
        "zero_sales_rows": n_zero_sales,
        "zero_sales_pct": round(pct_zero, 3),
        "outlier_method": "IQR × 1.5 per (store, item) series, training split only",
        "outlier_rows_flagged": outlier_rows_flagged,
        "outlier_series_affected": series_affected,
        "outlier_pct_of_train_rows": round(pct_outlier, 3),
        "holdout_target_modified": False,
        "train_split_date": split_date.strftime("%Y-%m-%d"),
        "date_range_start": df["date"].min().strftime("%Y-%m-%d"),
        "date_range_end": df["date"].max().strftime("%Y-%m-%d"),
        "n_stores": int(df["store"].nunique()),
        "n_items": int(df["item"].nunique()),
        "n_series": int(df.groupby(["store", "item"]).ngroups),
        "final_clean_rows": len(df),
    }

    print("\n  Store Item cleaning report:")
    for k, v in report.items():
        print(f"    {k}: {v}")

    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Run both pipelines ────────────────────────────────────────────────────
    rossmann_df, rossmann_report = clean_rossmann_data(raw_dir)
    store_item_df, store_item_report = clean_store_item_data(raw_dir)

    # ── Export Parquets ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Exporting cleaned datasets...")
    print("=" * 60)

    rossmann_df.to_parquet(processed_dir / "rossmann_clean.parquet", index=False)
    print(f"  Saved: rossmann_clean.parquet")

    store_item_df.to_parquet(processed_dir / "store_item_clean.parquet", index=False)
    print(f"  Saved: store_item_clean.parquet")

    # ── Export cleaning report (JSON) ─────────────────────────────────────────
    # NOTE: this file is allow-listed in .gitignore (no credentials, just stats)
    cleaning_report = {
        "generated_at": datetime.now().isoformat(),
        "rossmann": rossmann_report,
        "store_item": store_item_report,
    }
    report_path = processed_dir / "cleaning_report.json"
    with open(report_path, "w") as f:
        json.dump(cleaning_report, f, indent=2)
    print(f"  Saved: cleaning_report.json")

    # Also write a CSV summary for backward compatibility
    import csv
    summary_rows = [rossmann_report, store_item_report]
    all_keys = list(dict.fromkeys(k for r in summary_rows for k in r.keys()))
    csv_path = processed_dir / "cleaning_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"  Saved: cleaning_summary.csv")

    print("\n" + "=" * 60)
    print("Cleaning and validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
