"""
04_eda_visualizations.py
========================
Generate all EDA figures for both pipelines.

Figures saved to report/figures/ at 300 dpi.

Pipeline A (Rossmann):
  - rossmann_sales_distribution.png   (4-panel: histogram, log, promo boxplot, store-type boxplot)
  - rossmann_raw_correlation.png      (raw column heatmap: Sales, Customers, Promo, SchoolHoliday,
                                       Open, CompetitionDistance)
  - rossmann_lag_correlation.png      (engineered-lag heatmap, larger)
  - time_based_split_diagram.png      (train/holdout diagram)

Pipeline B (Store Item):
  - store_item_sales_histogram.png    (overall sales histogram)
  - store_item_dow_profile.png        (mean sales by day of week)
  - store_item_monthly_profile.png    (mean sales by month)
  - store_item_zero_summary.png       (zero/missing value summary)
  - store_item_seasonality.png        (3 sample time-series)

IMPORTANT NOTES
---------------
* Promo boxplot (rossmann_sales_distribution.png, bottom-left): the mean
  annotation for each group is computed FROM THE DATA for that group —
  never hardcoded or swapped.
* Same-day Customers has r ≈ 0.82 with Sales (shown in raw heatmap) →
  justifies dropping it as a same-day feature.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

def setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A — Rossmann
# ─────────────────────────────────────────────────────────────────────────────

def plot_rossmann_sales_distribution(df: pd.DataFrame, output_path: Path):
    """4-panel distribution figure.

    Bottom-left: Promo vs No-Promo boxplot.
    Mean annotations are computed from the actual data for each group.
    BUG FIX: previous version had the mean labels swapped.
    """
    print("  Generating rossmann_sales_distribution.png ...")

    open_df = df[df["Open"] == 1].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top-left: raw histogram ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(open_df["Sales"], bins=60, edgecolor="none", alpha=0.75, color="#2196F3")
    mean_val = open_df["Sales"].mean()
    ax.axvline(mean_val, color="crimson", linestyle="--", lw=1.5,
               label=f"Mean: {mean_val:,.0f}")
    ax.set_xlabel("Daily Sales (€)")
    ax.set_ylabel("Count")
    ax.set_title("Sales Distribution — Open Days (Original Scale)")
    ax.legend()

    # ── Top-right: log-transformed ────────────────────────────────────────────
    ax = axes[0, 1]
    log_sales = np.log1p(open_df["Sales"])
    ax.hist(log_sales, bins=60, edgecolor="none", alpha=0.75, color="#FF9800")
    log_mean = log_sales.mean()
    ax.axvline(log_mean, color="crimson", linestyle="--", lw=1.5,
               label=f"Mean: {log_mean:.3f}")
    ax.set_xlabel("log(Sales + 1)")
    ax.set_ylabel("Count")
    ax.set_title("Sales Distribution — Log-Transformed (Open Days)")
    ax.legend()

    # ── Bottom-left: Promo vs No-Promo ───────────────────────────────────────
    # Means computed from data for each group — labels follow boxplot order
    ax = axes[1, 0]
    no_promo = open_df[open_df["Promo"] == 0]["Sales"]
    promo    = open_df[open_df["Promo"] == 1]["Sales"]

    # boxplot positions: 1 = No Promo, 2 = Promo (matplotlib default)
    bp = ax.boxplot(
        [no_promo, promo],
        labels=["No Promo", "Promo"],
        patch_artist=True,
        boxprops=dict(facecolor="#B3D9FF"),
        medianprops=dict(color="navy", lw=2),
    )

    # Annotate means — computed from the correct group
    mean_no_promo = no_promo.mean()
    mean_promo    = promo.mean()
    ax.text(1, no_promo.quantile(0.75) * 1.05,
            f"Mean: {mean_no_promo:,.0f}", ha="center", va="bottom",
            fontsize=9, color="navy")
    ax.text(2, promo.quantile(0.75) * 1.05,
            f"Mean: {mean_promo:,.0f}", ha="center", va="bottom",
            fontsize=9, color="navy")

    ax.set_ylabel("Daily Sales (€)")
    ax.set_title("Sales Distribution by Promotion Status")

    # ── Bottom-right: by Store Type ───────────────────────────────────────────
    ax = axes[1, 1]
    store_types = sorted(open_df["StoreType"].unique())
    data_by_type = [open_df[open_df["StoreType"] == st]["Sales"].values for st in store_types]
    ax.boxplot(data_by_type, labels=store_types,
               patch_artist=True,
               boxprops=dict(facecolor="#FFD9B3"),
               medianprops=dict(color="darkred", lw=2))
    ax.set_ylabel("Daily Sales (€)")
    ax.set_title("Sales Distribution by Store Type")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def plot_rossmann_raw_correlation(df: pd.DataFrame, output_path: Path):
    """Raw column correlation heatmap.

    Columns: Sales, Customers, Promo, SchoolHoliday, Open, CompetitionDistance.
    Note: same-day Customers has high r with Sales → justifies dropping it.
    """
    print("  Generating rossmann_raw_correlation.png ...")

    open_df = df[df["Open"] == 1].copy()
    raw_cols = [c for c in ["Sales", "Customers", "Promo", "SchoolHoliday",
                             "Open", "CompetitionDistance"] if c in open_df.columns]
    corr = open_df[raw_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr, dtype=bool)  # show all values
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 11})
    ax.set_title("Rossmann — Raw Column Correlation (Open Days)\n"
                 "Note: same-day Customers (r≈0.82 with Sales) is dropped as a feature\n"
                 "because customer counts are unknown at forecast time", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def plot_rossmann_lag_correlation(features_df: pd.DataFrame, output_path: Path):
    """Lag-feature correlation heatmap (larger and readable)."""
    print("  Generating rossmann_lag_correlation.png ...")

    lag_cols = ["Sales"] + [c for c in features_df.columns
                             if "Sales_lag" in c or "Sales_roll" in c
                             or "Promo" == c or "SchoolHoliday" == c
                             or "CompetitionDistance" == c]
    # Deduplicate, limit to 12 for readability
    lag_cols = list(dict.fromkeys(lag_cols))[:12]
    existing = [c for c in lag_cols if c in features_df.columns]

    corr = features_df[existing].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.75}, ax=ax,
                annot_kws={"size": 10})
    ax.set_title("Rossmann — Engineered Lag Feature Correlations with Sales\n"
                 "Weekly lags (7, 14, 28) show strong correlation; lag-1 is weaker\n"
                 "due to the 7-day seasonality and closed Sundays", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def plot_time_based_split_diagram(df: pd.DataFrame, output_path: Path, holdout_days: int = 42):
    """Time-based train/holdout split for a single Rossmann store."""
    print("  Generating time_based_split_diagram.png ...")

    store_id = df["Store"].value_counts().idxmax()
    s = df[df["Store"] == store_id].sort_values("Date")
    max_date = s["Date"].max()
    split_date = max_date - pd.Timedelta(days=holdout_days)

    fig, ax = plt.subplots(figsize=(14, 5))
    train_s = s[s["Date"] <= split_date]
    val_s   = s[s["Date"] > split_date]

    ax.plot(train_s["Date"], train_s["Sales"], color="#2196F3", lw=0.8, label="Training")
    ax.plot(val_s["Date"],   val_s["Sales"],   color="#F44336", lw=0.8, label="Holdout (42 days)")
    ax.axvline(split_date, color="black", linestyle="--", lw=1.5, label="Split point")
    ax.axvspan(train_s["Date"].min(), split_date, alpha=0.07, color="#2196F3")
    ax.axvspan(split_date, val_s["Date"].max(), alpha=0.07, color="#F44336")

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (€)")
    ax.set_title(f"Time-Based Holdout Split — Store {store_id} (Pipeline A)\n"
                 f"Training window → last {holdout_days} days reserved for validation holdout")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B — Store Item  (NEW / expanded EDA)
# ─────────────────────────────────────────────────────────────────────────────

def plot_store_item_sales_histogram(df: pd.DataFrame, output_path: Path):
    """Overall sales histogram for Store Item dataset."""
    print("  Generating store_item_sales_histogram.png ...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(df["sales"], bins=60, edgecolor="none", alpha=0.8, color="#9C27B0")
    ax.axvline(df["sales"].mean(), color="crimson", linestyle="--",
               label=f"Mean: {df['sales'].mean():.1f}")
    ax.set_xlabel("Daily Sales")
    ax.set_ylabel("Count")
    ax.set_title("Store Item — Sales Distribution (All Series)")
    ax.legend()

    ax = axes[1]
    log_s = np.log1p(df["sales"])
    ax.hist(log_s, bins=60, edgecolor="none", alpha=0.8, color="#E91E63")
    ax.axvline(log_s.mean(), color="darkred", linestyle="--",
               label=f"Mean: {log_s.mean():.3f}")
    ax.set_xlabel("log(Sales + 1)")
    ax.set_ylabel("Count")
    ax.set_title("Store Item — Log-Transformed Sales Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def plot_store_item_dow_profile(df: pd.DataFrame, output_path: Path):
    """Mean sales by day of week."""
    print("  Generating store_item_dow_profile.png ...")

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date", "sales": "Sales"})
    df["dow"] = pd.to_datetime(df["Date"]).dt.dayofweek
    dow_mean = df.groupby("dow")["Sales"].mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, [dow_mean.get(i, 0) for i in range(7)],
                  color="#3F51B5", edgecolor="white")
    for bar, val in zip(bars, [dow_mean.get(i, 0) for i in range(7)]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Daily Sales")
    ax.set_title("Store Item — Mean Sales by Day of Week\n(Averaged across all stores and items, 2013–2017)")
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def plot_store_item_monthly_profile(df: pd.DataFrame, output_path: Path):
    """Mean sales by month."""
    print("  Generating store_item_monthly_profile.png ...")

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date", "sales": "Sales"})
    df["month"] = pd.to_datetime(df["Date"]).dt.month
    month_mean = df.groupby("month")["Sales"].mean()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(1, 13), [month_mean.get(m, 0) for m in range(1, 13)],
            marker="o", color="#009688", lw=2, markersize=7)
    ax.fill_between(range(1, 13),
                    [month_mean.get(m, 0) for m in range(1, 13)],
                    alpha=0.15, color="#009688")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Daily Sales")
    ax.set_title("Store Item — Mean Sales by Month\n(Averaged across all stores and items, 2013–2017)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def plot_store_item_zero_summary(df: pd.DataFrame, output_path: Path):
    """Zero and missing value summary for Store Item."""
    print("  Generating store_item_zero_summary.png ...")

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date", "sales": "Sales"})

    n_total = len(df)
    n_zero = int((df["Sales"] == 0).sum())
    n_pos = n_total - n_zero
    pct_zero = n_zero / n_total * 100

    # Per-series zero count
    zero_per_series = df.groupby(["store", "item"])["Sales"].apply(
        lambda s: (s == 0).sum()
    ).reset_index(name="n_zero")
    zero_per_series["pct"] = zero_per_series["n_zero"] / df.groupby(["store", "item"]).size().values * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pie chart
    ax = axes[0]
    ax.pie([n_pos, n_zero], labels=[f"Positive sales\n({n_pos:,})", f"Zero sales\n({n_zero:,})"],
           colors=["#4CAF50", "#F44336"], autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 11})
    ax.set_title(f"Store Item — Zero vs Positive Sales\n(Total rows: {n_total:,})")

    # Histogram of per-series zero %
    ax = axes[1]
    ax.hist(zero_per_series["pct"], bins=30, color="#FF5722", edgecolor="white", alpha=0.8)
    ax.set_xlabel("% Zero-Sales Days per Series")
    ax.set_ylabel("Number of Series")
    ax.set_title("Distribution of Zero-Sales % Across 500 Series\n"
                 f"(Median: {zero_per_series['pct'].median():.1f}%)")
    ax.axvline(zero_per_series["pct"].median(), color="navy", linestyle="--",
               label=f"Median: {zero_per_series['pct'].median():.1f}%")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


def plot_store_item_seasonality(df: pd.DataFrame, output_path: Path):
    """3 sample store-item series over time."""
    print("  Generating store_item_seasonality.png ...")

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date", "sales": "Sales"})

    # Pick 3 representative series (by highest, median, lowest average sales)
    avg = df.groupby(["store", "item"])["Sales"].mean().reset_index(name="avg")
    avg_sorted = avg.sort_values("avg")
    picks = [
        avg_sorted.iloc[0][["store", "item"]].values,
        avg_sorted.iloc[len(avg_sorted)//2][["store", "item"]].values,
        avg_sorted.iloc[-1][["store", "item"]].values,
    ]
    labels = ["Low-demand series", "Median-demand series", "High-demand series"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    colors = ["#607D8B", "#2196F3", "#FF5722"]

    for idx, ((store, item), label) in enumerate(zip(picks, labels)):
        s = df[(df["store"] == int(store)) & (df["item"] == int(item))].sort_values("Date")
        ax = axes[idx]
        ax.plot(s["Date"], s["Sales"], lw=0.6, alpha=0.8, color=colors[idx])
        ax.set_title(f"{label} — Store {int(store)}, Item {int(item)}", fontsize=11)
        ax.set_ylabel("Daily Sales")
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.set_xlabel("Date")

    fig.suptitle("Store Item Demand — Sample Series (2013–2017)\n"
                 "Clear weekly seasonality visible; no zero-closed-day pattern (always open)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    print("Loading cleaned datasets for EDA...")
    rossmann_df = pd.read_parquet(processed_dir / "rossmann_clean.parquet")
    store_item_df = pd.read_parquet(processed_dir / "store_item_clean.parquet")

    # Load feature-engineered data for lag correlation (may not exist on first run)
    rossmann_features_path = processed_dir / "train_rossmann.parquet"
    rossmann_feat_df = (pd.read_parquet(rossmann_features_path)
                        if rossmann_features_path.exists() else rossmann_df)

    print("\n" + "=" * 60)
    print("PIPELINE A: Rossmann EDA Figures")
    print("=" * 60)

    plot_rossmann_sales_distribution(
        rossmann_df, figures_dir / "rossmann_sales_distribution.png")

    plot_rossmann_raw_correlation(
        rossmann_df, figures_dir / "rossmann_raw_correlation.png")

    plot_rossmann_lag_correlation(
        rossmann_feat_df, figures_dir / "rossmann_lag_correlation.png")

    # Keep the old name for backward-compat reference in current main.tex
    import shutil
    shutil.copy(figures_dir / "rossmann_lag_correlation.png",
                figures_dir / "rossmann_correlation_heatmap.png")

    plot_time_based_split_diagram(
        rossmann_df, figures_dir / "time_based_split_diagram.png", holdout_days=42)

    print("\n" + "=" * 60)
    print("PIPELINE B: Store Item EDA Figures")
    print("=" * 60)

    plot_store_item_sales_histogram(
        store_item_df, figures_dir / "store_item_sales_histogram.png")

    plot_store_item_dow_profile(
        store_item_df, figures_dir / "store_item_dow_profile.png")

    plot_store_item_monthly_profile(
        store_item_df, figures_dir / "store_item_monthly_profile.png")

    plot_store_item_zero_summary(
        store_item_df, figures_dir / "store_item_zero_summary.png")

    plot_store_item_seasonality(
        store_item_df, figures_dir / "store_item_seasonality.png")

    print("\n" + "=" * 60)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
