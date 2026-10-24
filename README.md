# AI Demand Forecasting & Inventory Decision Support

## Overview

This project builds a web application that learns from historical sales data, forecasts near-future demand with an uncertainty range (P10, P50, and P90), and converts the forecast into a reorder recommendation using safety stock calculated from validation error. The system supports the manager's decision; it does not place orders automatically and does not replace a POS or ERP system.

### Deliverable 1 Submission Details

- **Track:** Track A (Application Development)
- **Technical Report:** [Deliverable 1 Technical Report (LaTeX)](report/main.tex)

### Team Members

| Name | Student ID | Core Contribution |
|---|---|---|
| Rai Fahd Sultan | 24L-2509 | Problem Definition & Data Pipelines |
| Muhammad Ayan | 24L-2568 | Data Cleaning & Feature Engineering |
| Taha Imran | 24L-2560 | Exploratory Data Analysis & Technical Report |

### Workflow

```
Sales data → validation → leakage-safe features → model (P10/P50/P90) → safety stock → reorder calculation → dashboard
```

## Datasets

This project uses two separate Kaggle datasets, maintained as strictly separate pipelines:

1. **Pipeline A: Rossmann Store Sales**
   - Competition: `rossmann-store-sales`
   - Files: `train.csv` (daily sales), `store.csv` (store metadata)
   - Location: `data/raw/rossmann/`

2. **Pipeline B: Store Item Demand Forecasting**
   - Competition: `demand-forecasting-kernels-only`
   - Files: `train.csv` (store-item daily sales)
   - Location: `data/raw/store_item/`

## Running the Data Scripts

The `data_scripts/` directory contains all Phase 2 data acquisition, cleaning, feature engineering, and EDA scripts (fulfilling the `/data scripts` deliverable requirement).

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Kaggle API credentials:
   - Option 1: Set `KAGGLE_API_TOKEN` environment variable
   - Option 2: Place `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<username>\.kaggle\`)
   - Get your API token from https://www.kaggle.com/settings

### One-Command Pipeline Execution

You can run the entire data pipeline from end to end using `run_all.py`:

```bash
# With real Kaggle data:
uv run python run_all.py
# Or with synthetic sample data (no credentials required):
uv run python run_all.py --sample
```

Alternatively, run each step individually:

### Step 1: Download Data

```bash
uv run python data_scripts/01_download_data.py
```

Optional: Generate synthetic sample data for testing without Kaggle credentials:
```bash
uv run python data_scripts/01_download_data.py --generate-sample
```

Outputs:
- `data/raw/rossmann/` (train.csv, store.csv)
- `data/raw/store_item/` (train.csv)
- `data/raw/acquisition_log.txt` & `data/raw/acquisition_log.json`

### Step 2: Clean and Validate

```bash
uv run python data_scripts/02_clean_and_validate.py
```

This script performs schema validation, handles missing values, removes duplicates, handles closed days, and caps outliers on training splits. Outputs:
- `data/processed/rossmann_clean.parquet`
- `data/processed/store_item_clean.parquet`
- `data/processed/cleaning_report.json`
- `data/processed/cleaning_summary.csv`

### Step 3: Feature Engineering

```bash
uv run python data_scripts/03_feature_engineering.py
```

This script creates leakage-safe temporal, cyclical, lag, and rolling features with strict time-based expanding/holdout splits (42 days for Rossmann, 90 days for Store Item). Outputs:
- `data/processed/rossmann_train_features.parquet` & `rossmann_val_features.parquet`
- `data/processed/store_item_train_features.parquet` & `store_item_val_features.parquet`
- `data/processed/feature_config.json`
- `models/scalers/` (StandardScaler fitted strictly on training data)

### Step 4: EDA Visualizations

```bash
uv run python data_scripts/04_eda_visualizations.py
```

This script generates 10 publication-quality plots (300 DPI) saved to `report/figures/`:
1. `rossmann_sales_distribution.png` (raw vs log, promo lift boxplot, store types)
2. `rossmann_raw_correlation.png` (correlations among raw variables)
3. `rossmann_correlation_heatmap.png` (engineered features correlation matrix)
4. `rossmann_lag_correlation.png` (autocorrelation structure across 28 lags)
5. `store_item_sales_histogram.png` (daily unit demand histogram)
6. `store_item_dow_profile.png` (day-of-week demand profile across stores)
7. `store_item_monthly_profile.png` (monthly seasonality across stores)
8. `store_item_seasonality.png` (multi-year longitudinal sales trends)
9. `store_item_zero_summary.png` (zero-demand incidence analysis)
10. `time_based_split_diagram.png` (time-based holdout validation schema)

## Important Disclaimers

1. **Inventory, lead times, and incoming orders are simulated/user-entered:** The datasets do not contain inventory levels or lead times. These values are simulated or user-entered and are labeled as such in the application.

2. **Datasets do not represent Pakistani retail:** Neither the Rossmann Store Sales nor the Store Item Demand Forecasting dataset represents Pakistani retail. We make no such claim. These are public Kaggle datasets used for methodology demonstration.

## Project Structure

```
.
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── run_all.py
├── data_scripts/
│   ├── 01_download_data.py
│   ├── 02_clean_and_validate.py
│   ├── 03_feature_engineering.py
│   └── 04_eda_visualizations.py
├── data/
│   ├── raw/
│   │   ├── rossmann/
│   │   ├── store_item/
│   │   ├── acquisition_log.txt
│   │   └── acquisition_log.json
│   └── processed/
│       ├── rossmann_clean.parquet
│       ├── store_item_clean.parquet
│       ├── rossmann_train_features.parquet
│       ├── rossmann_val_features.parquet
│       ├── store_item_train_features.parquet
│       ├── store_item_val_features.parquet
│       ├── feature_config.json
│       └── cleaning_report.json
├── models/
│   └── scalers/
│       ├── rossmann_standard_scaler.pkl
│       └── store_item_standard_scaler.pkl
├── report/
│   ├── main.tex
│   └── figures/
│       ├── rossmann_sales_distribution.png
│       ├── rossmann_raw_correlation.png
│       ├── rossmann_correlation_heatmap.png
│       ├── rossmann_lag_correlation.png
│       ├── store_item_sales_histogram.png
│       ├── store_item_dow_profile.png
│       ├── store_item_monthly_profile.png
│       ├── store_item_seasonality.png
│       ├── store_item_zero_summary.png
│       └── time_based_split_diagram.png
└── phase1.tex
```

## License

This project is for educational and demonstration purposes.
