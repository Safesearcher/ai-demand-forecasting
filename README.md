# AI Demand Forecasting & Inventory Decision Support

## Overview

This project builds a web application that learns from historical sales data, forecasts near-future demand with an uncertainty range (P10, P50, and P90), and converts the forecast into a reorder recommendation using safety stock calculated from validation error. The system supports the manager's decision; it does not place orders automatically and does not replace a POS or ERP system.

### Deliverable Submission Details

- **Track:** Track A (Application Development)
- **Deliverable 1 Technical Report (LaTeX):** [Deliverable 1 Technical Report (LaTeX)](report/main.tex)
- **Deliverable 2 Milestone:** Model Architecture (Phase 3) & Empirical Evaluation (Phase 4)

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

## Phase 3: Training

Phase 3 trains separate probabilistic forecasting pipelines for Rossmann (A)
and Store Item (B). The Deliverable 1 holdout remains untouched for final
evaluation; internal validation is carved chronologically from the training
period.

Install the pinned modeling dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the configurable MLP:

```bash
python src/train.py --pipeline A --model mlp --horizon 7 --config configs/config.yaml
python src/train.py --pipeline B --model mlp --horizon 7 --config configs/config.yaml
```

Train the LightGBM quantile comparison model:

```bash
python src/train.py --pipeline A --model lgbm --horizon 7 --config configs/config.yaml
python src/train.py --pipeline B --model lgbm --horizon 7 --config configs/config.yaml
```

Run baseline models (Seasonal Naive 7 & Moving Average 28):

```bash
python src/baselines.py --pipeline all
```

Run a short shape-and-loss smoke check before a full run:

```bash
python src/train.py --pipeline A --model mlp --horizon 7 --config configs/config.yaml --smoke
```

Training writes model weights to `models/`, loss curves and holdout
predictions to `results/`, and never writes source data. Tune with:

```bash
python src/tune.py --pipeline A --model mlp --trials 15 --horizon 7 --config configs/config.yaml
```

Model weights download link: [Hugging Face Hub (`TahaImran1/ai-demand-forecasting-weights`)](https://huggingface.co/TahaImran1/ai-demand-forecasting-weights)

## Phase 4: Evaluation

Evaluate the available prediction artifacts for Pipeline A or Pipeline B:

```bash
python src/evaluate.py --pipeline A
python src/evaluate.py --pipeline B
```

Evaluation writes metrics, chronological split summaries, subgroup error
tables, and reports to `results/phase4/`. Publication-quality figures are
written to `report/figures/phase4/`.

Generate the Git contribution table from commit history:

```bash
python scripts/contribution_table.py
```

Stockout-alert classification requires validation prediction files to choose
the simulated-stock multiplier on validation only. The evaluator refuses to
choose that multiplier from the test set.

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
├── configs/
│   ├── config.yaml
│   ├── best_A_mlp.yaml
│   └── best_A_lgbm.yaml
├── scripts/
│   └── contribution_table.py
├── src/
│   ├── data.py
│   ├── models.py
│   ├── losses.py
│   ├── baselines.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── evaluation/
│       ├── config.py
│       ├── metrics.py
│       ├── splits.py
│       ├── stockout.py
│       ├── classification.py
│       ├── error_analysis.py
│       └── figures.py
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
│   ├── tables/
│   │   └── contribution_table.tex
│   └── figures/
│       ├── phase4/
│       └── ...
└── phase1.tex
```

## License

This project is for educational and demonstration purposes.
