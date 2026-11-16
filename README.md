# AI Demand Forecasting & Inventory Decision Support

## Overview

This project delivers an end-to-end, production-grade AI system for probabilistic retail demand forecasting and automated inventory decision support. The application learns from historical daily sales records, forecasts multi-horizon demand with explicit uncertainty intervals ($P_{10}$, $P_{50}$, and $P_{90}$ quantiles), and translates forecast distributions into optimal $(R, S)$ replenishment recommendations with safety stock derived from empirical validation error.

The system is designed as a human-in-the-loop operational decision cockpit: it assists store and inventory managers in making proactive reorder decisions, prevents stockouts, reduces excess carrying costs, and does not autonomously place unverified purchase orders.

### Deliverable Submission Details

- **Track:** Track A (Application Development)
- **Deliverable 1 Milestone:** Problem Definition (Phase 1) & Data Foundations (Phase 2)
- **Deliverable 2 Milestone:** Model Architecture (Phase 3) & Empirical Evaluation (Phase 4)
- **Deliverable 3 Milestone:** Final Deployment (Phase 5) & Technical Report (Phase 6)
- **Technical Report (LaTeX):** [`report/main.tex`](report/main.tex)

### Team Members

| Name | Student ID | Core Contribution |
|---|---|---|
| Rai Fahd Sultan | 24L-2509 | Problem Definition & Data Pipelines |
| Muhammad Ayan | 24L-2568 | Data Cleaning & Feature Engineering |
| Taha Imran | 24L-2560 | Exploratory Data Analysis & Technical Report |

### System Workflow

```
Historical Sales Data (CSV)
       │
       ▼
Schema & Integrity Validation (FR-01, FR-02)
       │
       ▼
Leakage-Safe Feature Engineering (Lags, Rolling Stats, Calendar Cycles)
       │
       ▼
Probabilistic Multi-Quantile Model (P10 / P50 / P90 via LightGBM or PyTorch QuantileMLP)
       │
       ▼
Periodic-Review Safety Stock Engine: SS = z × σ_d × √(L + R)
       │
       ▼
Replenishment Recommendation: Q = max(0, ⌈D_T + SS - S - I⌉)
       │
       ▼
React Decision Cockpit & Stockout Alert Monitoring (Human-in-the-Loop)
```

---

## Phase 5: Live Web Application (React + FastAPI)

Deliverable 3 establishes a decoupled, client-server web application architecture adhering strictly to the SRS specifications (`AI_Demand_Forecasting_SRS_v3_short.md`). **The frontend contains zero hardcoded demo/mock datasets**; all predictions, time-series observations, metrics, and models are served dynamically by the Python backend.

### Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│               React 18 Frontend (Vite)                 │
│  • Executive Dashboard (KPIs & Quantile Bands)         │
│  • Inventory Cockpit (Sliders for L, R, S, I, z)       │
│  • Empirical Evaluation (Phase 4 Holdout Benchmarks)   │
│  • CSV Ingestion Gate (FR-01 / FR-02 Schema Checks)    │
│  • Model Registry (FR-13 Version Management & Switch)  │
└───────────────────────────┬────────────────────────────┘
                            │ REST / JSON (Reverse Proxy)
                            ▼
┌────────────────────────────────────────────────────────┐
│                 FastAPI Backend Service                │
│  • GET  /api/health                                    │
│  • GET  /api/series & /api/series/{id}                 │
│  • POST /api/reorder/calculate & /api/reorder/decision │
│  • GET  /api/evaluation & GET /api/models              │
│  • POST /api/models/switch & POST /api/upload          │
└──────────────┬──────────────────────────┬──────────────┘
               ▼                          ▼
   LightGBM / PyTorch Quantile     Parquet Clean Data &
       Inference Engine            Phase 4 Metrics CSVs
```

### Application Features & SRS Compliance

1. **Executive Decision Dashboard (FR-09, FR-10):**
   - High-level KPI status cards displaying monitored series, critical stockout alerts, reorder recommendations, and healthy SKU counts.
   - Interactive series table with instant status filtering (`All`, `Critical Risk`, `Reorder`, `Healthy`).
   - Visual forecast preview showing historical sales along with illuminated $P_{10} \text{--} P_{90}$ quantile uncertainty bands and $P_{50}$ median trendlines.

2. **Inventory Decision Cockpit (FR-07, FR-08, FR-12):**
   - Interactive parameters for Lead Time ($L$), Review Period ($R$), Stock on Hand ($S$), Incoming Orders ($I$), and Service Level ($z$).
   - Live mathematical formula breakdown:
     $$\text{Protection Interval: } T = L + R$$
     $$\text{Safety Stock: } SS = z \times \sigma_d \times \sqrt{T}$$
     $$\text{Recommended Order: } Q = \max\left(0, \left\lceil D_T + SS - S - I \right\rceil\right)$$
   - Real-time stockout alert triggered when $(S + I) < P_{90} \text{ demand over } L$.
   - Human-in-the-loop decision buttons: **Accept $Q$**, **Reject**, or submit an **Adjusted Custom Quantity** with persistent logging.

3. **Empirical Evaluation Dashboard (FR-04, FR-11):**
   - Real Phase 4 benchmark tables across pipelines: LightGBM vs. PyTorch QuantileMLP vs. Seasonal Naive vs. Moving Average.
   - Holdout metrics: MAE, RMSE, WMAPE, Pinball Losses ($P_{10}, P_{50}, P_{90}$), 80% Central Interval Coverage, and Mean Interval Width.
   - Stockout classification diagnostic metrics (Precision, Recall, $F_1$, AUC-ROC, Confusion Matrix).
   - Multi-horizon degradation curve (MAE progression across Days 1 through 14).
   - Subgroup fairness error breakdowns (store types, assortments, promotions).

4. **Data Ingestion & Schema Gate (FR-01, FR-02):**
   - Drag-and-drop CSV file uploader validating against Store Item Demand and Rossmann schemas.
   - Verifies column presence, date parsing (YYYY-MM-DD), chronological ordering, zero duplicate series timestamps, and non-negative sales values.
   - Downloadable valid and invalid test CSV templates.

5. **Model Registry & Dynamic Switching (FR-13):**
   - Version-controlled model registry displaying model names, architectures, training dates, validation windows, and performance metrics.
   - Single-click production switching between active models (e.g., LightGBM and PyTorch QuantileMLP).

---

## How to Run the Full Application

### 1. Prerequisites & Environment Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/Safesearcher/ai-demand-forecasting.git
cd ai-demand-forecasting

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend Service

From the root directory:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The backend API will start at `http://localhost:8000`. You can inspect the interactive OpenAPI Swagger documentation at `http://localhost:8000/docs`.

### 3. Start the React Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The React frontend will be accessible at **`http://localhost:5173`**. Requests to `/api/*` are automatically proxied to the FastAPI server at `http://localhost:8000`.

### 4. Or Run via Unified Launcher

Launch both backend and frontend concurrently with a single command:

```bash
python run_app.py
```

### 5. Automated Unit Tests

Run the full automated test suite verifying replenishment math, CSV ingestion validation gates, and model monotonicity:

```bash
python -m unittest discover -s tests -v
```

### 6. Build for Production

To create an optimized production build of the React application:

```bash
cd frontend
npm run build
```

Production assets will be generated in `frontend/dist/`.

---

## Datasets

This project maintains two separate Kaggle datasets as strictly independent pipelines (never merged):

1. **Pipeline A: Rossmann Store Sales**
   - Source: `rossmann-store-sales`
   - Files: `train.csv` (daily sales), `store.csv` (store metadata)
   - Scope: Store-level sales across 1,115 stores with promotional indicators, state/school holidays, and competitor distance. Contiguous 42-day chronological holdout.
   - Location: `data/raw/rossmann/`

2. **Pipeline B: Store Item Demand Forecasting**
   - Source: `demand-forecasting-kernels-only`
   - Files: `train.csv` (store-item daily sales)
   - Scope: Item-level daily demand across 10 stores $\times$ 50 items (500 time series, 5 years). Contiguous 90-day chronological holdout.
   - Location: `data/raw/store_item/`

---

## Data Pipelines & Preprocessing (Phase 2)

All Phase 2 data acquisition, cleaning, feature engineering, and exploratory data analysis scripts are located in `data_scripts/`.

### One-Command Pipeline Execution

```bash
# With Kaggle credentials:
python run_all.py

# Without Kaggle credentials (generates synthetic sample dataset):
python run_all.py --sample
```

### Step-by-Step Execution

1. **Download Data / Generate Samples:**
   ```bash
   python data_scripts/01_download_data.py --generate-sample
   ```
2. **Clean and Validate (Leakage-free outlier capping on train split only):**
   ```bash
   python data_scripts/02_clean_and_validate.py
   ```
3. **Feature Engineering (Lags, rolling means/stds, calendar/cyclical features):**
   ```bash
   python data_scripts/03_feature_engineering.py
   ```
4. **Generate Publication-Quality EDA Visualizations (300 DPI):**
   ```bash
   python data_scripts/04_eda_visualizations.py
   ```

---

## Model Training & Tuning (Phase 3)

Train the models using the training CLI (`src/train.py`):

```bash
# Train PyTorch QuantileMLP
python src/train.py --pipeline A --model mlp --horizon 7 --config configs/config.yaml
python src/train.py --pipeline B --model mlp --horizon 7 --config configs/config.yaml

# Train LightGBM Multi-Quantile Ensemble
python src/train.py --pipeline A --model lgbm --horizon 7 --config configs/config.yaml
python src/train.py --pipeline B --model lgbm --horizon 7 --config configs/config.yaml

# Evaluate Baselines (Seasonal Naive 7 & Moving Average 28)
python src/baselines.py --pipeline all
```

Trained model weights are hosted on the Hugging Face Hub:
👉 **[Hugging Face Hub (`TahaImran1/ai-demand-forecasting-weights`)](https://huggingface.co/TahaImran1/ai-demand-forecasting-weights)**

---

## Empirical Evaluation (Phase 4)

Run the comprehensive evaluation suite across holdout splits:

```bash
python src/evaluate.py --pipeline A
python src/evaluate.py --pipeline B
```

Outputs are saved in `results/phase4/` (metrics, subgroup breakdowns, classification diagnostics) and `report/figures/phase4/` (calibration plots, prediction interval bands).

Generate the Git contribution table:
```bash
python scripts/contribution_table.py
```

---

## Important Disclaimers

1. **Simulation Notice:**
   > *"Stock and lead-time values are simulated/user-entered."*
   The underlying benchmark datasets do not contain inventory levels, supplier lead times, or pending purchase orders. These parameters are simulated or entered by the user in the decision cockpit and are clearly labeled as such across all user-facing views.

2. **Benchmark Geographic Disclaimer:**
   > Neither the Rossmann Store Sales nor the Store Item Demand Forecasting dataset represents Pakistani retail. These are public Kaggle evaluation benchmarks used strictly for machine learning methodology demonstration.

---

## Project Structure

```
.
├── README.md
├── PROJECT.md
├── requirements.txt
├── .gitignore
├── .env.example
├── run_all.py
├── run_app.py                    # Unified FastAPI + React dev launcher
├── tests/                        # Automated unit test suite (Deliverable 3)
│   ├── test_reorder_logic.py     # Replenishment math & stockout tests
│   ├── test_csv_validation.py    # Schema gate & integrity tests
│   └── test_model_inference.py   # Model monotonicity & quantile sorting
├── configs/
│   ├── config.yaml
│   ├── best_A_mlp.yaml
│   ├── best_A_lgbm.yaml
│   ├── best_B_mlp.yaml
│   └── best_B_lgbm.yaml
├── scripts/
│   └── contribution_table.py
├── src/
│   ├── api.py                    # FastAPI backend service (Phase 5)
│   ├── data.py                   # Data loaders & leakage-safe feature transformations
│   ├── models.py                 # QuantileMLP & LightGBMQuantileModel
│   ├── losses.py                 # Multi-quantile Pinball Loss
│   ├── baselines.py              # Seasonal Naive & Moving Average baselines
│   ├── train.py                  # Training CLI & artifact serialization
│   ├── tune.py                   # Random search hyperparameter exploration
│   ├── evaluate.py               # Time-aware empirical evaluator
│   └── evaluation/
│       ├── config.py
│       ├── metrics.py
│       ├── splits.py
│       ├── stockout.py
│       ├── classification.py
│       ├── error_analysis.py
│       └── figures.py
├── frontend/                     # React 18 + Vite Web Application (Phase 5)
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── README.md
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── styles.css
│       ├── api/
│       │   └── client.js         # Zero-mock live backend API client
│       └── components/
│           ├── Header.jsx
│           ├── DisclaimerBanner.jsx
│           ├── Toast.jsx
│           ├── views/
│           │   ├── DashboardView.jsx
│           │   ├── ItemDetailView.jsx
│           │   ├── EvaluationView.jsx
│           │   ├── UploadView.jsx
│           │   └── ModelsView.jsx
│           └── charts/
│               ├── QuantileForecastChart.jsx
│               └── HorizonErrorChart.jsx
├── data_scripts/
│   ├── 01_download_data.py
│   ├── 02_clean_and_validate.py
│   ├── 03_feature_engineering.py
│   └── 04_eda_visualizations.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── scalers/
│   ├── a_lgbm_p*.txt
│   ├── b_lgbm_p*.txt
│   ├── a_mlp.pt
│   └── b_mlp.pt
├── results/
│   ├── phase4/
│   ├── curves_*.csv
│   └── tuning_*.csv
├── report/
│   ├── main.tex                  # Comprehensive LaTeX Technical Report
│   ├── tables/
│   └── figures/
└── docs/
    ├── AI_Demand_Forecasting_SRS_v3_short.md
    ├── Deliverable 1.md
    ├── Deliverable 2.md
    └── Deliverable 3.md
```

---

## License

This project is developed for educational and academic research demonstration purposes.
