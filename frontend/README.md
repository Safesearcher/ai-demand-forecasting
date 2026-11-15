# AI Demand Forecasting & Inventory Decision Support — React Frontend

A production React.js web application for probabilistic retail demand forecasting and automated inventory decision support, fulfilling Deliverable 3 requirements and built strictly to the specifications of `AI_Demand_Forecasting_SRS_v3_short.md`.

---

## 1. Product Overview

Traditional retail planning relies on single-point heuristics that hide variance and lead directly to costly stockouts or bloated carrying costs. **AI Demand Forecasting and Inventory Decision Support** provides an end-to-end operational decision cockpit that pairs machine-learned probabilistic quantile forecasts ($P_{10}$, $P_{50}$, $P_{90}$) with standard periodic-review $(R, S)$ inventory safety stock calculations.

### Core Mathematical Engine (SRS Section 4)
The application computes optimal replenishment quantities $Q$ and evaluates stockout vulnerabilities in real time:

$$\text{Protection Interval: } T = L + R$$
$$\text{Safety Stock: } SS = z \times \sigma_d \times \sqrt{T}$$
$$\text{Recommended Order Quantity: } Q = \max\left(0, \left\lceil D_T + SS - S - I \right\rceil\right)$$
$$\text{Stockout Risk Alert: flagged if } (S + I) < P_{90} \text{ lead-time demand over } L$$

---

## 2. Tech Stack & Zero Hardcoded Data Architecture

The frontend is built with a modern React.js + Vite stack and **contains zero hardcoded mock or demo data**. All metrics, time-series observations, model quantiles, and registry states are fetched dynamically from the FastAPI backend service (`src/api.py`).

- **Framework**: React 18 (Vite 6)
- **State Management & Data Flow**: Reactive hooks (`useState`, `useEffect`) communicating with `/api/...` through `src/api/client.js`.
- **Charting Engine**: Chart.js v4 configured with luminous quantile uncertainty area fills ($P_{10}$ to $P_{90}$) and $P_{50}$ median trendlines.
- **Iconography**: Lucide React.
- **Design Tokens & Theme**: Defined in `src/styles.css` using CSS custom properties with light/dark theme toggling and tabular numeric alignment (`font-variant-numeric: tabular-nums`).

---

## 3. Views & Functional Coverage

| View | SRS ID | Capabilities |
|---|---|---|
| **Executive Dashboard** | FR-09, FR-10 | Real-time KPI summaries (critical stockouts, reorders needed, healthy items), multi-status filter, series overview table, and interactive visual forecast preview. |
| **Inventory Decision Cockpit** | FR-07, FR-08, FR-12 | Series selector, live sliders for $L, R, S, I, z$, instant mathematical recalculation of $SS$ and $Q$, stockout risk alerts, and human-in-the-loop decision recording (accept/reject/adjust). |
| **Empirical Evaluation** | FR-04, FR-11 | Real Phase 4 benchmark tables (LightGBM vs PyTorch QuantileMLP vs baselines: MAE, RMSE, WMAPE, Pinball Losses, 80% Coverage), stockout classification metrics ($F_1$, AUC-ROC), and error-by-horizon curves. |
| **Data Ingestion Gate** | FR-01, FR-02 | CSV drag-and-drop uploader, schema validation (Store Item vs Rossmann), duplicate checking, date format validation, negative value checks, and row preview. |
| **Model Registry** | FR-13 | Active model versions, holdout metrics, architecture details, and live production switching between LightGBM and PyTorch models. |

---

## 4. How to Run

### Step 1: Start the Backend API (FastAPI)
From the repository root:
```bash
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Start the React Frontend (Vite)
From the `frontend/` directory:
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser. All requests to `/api` are automatically proxied to `http://localhost:8000`.

### Production Build
```bash
cd frontend
npm run build
```
Build artifacts are placed in `frontend/dist/`.
