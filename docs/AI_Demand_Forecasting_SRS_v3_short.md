# SRS — AI Demand Forecasting & Inventory Decision Support

**Track:** A — Application Development | **Type:** Web-based AI/ML app | **Users:** Small/medium distributors and multi-SKU retailers

---

## 1. Purpose

A web app that learns from historical sales data, forecasts near-future demand with an uncertainty range, and turns the forecast into a reorder recommendation.

> **What will likely sell in the next few days/weeks, how uncertain is that, and how much should I reorder?**

It does **not** replace a POS/ERP. It consumes sales data from a CSV export.

**Workflow:**
```text
Sales data → validation → leakage-safe features → model (P10/P50/P90)
  → safety stock from forecast error → reorder calculation → dashboard
```

**Architecture (MVP):**
```text
[Google Colab]  training + evaluation (offline)
      │  saves model files, metrics, feature config (versioned)
      ▼
[Streamlit app]  one Python app: CSV upload/validation, feature building,
                 forecasting, reorder logic, dashboard
      │
      ▼
[Local storage]  SQLite (or parquet files): sales history, inventory inputs,
                 recommendation decisions, active model version
```
The app loads saved model artifacts and never depends on a running Colab session. There is no separate backend API in the MVP; the app imports the same feature and reorder code (`src/`) used in training.

---

## 2. Datasets (two separate pipelines, never merged)

| | A: Rossmann Store Sales | B: Store Item Demand |
|---|---|---|
| Kaggle | `rossmann-store-sales` | `demand-forecasting-kernels-only` |
| Granularity | Store-level daily sales (1,115 stores) | Store × item (10 stores × 50 items, ~5 yrs) |
| Features | Promo, holidays, competition, store info | Date and sales only |
| Role | Feature-rich modeling and benchmarking | Product-level reorder demo |

**Handling rules**
- Neither dataset has public test labels usable for us → evaluate on a **time-based holdout** from the training file.
- Rossmann: closed days (`Open = 0`) handled explicitly; `Customers` is **not** a same-day feature (unknown in advance), only lagged values.
- Neither dataset has inventory, lead times or incoming orders → these are **simulated or user-entered** and labeled as such everywhere.
- Neither is Pakistani data; do not claim it represents Pakistani retail.

---

## 3. Modeling

- **Outputs:** P10 / P50 / P90 demand per series and date.
- **Strategy:** direct multi-horizon forecasting; features built from the forecast origin, no information from after it.
- **Models:** baselines (seasonal naive, moving average) vs LightGBM (quantile regression for P10/P50/P90); XGBoost optional.
- **Metrics:** MAE, RMSE, WMAPE/MASE, P10–P90 interval coverage (target ≈ 80%).
- **Validation:** time-based only (expanding window). No random splits.

---

## 4. Reorder Logic

| Symbol | Meaning |
|---|---|
| `L`, `R` | Lead time, review period (days); `T = L + R` |
| `D_T` | P50 forecast demand over `T` days |
| `σ_d` | Daily validation RMSE for the series |
| `z` | Service-level factor (1.65 ≈ 95%) |
| `S`, `I` | Stock on hand, incoming orders |

```text
Safety stock:   SS = z × σ_d × √T
Order quantity: Q  = max(0, ceil(D_T + SS − S − I))
```

**Worked example:** L = 3, R = 7 → T = 10. D_T = 120, σ_d = 5.7, z = 1.65 → SS ≈ 30. S = 40, I = 0 → **Q = 120 + 30 − 40 = 110 units**.

**Stockout-risk alert:** flag a series when `S + I` < P90 demand over `L`.

The manager can accept, reject, or adjust any recommendation. No automatic ordering.

---

## 5. Data Updates and Retraining

- Initial setup: import history → build features → train in Colab → save versioned model artifacts that the app loads.
- Daily: append new records (stored in SQLite), recompute features, regenerate forecasts with the active model. **No retraining per sale.**
- Retraining: run offline in Colab as a manual step (e.g. weekly, frequency decided experimentally), producing a new model version. The app switches the active version only if the new one is not worse than the current one on a recent time-based window. Retraining inside the hosted app is out of scope for the MVP.
- Integration: **CSV upload (MVP)**; database connection and a REST API layer (e.g. FastAPI) are future work.

---

## 6. User Interface

- **Dashboard:** series needing attention, reorder recommendations, stockout alerts, history vs forecast chart with P10–P90 band.
- **Item/Store detail:**
```text
Item 17 @ Store 3
Stock 40 | Incoming 0 | L/R 3/7 days
Forecast demand (10 days): 120 (P10–P90: 98–145)
Safety stock (95%): 30   →   ORDER 110 UNITS  ⚠ Reorder recommended
```
- **Evaluation page:** baseline vs ML metrics, interval coverage, error by horizon.
- Demo screens must state: *"Stock and lead-time values are simulated/user-entered."*

---

## 7. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Upload historical sales CSV (authorized users). |
| FR-02 | Validate: missing columns, invalid dates, duplicates, invalid numbers. |
| FR-03 | Persist imported data. |
| FR-04 | Train ML model(s) and compare against baselines using time-based validation. |
| FR-05 | Generate P10/P50/P90 forecasts for a configurable horizon, without leakage. |
| FR-06 | Append new daily records and regenerate forecasts. |
| FR-07 | Enter/import stock, incoming orders, lead time, review period, service level (labeled user-entered/simulated). |
| FR-08 | Compute safety stock and order quantity per Section 4. |
| FR-09 | Flag stockout-risk series. |
| FR-10 | Display history, forecasts with intervals, recommendations. |
| FR-11 | Display evaluation metrics for all models. |
| FR-12 | Let users accept, reject or adjust recommendations. |
| FR-13 | Store model versions and let an authorized user switch the active version; new versions (trained offline in Colab) are activated only after validation. |

---

## 8. Non-Functional Requirements

- Deployable online as a single Streamlit application (UI, inference and reorder logic in one app).
- Typical predictions return in a few seconds (cached forecasts allowed).
- Stored data (SQLite/parquet) must survive app restarts: use a persistent volume or hosted database if the host's disk is ephemeral.
- Training and serving use the same feature code and pinned library versions, so saved models load and behave identically in the app.
- Data isolated per business if multi-user is implemented.
- No large datasets in GitHub; provide download scripts.
- Reproducible training (fixed seeds, pinned dependencies, documented split dates, leakage check).

---

## 9. Scope

**Must have:** both pipelines (preprocessing, features, baseline + LightGBM, time-based evaluation), P10/P50/P90 forecasts, reorder + stockout logic, dashboard, CSV import, online deployment.

**Stretch (only after evaluation is complete):** scheduled retraining, database integration, REST API layer, multi-user, email alerts, LLM explanations.

**Out of scope:** POS, accounting, payments/purchasing, supplier or warehouse management, autonomous ordering.

---

## 10. Academic Framing

Demonstrates the full ML lifecycle: data acquisition, cleaning, leakage-safe feature engineering, training, baseline comparison, time-aware validation, interval evaluation, deployment, and turning probabilistic forecasts into an inventory decision. The contribution is the complete, evaluated application, not a new algorithm.

Present it as **AI Demand Forecasting and Inventory Decision Support**, not "Sales Prediction."
