# AI Demand Forecasting & Inventory Decision Support

## What
An end-to-end operational decision support system that produces probabilistic daily demand forecasts (P10, P50, and P90) and translates them into periodic-review (R, S) replenishment recommendations with safety stock derived from empirical error.

## Scope
- In: Leakage-safe data pipelines, PyTorch QuantileMLP, LightGBM Quantile Ensemble, baselines, multi-quantile pinball loss, time-aware empirical validation, stockout alerting classification, FastAPI backend, and React 18 frontend with zero hardcoded mock data.
- Out: Autonomous ordering, point-of-sale (POS) integration, and ERP replacements.

## Stack
- Backend: Python 3.12+, pandas, NumPy, PyTorch 2.5, LightGBM 4.5, scikit-learn, FastAPI, Uvicorn, Pydantic.
- Frontend: React 18, Vite 6, Chart.js 4, Lucide React, CSS Custom Properties (Light/Dark mode).

## Structure
- `src/api.py`: FastAPI backend service with live inference, evaluation data delivery, and CSV validation.
- `src/data.py`: Multi-horizon feature engineering and chronological splitting.
- `src/models.py`: QuantileMLP (PyTorch) and LightGBMQuantileModel.
- `src/losses.py`: Summed pinball loss.
- `src/baselines.py`: Seasonal-naive (7d) and moving-average (28d) baselines.
- `src/train.py`: Model training CLI and artifact generation.
- `src/tune.py`: Hyperparameter exploration.
- `src/evaluate.py`: Empirical holdout evaluation and subgroup metrics.
- `frontend/`: React 18 + Vite web application with responsive decision cockpit and zero mock data.
- `configs/`: YAML configuration files.
- `models/`: Model weights and scalers.
- `results/`: Training curves, tuning tables, and Phase 4 metrics.

## Status & Progress
- [x] Phase 1: Problem Definition & Mathematical Formalization
- [x] Phase 2: Data Acquisition, Cleaning & Leakage-Safe Feature Engineering
- [x] Phase 3: Probabilistic Model Architecture & Hyperparameter Tuning
- [x] Phase 4: Time-Based Empirical Evaluation & Stockout Classification
- [x] Phase 5: React.js Frontend & FastAPI Backend Live Deployment
- [ ] Phase 6: Final Comprehensive LaTeX Technical Report (Assigned to peer group member)

## Open Issues / Notes
- Deliverable 2 model weights are hosted on Hugging Face: https://huggingface.co/TahaImran1/ai-demand-forecasting-weights
- Frontend successfully runs on `http://localhost:5173` via Vite with reverse proxy to FastAPI on `http://localhost:8000`.
- Report Deliverable 3 LaTeX compilation is assigned to peer team member.
