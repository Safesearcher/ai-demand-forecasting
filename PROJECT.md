# AI Demand Forecasting & Inventory Decision Support

## What
Phase 3 adds separate probabilistic forecasting pipelines for Rossmann (A)
and Store Item (B), producing P10, P50, and P90 daily demand forecasts.

## Scope
In: leakage-safe data preparation, PyTorch MLP, LightGBM comparison,
pinball loss, baselines, training, smoke checks, and tuning.
Out: production deployment, dashboard work, and changes to Deliverable 1 data/report scripts.

## Stack
Python 3.12, pandas, NumPy, PyTorch, LightGBM, scikit-learn, PyYAML,
Parquet, matplotlib. Windows uses pathlib and DataLoader num_workers=0.

## Structure
- `src/data.py`: horizon-safe features, chronological splits, encoders, scalers.
- `src/models.py`: QuantileMLP and LightGBM quantile wrapper.
- `src/losses.py`: summed pinball loss.
- `src/baselines.py`: seasonal-naive and moving-average baselines.
- `src/train.py`: model training CLI and artifacts.
- `src/tune.py`: random-search tuning and trial tables.
- `configs/`: training and tuning configuration.
- `models/`: ignored model weights and existing scalers.
- `results/`: ignored curves and predictions.

## Run and test
- Install: `python -m pip install -r requirements.txt`
- Smoke MLP: `python src/train.py --pipeline B --model mlp --horizon 7 --config configs/config.yaml --smoke`
- Smoke LightGBM: `python src/train.py --pipeline B --model lgbm --horizon 7 --config configs/config.yaml --smoke`
- Tune: `python src/tune.py --pipeline A --model mlp --trials 15 --horizon 7 --config configs/config.yaml`

## Decisions
- Pipelines remain separate: dataset schemas and series IDs differ.
- Holdouts remain untouched: validation is carved from the end of training data.
- History features use data through `t-h`: same-day Customers is never used.
- Original-unit predictions are clipped at zero; Rossmann closed days are zero.

## Plan
Status: [x] Phase 3 implementation and smoke gate complete.
- [x] Data preparation and leakage controls [L]
- [x] Model architectures [M]
- [x] Losses and baselines [M]
- [x] Training CLI and artifacts [L]
- [x] Hyperparameter tuning [L]
- [x] Housekeeping and smoke validation [M]

## Current
Phase 4 evaluation modules and stockout classification are fully implemented and passing.
Tuning trial exploration is executed and recorded in `results/tuning_*.csv` with best parameter sets in `configs/best_*.yaml`.

## Open issues
- Deliverable 2 model weights are hosted on Hugging Face: https://huggingface.co/TahaImran1/ai-demand-forecasting-weights
- Report Deliverable2_Report.pdf includes all Phase 3 & 4 sections, architecture diagrams, loss curves, and stockout metrics.
