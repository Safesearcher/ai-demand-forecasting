"""Train and evaluate Phase 3 probabilistic forecasting models."""

from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for src.train. Install requirements.txt first."
    ) from exc

try:
    from .data import PreparedData, create_dataloader, prepare_data
    from .losses import pinball_loss
    from .models import LightGBMQuantileModel, QuantileMLP
except ImportError:
    from data import PreparedData, create_dataloader, prepare_data
    from losses import pinball_loss
    from models import LightGBMQuantileModel, QuantileMLP


@dataclass
class TrainingResult:
    model: Any
    curves: pd.DataFrame
    best_epoch: int
    best_validation_loss: float


def set_random_seed(seed: int) -> None:
    """Set all relevant random generators for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "training": {
            "epochs": 50,
            "batch_size": 1024,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "patience": 8,
            "min_delta": 1e-5,
        },
        "mlp": {
            "hidden_sizes": [256, 128, 64],
            "dropout": 0.2,
            "embedding_dims": {},
        },
        "lgbm": {
            "num_boost_round": 2000,
            "early_stopping_rounds": 100,
            "params": {
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
                "feature_fraction": 0.9,
            },
        },
    }


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration, applying safe defaults for omitted options."""
    if not path.exists():
        return _default_config()
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return _deep_update(_default_config(), loaded)


def _limit_for_smoke(
    frame: pd.DataFrame,
    series_columns: list[str],
    rows: int = 4096,
) -> pd.DataFrame:
    series_columns = list(series_columns)
    if len(frame) <= rows:
        return frame
    return (
        frame.sort_values("Date")
        .groupby(series_columns, sort=False, group_keys=False)
        .head(max(1, rows // max(1, frame[series_columns].drop_duplicates().shape[0])))
        .reset_index(drop=True)
    )


def _run_mlp_epoch(
    model: QuantileMLP,
    loader: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    for categorical_ids, numeric_features, targets in loader:
        categorical_ids = categorical_ids.to(device)
        numeric_features = numeric_features.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            predictions = model(categorical_ids, numeric_features)
            loss = pinball_loss(predictions, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_rows = len(targets)
        total_loss += float(loss.detach().cpu())
        total_rows += batch_rows
    return total_loss / max(1, total_rows)


def _predict_mlp(
    model: QuantileMLP,
    loader: Any,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for categorical_ids, numeric_features, _ in loader:
            output = model(
                categorical_ids.to(device),
                numeric_features.to(device),
            )
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def train_mlp(
    data: PreparedData,
    config: dict[str, Any],
    pipeline: str,
    horizon: int,
    output_dir: Path,
    smoke: bool = False,
) -> TrainingResult:
    """Train the MLP on train data and early-stop on internal validation."""
    del horizon
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_config = config["training"]
    epochs = 2 if smoke else int(training_config["epochs"])
    train_frame = _limit_for_smoke(data.train, data.spec.series_id_cols) if smoke else data.train
    validation_frame = (
        _limit_for_smoke(data.validation, data.spec.series_id_cols) if smoke else data.validation
    )
    train_loader = create_dataloader(
        train_frame,
        data,
        int(training_config["batch_size"]),
        shuffle=True,
    )
    validation_loader = create_dataloader(
        validation_frame,
        data,
        int(training_config["batch_size"]),
        shuffle=False,
    )
    model = QuantileMLP(
        data.categorical_sizes,
        len(data.numeric_columns),
        embedding_dims=config["mlp"].get("embedding_dims"),
        hidden_sizes=config["mlp"]["hidden_sizes"],
        dropout=float(config["mlp"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_mlp_epoch(model, train_loader, device, optimizer)
        validation_loss = _run_mlp_epoch(model, validation_loader, device, None)
        scheduler.step(validation_loss)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - float(training_config["min_delta"]):
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= int(training_config["patience"]):
                break

    if best_state is None:
        raise RuntimeError("MLP training produced no best checkpoint.")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"{pipeline.lower()}_mlp.pt")
    return TrainingResult(model, pd.DataFrame(rows), best_epoch, best_loss)


def _lgbm_features(data: PreparedData, frame: pd.DataFrame) -> pd.DataFrame:
    return frame[data.categorical_columns + data.numeric_columns].copy()


def train_lgbm(
    data: PreparedData,
    config: dict[str, Any],
    pipeline: str,
    horizon: int,
    output_dir: Path,
    smoke: bool = False,
) -> TrainingResult:
    """Train three LightGBM quantile models on log-sales targets."""
    del horizon
    train_frame = _limit_for_smoke(data.train, data.spec.series_id_cols) if smoke else data.train
    validation_frame = (
        _limit_for_smoke(data.validation, data.spec.series_id_cols) if smoke else data.validation
    )
    lgbm_config = config["lgbm"]
    model = LightGBMQuantileModel(
        params=lgbm_config["params"],
        num_boost_round=20 if smoke else int(lgbm_config["num_boost_round"]),
        early_stopping_rounds=5 if smoke else int(lgbm_config["early_stopping_rounds"]),
    )
    model.fit(
        _lgbm_features(data, train_frame),
        np.log1p(train_frame[data.spec.target_col].to_numpy()),
        _lgbm_features(data, validation_frame),
        np.log1p(validation_frame[data.spec.target_col].to_numpy()),
    )
    validation_predictions = model.predict(_lgbm_features(data, validation_frame))
    errors = validation_frame[data.spec.target_col].to_numpy()[:, None] - validation_predictions
    taus = np.asarray((0.1, 0.5, 0.9))
    validation_loss = float(np.maximum(taus * errors, (taus - 1) * errors).mean())
    train_loss = model.models[1].evals_result_["train"]["quantile"]
    valid_loss = model.models[1].evals_result_["valid"]["quantile"]
    curves = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_loss) + 1)),
            "train_loss": train_loss,
            "validation_loss": valid_loss,
        }
    )
    model.save(output_dir, pipeline)
    return TrainingResult(model, curves, int(model.models[1].best_iteration_), validation_loss)


def _save_curve(curves: pd.DataFrame, results_dir: Path, pipeline: str, model_name: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(results_dir / f"curves_{pipeline}_{model_name}.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(curves["epoch"], curves["train_loss"], label="train")
    axis.plot(curves["epoch"], curves["validation_loss"], label="validation")
    axis.set(xlabel="Epoch", ylabel="Pinball loss", title=f"{pipeline} {model_name} loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / f"curves_{pipeline}_{model_name}.png", dpi=150)
    plt.close(figure)


def _save_holdout_predictions(
    result: TrainingResult,
    data: PreparedData,
    model_name: str,
    pipeline: str,
    results_dir: Path,
) -> Path:
    frame = data.holdout
    if model_name == "mlp":
        loader = create_dataloader(frame, data, 4096, shuffle=False)
        predictions = _predict_mlp(result.model, loader, next(result.model.parameters()).device)
    else:
        predictions = result.model.predict(_lgbm_features(data, frame))
    predictions = np.expm1(predictions).clip(0, None)
    if data.spec.name == "rossmann":
        predictions[frame["Open"].to_numpy() == 0] = 0
    output = frame[[*data.spec.series_id_cols, "Date", data.spec.target_col]].copy()
    output = output.rename(columns={"Date": "date", data.spec.target_col: "actual"})
    output[["p10", "p50", "p90"]] = predictions
    output["model"] = model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"predictions_{pipeline}_{model_name}.parquet"
    output[["date", *data.spec.series_id_cols, "actual", "p10", "p50", "p90", "model"]].to_parquet(
        path, index=False
    )
    return path


def _save_validation_predictions(
    result: TrainingResult,
    data: PreparedData,
    model_name: str,
    pipeline: str,
    results_dir: Path,
) -> Path:
    frame = data.validation
    if model_name == "mlp":
        loader = create_dataloader(frame, data, 4096, shuffle=False)
        predictions = _predict_mlp(result.model, loader, next(result.model.parameters()).device)
    else:
        predictions = result.model.predict(_lgbm_features(data, frame))
    predictions = np.expm1(predictions).clip(0, None)
    if data.spec.name == "rossmann" and "Open" in frame.columns:
        predictions[frame["Open"].to_numpy() == 0] = 0
    output = frame[[*data.spec.series_id_cols, "Date", data.spec.target_col]].copy()
    output = output.rename(columns={"Date": "date", data.spec.target_col: "actual"})
    output[["p10", "p50", "p90"]] = predictions
    output["model"] = model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"validation_predictions_{pipeline}_{model_name}.parquet"
    output[["date", *data.spec.series_id_cols, "actual", "p10", "p50", "p90", "model"]].to_parquet(
        path, index=False
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Phase 3 forecasting model.")
    parser.add_argument("--pipeline", choices=("A", "B"), required=True)
    parser.add_argument("--model", choices=("mlp", "lgbm"), required=True)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive.")

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config if not args.config.is_absolute() else args.config)
    set_random_seed(int(config["seed"]))
    data = prepare_data(args.pipeline, args.horizon, root / "data" / "processed", config)
    model_output = root / "models"
    results_dir = root / "results"
    started = time.perf_counter()
    if args.model == "mlp":
        result = train_mlp(data, config, args.pipeline, args.horizon, model_output, args.smoke)
    else:
        result = train_lgbm(data, config, args.pipeline, args.horizon, model_output, args.smoke)
    _save_curve(result.curves, results_dir, args.pipeline, args.model)
    prediction_path = _save_holdout_predictions(
        result, data, args.model, args.pipeline, results_dir
    )
    val_path = _save_validation_predictions(
        result, data, args.model, args.pipeline, results_dir
    )
    print(
        f"Completed {args.pipeline}/{args.model}: best validation loss "
        f"{result.best_validation_loss:.6f}; predictions={prediction_path}; "
        f"validation_predictions={val_path}; elapsed={time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
