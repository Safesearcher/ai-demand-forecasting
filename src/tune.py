"""Random-search hyperparameter tuning for Phase 3 models."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from .data import create_dataloader, prepare_data
    from .train import (
        _default_config,
        _lgbm_features,
        _run_mlp_epoch,
        _limit_for_smoke,
        load_config,
        set_random_seed,
    )
    from .models import LightGBMQuantileModel, QuantileMLP
    import torch
except ImportError:
    from data import create_dataloader, prepare_data
    from train import (
        _default_config,
        _lgbm_features,
        _run_mlp_epoch,
        _limit_for_smoke,
        load_config,
        set_random_seed,
    )
    from models import LightGBMQuantileModel, QuantileMLP
    import torch


def sample_mlp_params(rng: random.Random) -> dict[str, Any]:
    """Sample one MLP configuration from the fixed project search space."""
    return {
        "learning_rate": rng.choice([1e-3, 3e-4, 1e-4]),
        "batch_size": rng.choice([512, 1024, 2048]),
        "dropout": rng.choice([0.1, 0.2, 0.3]),
        "hidden_sizes": rng.choice([[256, 128, 64], [512, 256, 128]]),
        "weight_decay": rng.choice([0.0, 1e-4, 1e-3]),
    }


def sample_lgbm_params(rng: random.Random) -> dict[str, Any]:
    """Sample one LightGBM configuration from the fixed project search space."""
    return {
        "num_leaves": rng.choice([15, 31, 63, 127]),
        "learning_rate": rng.choice([0.01, 0.03, 0.05, 0.1]),
        "min_child_samples": rng.choice([10, 20, 50, 100]),
        "feature_fraction": rng.choice([0.7, 0.85, 1.0]),
    }


def _subsample(data: Any, smoke: bool) -> Any:
    if not smoke:
        return data
    data.train = _limit_for_smoke(data.train, list(data.spec.series_id_cols), rows=4096)
    data.validation = _limit_for_smoke(
        data.validation, list(data.spec.series_id_cols), rows=2048
    )
    return data


def _mlp_trial(
    data: Any,
    config: dict[str, Any],
    params: dict[str, Any],
    smoke: bool,
) -> tuple[float, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_dataloader(
        data.train,
        data,
        params["batch_size"],
        shuffle=True,
    )
    valid_loader = create_dataloader(
        data.validation,
        data,
        params["batch_size"],
        shuffle=False,
    )
    model = QuantileMLP(
        data.categorical_sizes,
        len(data.numeric_columns),
        embedding_dims=config["mlp"].get("embedding_dims"),
        hidden_sizes=params["hidden_sizes"],
        dropout=params["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    epochs = 2 if smoke else int(config["training"]["epochs"])
    patience = 2 if smoke else int(config["training"]["patience"])
    best_loss = float("inf")
    stale = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        _run_mlp_epoch(model, train_loader, device, optimizer)
        validation_loss = _run_mlp_epoch(model, valid_loader, device, None)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_loss, best_epoch


def _lgbm_trial(
    data: Any,
    params: dict[str, Any],
    smoke: bool,
) -> tuple[float, int]:
    model = LightGBMQuantileModel(
        params=params,
        num_boost_round=20 if smoke else 2000,
        early_stopping_rounds=5 if smoke else 100,
    )
    model.fit(
        _lgbm_features(data, data.train),
        np.log1p(data.train[data.spec.target_col].to_numpy()),
        _lgbm_features(data, data.validation),
        np.log1p(data.validation[data.spec.target_col].to_numpy()),
    )
    predictions = model.predict(_lgbm_features(data, data.validation))
    errors = data.validation[data.spec.target_col].to_numpy()[:, None] - predictions
    taus = np.asarray((0.1, 0.5, 0.9))
    loss = float(np.maximum(taus * errors, (taus - 1) * errors).mean())
    return loss, int(model.models[0].best_iteration_)


def run_trial(
    pipeline: str,
    model: str,
    params: dict[str, Any],
    config: dict[str, Any],
    horizon: int,
    data_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run one trial and return serializable metrics and parameters."""
    data = _subsample(
        prepare_data(pipeline, horizon, data_dir, config),
        smoke,
    )
    started = time.perf_counter()
    if model == "mlp":
        best_loss, epochs = _mlp_trial(data, config, params, smoke)
    elif model == "lgbm":
        best_loss, epochs = _lgbm_trial(data, params, smoke)
    else:
        raise ValueError("model must be 'mlp' or 'lgbm'.")
    return {
        **params,
        "best_validation_loss": best_loss,
        "epochs_run": epochs,
        "train_time_seconds": time.perf_counter() - started,
    }


def _write_trial(path: Path, trial: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(trial)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                key: yaml.safe_dump(value, default_flow_style=True).strip()
                if isinstance(value, (list, dict))
                else value
                for key, value in trial.items()
            }
        )


def tune(
    pipeline: str,
    model: str,
    n_trials: int,
    config_path: Path,
    horizon: int,
    data_dir: Path,
    results_dir: Path,
    smoke: bool = False,
) -> Path:
    """Run append-only random search and return the best-config YAML path."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    config = load_config(config_path)
    rng = random.Random(int(config["seed"]))
    trials_path = results_dir / f"tuning_{pipeline}_{model}.csv"
    trial_results = []
    for _ in range(n_trials):
        params = (
            sample_mlp_params(rng)
            if model == "mlp"
            else sample_lgbm_params(rng)
        )
        trial = run_trial(
            pipeline,
            model,
            params,
            config,
            horizon,
            data_dir,
            smoke,
        )
        _write_trial(trials_path, trial)
        trial_results.append(trial)

    best = min(trial_results, key=lambda row: row["best_validation_loss"])
    best_path = config_path.parent / f"best_{pipeline}_{model}.yaml"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {key: value for key, value in best.items() if key not in {
                "best_validation_loss", "epochs_run", "train_time_seconds"
            }},
            handle,
            sort_keys=False,
        )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Phase 3 model hyperparameters.")
    parser.add_argument("--pipeline", choices=("A", "B"), required=True)
    parser.add_argument("--model", choices=("mlp", "lgbm"), required=True)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive.")

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    set_random_seed(int(config["seed"]))
    path = tune(
        args.pipeline,
        args.model,
        args.trials,
        config_path,
        args.horizon,
        root / "data" / "processed",
        root / "results",
        args.smoke,
    )
    print(f"Best configuration written to {path}")


if __name__ == "__main__":
    main()
