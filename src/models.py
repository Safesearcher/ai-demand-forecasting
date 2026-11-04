"""Forecasting model implementations for the two Phase 3 pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # Fail explicitly when model construction is requested.
    raise ImportError(
        "PyTorch is required for src.models. Install requirements.txt first."
    ) from exc


QUANTILES = (0.1, 0.5, 0.9)


class QuantileMLP(nn.Module):
    """MLP with embeddings and monotonic P10/P50/P90 outputs."""

    def __init__(
        self,
        categorical_sizes: dict[str, int],
        numeric_dim: int,
        embedding_dims: dict[str, int] | None = None,
        hidden_sizes: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if numeric_dim < 0:
            raise ValueError("numeric_dim must be non-negative.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        embedding_dims = embedding_dims or {}
        self.categorical_columns = tuple(categorical_sizes)
        embeddings: dict[str, nn.Embedding] = {}
        embedding_size = 0
        for column, category_count in categorical_sizes.items():
            if category_count < 1:
                raise ValueError(f"{column} must have at least one category.")
            dimension = int(embedding_dims.get(column, min(32, max(4, category_count // 2))))
            if dimension < 1:
                raise ValueError(f"Embedding dimension for {column} must be positive.")
            embeddings[column] = nn.Embedding(category_count, dimension)
            embedding_size += dimension
        self.embeddings = nn.ModuleDict(embeddings)

        layers: list[nn.Module] = []
        input_size = embedding_size + numeric_dim
        for hidden_size in hidden_sizes:
            if hidden_size < 1:
                raise ValueError("All hidden layer sizes must be positive.")
            layers.extend(
                [
                    nn.Linear(input_size, int(hidden_size)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_size = int(hidden_size)
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(input_size, 3)

    def forward(
        self,
        categorical_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return columns ordered as P10, P50, P90 in log-sales space."""
        if categorical_ids.ndim != 2:
            raise ValueError("categorical_ids must have shape [batch, categories].")
        if numeric_features.ndim != 2:
            raise ValueError("numeric_features must have shape [batch, features].")
        if categorical_ids.shape[1] != len(self.categorical_columns):
            raise ValueError("categorical_ids does not match configured embeddings.")

        embedded = [
            self.embeddings[column](categorical_ids[:, index])
            for index, column in enumerate(self.categorical_columns)
        ]
        inputs = [*embedded, numeric_features]
        hidden = self.hidden(torch.cat(inputs, dim=1))
        raw = self.output(hidden)
        p50 = raw[:, 0]
        lower_offset = torch.nn.functional.softplus(raw[:, 1])
        upper_offset = torch.nn.functional.softplus(raw[:, 2])
        return torch.stack((p50 - lower_offset, p50, p50 + upper_offset), dim=1)


class LightGBMQuantileModel:
    """Three LightGBM quantile regressors trained on the same validation block."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        quantiles: Sequence[float] = QUANTILES,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100,
    ) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required for LightGBMQuantileModel. "
                "Install requirements.txt first."
            ) from exc
        self._lgb = lgb
        self.quantiles = tuple(float(value) for value in quantiles)
        if self.quantiles != QUANTILES:
            raise ValueError("LightGBMQuantileModel requires quantiles 0.1, 0.5, 0.9.")
        self.params = dict(params or {})
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.models: list[Any] = []

    def fit(
        self,
        x_train: Any,
        y_train: np.ndarray,
        x_valid: Any,
        y_valid: np.ndarray,
    ) -> "LightGBMQuantileModel":
        """Fit one quantile model per requested quantile."""
        self.models = []
        for quantile in self.quantiles:
            model_params = {
                "objective": "quantile",
                "alpha": quantile,
                "verbosity": -1,
                **self.params,
            }
            model = self._lgb.LGBMRegressor(
                **model_params,
                n_estimators=self.num_boost_round,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="quantile",
                callbacks=[
                    self._lgb.early_stopping(
                        self.early_stopping_rounds,
                        verbose=False,
                    )
                ],
            )
            self.models.append(model)
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Return columns ordered as P10, P50, P90."""
        if len(self.models) != len(self.quantiles):
            raise RuntimeError("The LightGBM model must be fitted before prediction.")
        predictions = np.column_stack(
            [model.predict(features) for model in self.models]
        )
        return np.sort(predictions, axis=1)

    def save(self, output_dir: Path, pipeline: str) -> list[Path]:
        """Save each fitted quantile estimator as a LightGBM text model."""
        if len(self.models) != len(self.quantiles):
            raise RuntimeError("The LightGBM model must be fitted before saving.")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for quantile, model in zip(self.quantiles, self.models):
            path = output_dir / f"{pipeline.lower()}_lgbm_p{int(quantile * 100)}.txt"
            model.booster_.save_model(str(path))
            paths.append(path)
        return paths
