"""Loss functions for probabilistic demand forecasting."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for src.losses. Install requirements.txt first."
    ) from exc


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """Return summed pinball loss: max(tau*(y-q), (tau-1)*(y-q))."""
    if predictions.ndim != 2 or predictions.shape[1] != len(quantiles):
        raise ValueError("predictions must have shape [batch, number of quantiles].")
    if targets.ndim != 1 or targets.shape[0] != predictions.shape[0]:
        raise ValueError("targets must have shape [batch].")
    if any(not 0 < quantile < 1 for quantile in quantiles):
        raise ValueError("Each quantile must be strictly between 0 and 1.")

    errors = targets.unsqueeze(1) - predictions
    tau = torch.tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
    return torch.maximum(tau * errors, (tau - 1) * errors).sum()
