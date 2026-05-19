"""Loss functions for VaR/ES forecasting.

- :func:`pinball_loss` is the strictly consistent scoring function for
  the alpha-quantile (Koenker & Bassett 1978). Drives VaR estimation.
- :func:`fz0_loss` is the FZ0 loss of Fissler & Ziegel (2016), strictly
  consistent for the joint pair (VaR, ES). Drives the joint VaR/ES
  estimation in the LSTM-Quantile baseline.
"""

from __future__ import annotations

import torch


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Quantile (pinball) loss at level ``alpha``.

    For the loss-positive convention used in this project, the
    prediction is the alpha-quantile of the LOSS distribution and the
    target is ``-return``. The pinball loss penalises under-prediction
    of the quantile by ``(1 - alpha)`` and over-prediction by
    ``alpha``.
    """
    diff = target - prediction
    return torch.where(diff >= 0, alpha * diff, (alpha - 1) * diff).mean()


def fz0_loss(
    var_pred: torch.Tensor,
    es_pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """FZ0 loss of Fissler-Ziegel (2016), strictly consistent for ``(VaR, ES)``.

    Loss-positive convention: ``target = -return``, ``var_pred`` and
    ``es_pred`` are positive numbers indicating losses. ``ES >= VaR``
    is enforced by the architecture, but ``es_pred`` is clamped at
    ``eps`` here to keep the logarithm finite at training start.
    """
    es_safe = torch.clamp(es_pred, min=eps)
    indicator = (target > var_pred).float()
    term1 = (indicator / alpha) * (target - var_pred)
    term2 = var_pred + es_safe * torch.log(es_safe) - es_safe
    return (term1 / es_safe + term2 / es_safe).mean()
