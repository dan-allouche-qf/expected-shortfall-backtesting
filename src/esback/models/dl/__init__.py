"""Deep-learning baselines for VaR/ES forecasting.

Two heads share a common LSTM backbone:

- :class:`LSTMQuantile` outputs $(\\widehat{\\mathrm{VaR}},
  \\widehat{\\mathrm{ES}})$ directly and is trained on a pinball + FZ0 loss.
  Five backtests apply (the four that need only a VaR / ES path plus the
  McNeil--Frey bootstrap).
- :class:`LSTMMDNForecast` outputs the Student-$t$ parameters
  $(\\mu, \\sigma, \\nu)$ of the conditional return distribution; the
  analytic CDF that follows supplies the PIT residual, closed-form VaR / ES
  and a parametric simulator, so eleven of the thirteen battery statistics
  apply.

Every model evaluated through this package goes through the same backtest
battery as the classical GARCH variants, so classical and neural approaches
compare under a unified evaluation.
"""

from __future__ import annotations

from esback.models.dl.lstm_mdn import (
    LSTMMDNForecast,
    MDNTrainingHistory,
    es_from_mdn,
    fit_and_forecast_mdn,
    nll_loss_studentt,
    pit_from_mdn,
    predict_mdn,
    train_lstm_mdn,
    var_from_mdn,
)
from esback.models.dl.lstm_quantile import (
    LSTMQuantile,
    TrainingHistory,
    fit_and_forecast,
    predict,
    select_device,
    train_lstm_quantile,
)

__all__ = [
    "LSTMMDNForecast",
    "LSTMQuantile",
    "MDNTrainingHistory",
    "TrainingHistory",
    "es_from_mdn",
    "fit_and_forecast",
    "fit_and_forecast_mdn",
    "nll_loss_studentt",
    "pit_from_mdn",
    "predict",
    "predict_mdn",
    "select_device",
    "train_lstm_mdn",
    "train_lstm_quantile",
    "var_from_mdn",
]
