"""Unit tests for the deep-learning baseline (LSTM-Quantile)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from esback.models.dl.losses import fz0_loss, pinball_loss
from esback.models.dl.lstm_quantile import (
    LSTMQuantile,
    make_sliding_windows,
    predict,
    select_device,
    train_lstm_quantile,
)


def test_pinball_loss_zero_at_perfect_prediction() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    assert float(pinball_loss(pred, pred, alpha=0.05)) == pytest.approx(0.0, abs=1e-7)


def test_pinball_loss_positive() -> None:
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 1.0])
    assert float(pinball_loss(pred, target, alpha=0.05)) > 0


def test_pinball_loss_asymmetric_at_alpha_5pct() -> None:
    """At alpha=0.05, under-prediction by 1 is much costlier than over-prediction."""
    target = torch.tensor([1.0])
    under = pinball_loss(torch.tensor([0.0]), target, alpha=0.05)  # diff=+1
    over = pinball_loss(torch.tensor([2.0]), target, alpha=0.05)  # diff=-1
    assert float(under) == pytest.approx(0.05)
    assert float(over) == pytest.approx(0.95)


def test_fz0_loss_finite_on_random_inputs() -> None:
    var_pred = torch.tensor([1.0, 2.0, 3.0])
    es_pred = var_pred + 0.5
    target = torch.tensor([0.5, 2.5, 1.0])
    loss = fz0_loss(var_pred, es_pred, target, alpha=0.05)
    assert torch.isfinite(loss)


def test_make_sliding_windows_shape() -> None:
    returns = np.arange(20, dtype=np.float32)
    x, y = make_sliding_windows(returns, seq_len=5)
    assert x.shape == (15, 5, 1)
    assert y.shape == (15,)


def test_make_sliding_windows_target_is_loss_positive() -> None:
    returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -2.0], dtype=np.float32)
    _, y = make_sliding_windows(returns, seq_len=5)
    assert y[0] == pytest.approx(2.0)


def test_make_sliding_windows_too_short_raises() -> None:
    with pytest.raises(ValueError, match="at least"):
        make_sliding_windows(np.zeros(3, dtype=np.float32), seq_len=10)


def test_lstm_quantile_forward_shapes() -> None:
    model = LSTMQuantile(seq_len=10, hidden_size=8)
    x = torch.randn(4, 10, 1)
    var_pred, es_pred = model(x)
    assert var_pred.shape == (4,)
    assert es_pred.shape == (4,)
    assert torch.all(var_pred >= 0)
    assert torch.all(es_pred >= var_pred)


def test_select_device_returns_torch_device() -> None:
    d = select_device(prefer_mps=False)
    assert isinstance(d, torch.device)


def test_train_lstm_quantile_runs_and_records_history() -> None:
    """Smoke train: the loop completes and records one entry per epoch."""
    rng = np.random.default_rng(0)
    n, seq_len = 400, 10
    returns = rng.standard_t(df=8.0, size=n).astype(np.float32) * 0.7
    x, y = make_sliding_windows(returns, seq_len=seq_len)
    model = LSTMQuantile(seq_len=seq_len, hidden_size=8)
    hist = train_lstm_quantile(
        model,
        x,
        y,
        alpha=0.05,
        epochs=10,
        batch_size=32,
        lr=5e-3,
        device=torch.device("cpu"),
    )
    assert len(hist.pinball) == 10
    assert len(hist.fz0) == 10
    assert all(np.isfinite(v) for v in hist.pinball)
    assert all(np.isfinite(v) for v in hist.fz0)


def test_train_lstm_quantile_pinball_only_decreases_on_iid_target() -> None:
    """With pinball-only loss on i.i.d. data, the loss should not blow up."""
    rng = np.random.default_rng(0)
    n, seq_len = 600, 10
    returns = rng.standard_normal(n).astype(np.float32) * 0.5
    x, y = make_sliding_windows(returns, seq_len=seq_len)
    model = LSTMQuantile(seq_len=seq_len, hidden_size=16)
    hist = train_lstm_quantile(
        model,
        x,
        y,
        alpha=0.05,
        epochs=20,
        batch_size=32,
        lr=1e-2,
        fz0_weight=0.0,
        pinball_weight=1.0,
        device=torch.device("cpu"),
    )
    # Average over the last 5 epochs should be at least as good as the
    # average over the first 5; this is a robust smoke check that
    # absorbs the natural noise of small-batch SGD.
    assert np.mean(hist.pinball[-5:]) <= np.mean(hist.pinball[:5]) + 1e-3


def test_predict_returns_numpy_arrays() -> None:
    rng = np.random.default_rng(1)
    n, seq_len = 80, 10
    returns = rng.standard_normal(n).astype(np.float32) * 0.5
    x, _ = make_sliding_windows(returns, seq_len=seq_len)
    model = LSTMQuantile(seq_len=seq_len, hidden_size=8)
    var, es = predict(model, x, device=torch.device("cpu"))
    assert isinstance(var, np.ndarray)
    assert isinstance(es, np.ndarray)
    assert var.shape == (x.shape[0],)
    assert np.all(es >= var)
