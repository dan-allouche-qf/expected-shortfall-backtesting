"""LSTM-Quantile model: maps lagged returns to a ``(VaR, ES)`` pair.

Architecture: a stacked LSTM over the past ``seq_len`` returns,
followed by two linear heads mapping the last hidden state to ``VaR``
and ``ES``. The ``ES`` head is parameterised as ``VaR + Softplus(.)``
so the architectural constraint ``ES >= VaR`` (loss-positive
convention) holds by construction.

The training loop is plain PyTorch and supports ``cpu`` and Apple
Silicon ``mps`` devices. The model is trained on ``pinball + fz0``
loss; both components are exposed in :mod:`esback.models.dl.losses`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esback.models.dl.losses import fz0_loss, pinball_loss


def select_device(prefer_mps: bool = True) -> torch.device:
    """Pick the best available torch device, defaulting to MPS on Apple Silicon."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_sliding_windows(returns: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window inputs and one-step-ahead loss-positive targets.

    Given a 1D returns array of length ``n``, returns ``X`` of shape
    ``(n - seq_len, seq_len, 1)`` and ``y`` of shape ``(n - seq_len,)``
    with ``y[k] = -returns[seq_len + k]``.
    """
    if len(returns) <= seq_len:
        raise ValueError(f"Need at least {seq_len + 1} observations, got {len(returns)}.")
    windows = np.lib.stride_tricks.sliding_window_view(returns, seq_len)[:-1]
    x = windows[..., None].astype(np.float32)
    y = (-returns[seq_len:]).astype(np.float32)
    return x, y


class LSTMQuantile(nn.Module):
    """Stacked LSTM mapping a return window to a ``(VaR, ES)`` pair."""

    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        seq_len: int = 20,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_var = nn.Linear(hidden_size, 1)
        self.fc_es_residual = nn.Linear(hidden_size, 1)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs: ``(batch, seq_len, input_size)``. Outputs: two ``(batch,)`` tensors."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        var_pred = nn.functional.softplus(self.fc_var(last)).squeeze(-1)
        es_residual = nn.functional.softplus(self.fc_es_residual(last)).squeeze(-1)
        es_pred = var_pred + es_residual
        return var_pred, es_pred


@dataclass
class TrainingHistory:
    """Per-epoch loss trace returned by :func:`train_lstm_quantile`."""

    pinball: list[float]
    fz0: list[float]


def train_lstm_quantile(
    model: LSTMQuantile,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float = 0.05,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    fz0_weight: float = 1.0,
    pinball_weight: float = 1.0,
    device: torch.device | None = None,
    seed: int = 42,
) -> TrainingHistory:
    """Train ``model`` on ``(x_train, y_train)`` with a pinball + FZ0 loss."""
    device = device or select_device()
    torch.manual_seed(seed)
    model.to(device)

    x_t = torch.from_numpy(x_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = TrainingHistory(pinball=[], fz0=[])
    for _ in range(epochs):
        epoch_pinball, epoch_fz0, n_batches = 0.0, 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            var_pred, es_pred = model(xb)
            l_pinball = pinball_loss(var_pred, yb, alpha)
            l_fz0 = fz0_loss(var_pred, es_pred, yb, alpha)
            loss = pinball_weight * l_pinball + fz0_weight * l_fz0
            loss.backward()
            opt.step()
            epoch_pinball += float(l_pinball.detach().cpu())
            epoch_fz0 += float(l_fz0.detach().cpu())
            n_batches += 1
        history.pinball.append(epoch_pinball / max(n_batches, 1))
        history.fz0.append(epoch_fz0 / max(n_batches, 1))
    return history


@torch.no_grad()
def predict(
    model: LSTMQuantile, x: np.ndarray, *, device: torch.device | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return ``(var, es)`` numpy arrays."""
    device = device or select_device()
    model.to(device).eval()
    x_t = torch.from_numpy(x).to(device)
    var_pred, es_pred = model(x_t)
    return var_pred.cpu().numpy(), es_pred.cpu().numpy()


def fit_and_forecast(
    returns_full: np.ndarray,
    n_is: int,
    *,
    alpha: float = 0.05,
    seq_len: int = 20,
    hidden_size: int = 32,
    num_layers: int = 1,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    fz0_weight: float = 1.0,
    pinball_weight: float = 1.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, TrainingHistory]:
    """End-to-end: train LSTM-Quantile on the in-sample window, predict on OOS.

    ``returns_full`` is the concatenation of the in-sample and the
    out-of-sample returns (1D, in chronological order). The first
    ``n_is`` observations are used to build the training sliding
    windows; the remainder is the OOS slice. The OOS predictions are
    one-step-ahead and use only data available up to ``t - 1``.
    """
    returns = np.asarray(returns_full, dtype=np.float32)
    n_total = len(returns)
    n_oos = n_total - n_is
    if n_oos <= 0:
        raise ValueError("n_is must be strictly less than the total length.")
    if n_is < seq_len + 1:
        raise ValueError(
            f"In-sample window must have at least seq_len + 1 = {seq_len + 1} observations."
        )

    device = device or select_device()
    x_train, y_train = make_sliding_windows(returns[:n_is], seq_len)

    torch.manual_seed(seed)
    model = LSTMQuantile(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers)
    history = train_lstm_quantile(
        model,
        x_train,
        y_train,
        alpha=alpha,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        fz0_weight=fz0_weight,
        pinball_weight=pinball_weight,
        device=device,
        seed=seed,
    )

    x_oos = np.zeros((n_oos, seq_len, 1), dtype=np.float32)
    for k in range(n_oos):
        t = n_is + k
        x_oos[k] = returns[t - seq_len : t][:, None]

    var_oos, es_oos = predict(model, x_oos, device=device)
    return var_oos.astype(np.float64), es_oos.astype(np.float64), history
