"""Mixture-density-network LSTM head: maps lagged returns to a
parametric Student-$t$ distribution on the next return.

Architecture: a stacked LSTM over the past ``seq_len`` returns,
followed by three linear heads mapping the last hidden state to the
parameters $(\\mu, \\sigma, \\nu)$ of a Student-$t$ on the
\\emph{return} $Y_{t+1}$. ``scale`` is enforced positive via softplus;
``df`` is enforced ``> 2`` via ``softplus(.) + 2`` (so the variance is
finite). The Student-$t$ uses the standard scipy / ``torch.distributions``
parameterisation $X = \\mu + \\sigma\\,T_\\nu$ where $T_\\nu$ is the
unit-scale Student-$t$ with $\\nu$ degrees of freedom.

The training loss is the negative log-likelihood of the predicted
distribution; once trained, the model produces the analytic
out-of-sample CDF, which in turn provides:

* the PIT residual ``u_t = F_t(Y_t)`` for the Du-Escanciano battery,
* closed-form VaR and ES at any level,
* a parametric simulator for the Acerbi-Szekely Monte-Carlo p-values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esback.models.dl.lstm_quantile import select_device


def _make_return_target(returns: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window inputs paired with the **next return** as the MDN target.

    The LSTM-Quantile head trains against the loss-positive target
    ``-returns[seq_len:]``. The MDN trains against the next *return*
    (signed), because the distribution it parameterises is the return
    distribution $F_t$, from which the PIT residual
    $u_t = F_t(Y_t)$ falls out without sign conventions.
    """
    if len(returns) <= seq_len:
        raise ValueError(f"Need at least {seq_len + 1} observations, got {len(returns)}.")
    windows = np.lib.stride_tricks.sliding_window_view(returns, seq_len)[:-1]
    x = windows[..., None].astype(np.float32)
    y = returns[seq_len:].astype(np.float32)
    return x, y


class LSTMMDNForecast(nn.Module):
    """LSTM mapping a return window to Student-$t$ parameters $(\\mu, \\sigma, \\nu)$.

    Output convention follows ``torch.distributions.StudentT`` and
    ``scipy.stats.t``: the predicted return distribution is
    $Y_{t+1} \\sim t_\\nu(\\mu, \\sigma)$ with location $\\mu$, scale
    $\\sigma > 0$ and degrees of freedom $\\nu > 2$.
    """

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
        self.fc_loc = nn.Linear(hidden_size, 1)
        self.fc_scale = nn.Linear(hidden_size, 1)
        self.fc_df = nn.Linear(hidden_size, 1)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inputs: ``(batch, seq_len, input_size)``.

        Returns three ``(batch,)`` tensors ``(loc, scale, df)`` with
        ``scale > 0`` and ``df > 2``.
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        loc = self.fc_loc(last).squeeze(-1)
        scale = nn.functional.softplus(self.fc_scale(last)).squeeze(-1) + 1e-4
        df = nn.functional.softplus(self.fc_df(last)).squeeze(-1) + 2.0 + 1e-4
        return loc, scale, df


def nll_loss_studentt(
    loc: torch.Tensor,
    scale: torch.Tensor,
    df: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of the predicted Student-$t$.

    Uses ``torch.distributions.StudentT`` with the same $(df, loc, scale)$
    parameterisation as ``scipy.stats.t``. Returns the batch mean.
    """
    dist = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
    nll: torch.Tensor = -dist.log_prob(target).mean()
    return nll


@dataclass
class MDNTrainingHistory:
    """Per-epoch NLL trace returned by :func:`train_lstm_mdn`."""

    nll: list[float] = field(default_factory=list)


def train_lstm_mdn(
    model: LSTMMDNForecast,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
    seed: int = 42,
) -> MDNTrainingHistory:
    """Train ``model`` on ``(x_train, y_train)`` with a Student-$t$ NLL loss."""
    device = device or select_device()
    torch.manual_seed(seed)
    model.to(device)

    x_t = torch.from_numpy(x_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = MDNTrainingHistory()
    for _ in range(epochs):
        epoch_nll, n_batches = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loc, scale, df = model(xb)
            loss = nll_loss_studentt(loc, scale, df, yb)
            loss.backward()
            opt.step()
            epoch_nll += float(loss.detach().cpu())
            n_batches += 1
        history.nll.append(epoch_nll / max(n_batches, 1))
    return history


@torch.no_grad()
def predict_mdn(
    model: LSTMMDNForecast,
    x: np.ndarray,
    *,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run inference and return $(\\mu, \\sigma, \\nu)$ as float64 numpy arrays."""
    device = device or select_device()
    model.to(device).eval()
    x_t = torch.from_numpy(x).to(device)
    loc, scale, df = model(x_t)
    return {
        "loc": loc.cpu().numpy().astype(np.float64),
        "scale": scale.cpu().numpy().astype(np.float64),
        "df": df.cpu().numpy().astype(np.float64),
    }


def fit_and_forecast_mdn(
    returns_full: np.ndarray,
    n_is: int,
    *,
    seq_len: int = 20,
    hidden_size: int = 32,
    num_layers: int = 1,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[dict[str, np.ndarray], MDNTrainingHistory]:
    """End-to-end: train MDN-LSTM on the in-sample window, predict on OOS.

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
    x_train, y_train = _make_return_target(returns[:n_is], seq_len)

    torch.manual_seed(seed)
    model = LSTMMDNForecast(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers)
    history = train_lstm_mdn(
        model,
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )

    x_oos = np.zeros((n_oos, seq_len, 1), dtype=np.float32)
    for k in range(n_oos):
        t = n_is + k
        x_oos[k] = returns[t - seq_len : t][:, None]

    params = predict_mdn(model, x_oos, device=device)
    return params, history


def pit_from_mdn(
    loc: np.ndarray,
    scale: np.ndarray,
    df: np.ndarray,
    oos_returns: np.ndarray,
) -> np.ndarray:
    """PIT residual ``u_t = F_t(Y_t)`` from the predicted Student-$t$.

    Under correct specification the returned array is i.i.d.\\
    $\\mathrm{Uniform}(0,1)$, which is the substrate the Du--Escanciano
    battery operates on.
    """
    u_t = stats.t.cdf(oos_returns, df=df, loc=loc, scale=scale)
    return np.asarray(u_t, dtype=np.float64)


def var_from_mdn(
    loc: np.ndarray,
    scale: np.ndarray,
    df: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Value-at-Risk at level ``alpha`` (loss-positive sign convention).

    ``VaR_t = -F_t^{-1}(alpha)`` where $F_t$ is the predicted return CDF.
    """
    quantile = stats.t.ppf(alpha, df=df, loc=loc, scale=scale)
    return -np.asarray(quantile, dtype=np.float64)


def es_from_mdn(
    loc: np.ndarray,
    scale: np.ndarray,
    df: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Expected Shortfall at level ``alpha`` (loss-positive sign convention).

    Closed form for $X = \\mu + \\sigma\\,T_\\nu$:
    $$
        \\mathrm{ES}_\\alpha = -\\mu + \\sigma \\cdot
        \\frac{\\nu + q_\\alpha^2}{(\\nu - 1)\\,\\alpha}\\,f_\\nu(q_\\alpha)
    $$
    where $q_\\alpha = F_{T_\\nu}^{-1}(\\alpha)$ is the standardised
    quantile and $f_\\nu$ is the standard Student-$t$ PDF. The formula
    is derived from $\\mathbb{E}[T_\\nu \\mid T_\\nu < q_\\alpha] =
    -(\\nu + q_\\alpha^2) f_\\nu(q_\\alpha) / [(\\nu - 1)\\,\\alpha]$.
    """
    q_std = stats.t.ppf(alpha, df=df)
    pdf_q = stats.t.pdf(q_std, df=df)
    es_residual = scale * (df + q_std**2) / ((df - 1.0) * alpha) * pdf_q
    es_value = -loc + es_residual
    return np.asarray(es_value, dtype=np.float64)
