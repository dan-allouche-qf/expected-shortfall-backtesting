"""Data-generating processes for size and power studies.

Two DGP families implementing the :class:`DGP` protocol:

- :class:`AR1GarchDGP` for AR(1)-GARCH(1,1) with either Student-t or
  Normal innovations. Drives the size study (when the claimed model
  matches) and one of the power scenarios (Normal data fitted by t).
- :class:`GjrGarchDGP` for AR(1)-GJR-GARCH(1,1,1)-t. Drives the
  asymmetric-leverage power scenario where the claimed symmetric
  GARCH-t mis-specifies the conditional variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class DGP(Protocol):
    """Common API for simulation-study data generating processes."""

    def simulate(self, n: int, rng: np.random.Generator) -> pd.Series: ...


def _innovations(dist: str, nu: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if dist == "t":
        scale = float(np.sqrt((nu - 2) / nu))
        out: np.ndarray = rng.standard_t(df=nu, size=n) * scale
        return out
    if dist == "normal":
        out = rng.standard_normal(size=n)
        return out
    raise ValueError(f"Unsupported innovation distribution: {dist!r}")


@dataclass(frozen=True)
class AR1GarchDGP:
    """AR(1)-GARCH(1,1) with ``t`` or ``normal`` innovations."""

    const: float = 0.05
    a0: float = 0.05
    omega: float = 0.05
    alpha1: float = 0.08
    beta1: float = 0.90
    nu: float = 8.0
    dist: str = "t"

    def simulate(self, n: int, rng: np.random.Generator) -> pd.Series:
        eps = _innovations(self.dist, self.nu, n, rng)
        persistence = max(1 - self.alpha1 - self.beta1, 1e-3)
        sigma2 = np.zeros(n)
        sigma2[0] = self.omega / persistence
        r = np.zeros(n)
        r[0] = self.const + np.sqrt(sigma2[0]) * eps[0]
        for t in range(1, n):
            mu_prev = self.const + self.a0 * (r[t - 2] if t >= 2 else 0.0)
            eps_prev = r[t - 1] - mu_prev
            sigma2[t] = self.omega + self.alpha1 * eps_prev**2 + self.beta1 * sigma2[t - 1]
            r[t] = self.const + self.a0 * r[t - 1] + np.sqrt(sigma2[t]) * eps[t]
        return pd.Series(r, index=pd.date_range("2020-01-01", periods=n, freq="B"))


@dataclass(frozen=True)
class GjrGarchDGP:
    """AR(1)-GJR-GARCH(1,1,1) with ``t`` innovations."""

    const: float = 0.05
    a0: float = 0.05
    omega: float = 0.05
    alpha1: float = 0.05
    beta1: float = 0.85
    gamma: float = 0.10
    nu: float = 8.0

    def simulate(self, n: int, rng: np.random.Generator) -> pd.Series:
        eps = _innovations("t", self.nu, n, rng)
        persistence = max(1 - self.alpha1 - self.gamma / 2 - self.beta1, 1e-3)
        sigma2 = np.zeros(n)
        sigma2[0] = self.omega / persistence
        r = np.zeros(n)
        r[0] = self.const + np.sqrt(sigma2[0]) * eps[0]
        for t in range(1, n):
            mu_prev = self.const + self.a0 * (r[t - 2] if t >= 2 else 0.0)
            eps_prev = r[t - 1] - mu_prev
            leverage = self.gamma if eps_prev < 0 else 0.0
            sigma2[t] = (
                self.omega + (self.alpha1 + leverage) * eps_prev**2 + self.beta1 * sigma2[t - 1]
            )
            r[t] = self.const + self.a0 * r[t - 1] + np.sqrt(sigma2[t]) * eps[t]
        return pd.Series(r, index=pd.date_range("2020-01-01", periods=n, freq="B"))
