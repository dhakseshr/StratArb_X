"""
OLS and Rolling OLS hedge ratio estimation.

Static OLS — baseline, simple, but frozen in time.
Rolling OLS — window-based adaptation, computationally cheap.
Kalman filter (kalman_filter.py) — optimal sequential estimation.

Comparison:
  Method        | Adapts? | Causal? | Noise | Complexity
  ──────────────┼─────────┼─────────┼───────┼───────────
  Static OLS    | No      | No      | Low   | O(n)
  Rolling OLS   | Yes     | Yes     | Med   | O(n·W)
  Kalman Filter | Yes     | Yes     | Low   | O(n)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class OLSHedgeRatio:
    """Full-sample OLS hedge ratio. Y = α + β·X + ε"""

    def fit(self, y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
        """
        Returns (alpha, beta, residuals).
        beta is the hedge ratio: short beta shares of X for every 1 share of Y.
        """
        y, x = y.align(x, join="inner")
        y, x = y.dropna(), x.dropna()
        y, x = y.align(x, join="inner")

        X = np.column_stack([np.ones(len(x)), x.values])
        coeffs, _, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
        alpha, beta = float(coeffs[0]), float(coeffs[1])
        residuals = pd.Series(
            y.values - alpha - beta * x.values,
            index=y.index,
            name="spread_ols",
        )
        logger.debug(f"OLS: α={alpha:.4f} β={beta:.4f}")
        return alpha, beta, residuals


class RollingOLSHedgeRatio:
    """
    Rolling window OLS — recomputes hedge ratio over a sliding window.

    More adaptive than static OLS but has two weaknesses:
      1. Abrupt jumps when influential observations enter/leave the window
      2. Computationally wasteful — throws away information outside window
    """

    def __init__(self, window: int = 60):
        self.window = window

    def fit(self, y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Returns (alphas, betas, spreads) as time series.
        First (window-1) values are NaN during warm-up.
        """
        y, x = y.align(x, join="inner")
        y, x = y.dropna(), x.dropna()
        y, x = y.align(x, join="inner")

        alphas = pd.Series(np.nan, index=y.index, name="alpha")
        betas = pd.Series(np.nan, index=y.index, name="beta")

        for i in range(self.window - 1, len(y)):
            y_win = y.iloc[i - self.window + 1: i + 1].values
            x_win = x.iloc[i - self.window + 1: i + 1].values
            X = np.column_stack([np.ones(self.window), x_win])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y_win, rcond=None)
                alphas.iloc[i] = coeffs[0]
                betas.iloc[i] = coeffs[1]
            except Exception:
                pass

        spreads = y - alphas - betas * x
        spreads.name = "spread_rolling_ols"
        return alphas, betas, spreads
