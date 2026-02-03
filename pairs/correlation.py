"""
Correlation Filtering — fast pre-screen for co-integration candidates.

Three correlation methods:

  1. Pearson Correlation
  ──────────────────────
  ρ = Cov(X,Y) / (σ_X · σ_Y)

  Measures linear relationship between X and Y.
  Sensitive to outliers (uses actual values, not ranks).
  Assumes normal distributions.
  Problem: correlation between two I(1) processes is spurious —
           both trending up will have high correlation even if
           there's no equilibrium relationship.

  2. Spearman Rank Correlation
  ────────────────────────────
  ρ_s = Pearson(rank(X), rank(Y))

  Non-parametric: replaces values with ranks before computing Pearson.
  Robust to outliers and non-normal distributions.
  Captures monotonic (not just linear) relationships.
  Better for fat-tailed financial data.

  3. Rolling Correlation
  ──────────────────────
  ρ_t = Pearson(X_{t-W:t}, Y_{t-W:t})

  Measures how correlation *changes over time*.
  Stable rolling correlation → structural relationship likely.
  Regime-shifting correlation → relationship may be unstable.
  Low rolling correlation variance → good pair candidate.

  Why correlation is NOT sufficient for pair trading:
  Two price series can have high correlation yet no mean reversion.
  Example: two series both trend up +10% per year will have ρ=0.99
  but the spread between them will also trend, not mean-revert.
  Co-integration testing (below) is the proper test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class CorrelationResult:
    symbol_a: str
    symbol_b: str
    pearson: float
    spearman: float
    pearson_pvalue: float
    spearman_pvalue: float
    rolling_mean: float          # mean of rolling correlation
    rolling_std: float           # std of rolling correlation (stability measure)
    rolling_min: float
    passes_filter: bool


class CorrelationFilter:
    """
    Compute multiple correlation metrics and filter pairs by threshold.

    Usage:
        cf = CorrelationFilter(min_pearson=0.70, min_spearman=0.65)
        results = cf.filter(prices, candidate_pairs)
    """

    def __init__(
        self,
        min_pearson: float = 0.70,
        min_spearman: float = 0.65,
        rolling_window: int = 60,
        min_rolling_mean: float = 0.60,
        max_rolling_std: float = 0.20,   # low std → stable relationship
    ):
        self.min_pearson = min_pearson
        self.min_spearman = min_spearman
        self.rolling_window = rolling_window
        self.min_rolling_mean = min_rolling_mean
        self.max_rolling_std = max_rolling_std

    def pearson_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
    ) -> Tuple[float, float]:
        """
        Compute Pearson correlation on returns (not prices).
        Using returns instead of prices avoids spurious correlation
        from common trends.

        Returns: (correlation, p-value)
        """
        # Use log returns — more stationary than prices
        rx = np.log(x / x.shift(1)).dropna()
        ry = np.log(y / y.shift(1)).dropna()
        # Align
        rx, ry = rx.align(ry, join="inner")
        if len(rx) < 30:
            return np.nan, np.nan
        corr, pval = stats.pearsonr(rx, ry)
        return corr, pval

    def spearman_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
    ) -> Tuple[float, float]:
        """
        Spearman rank correlation on returns.

        Advantages over Pearson:
          - No assumption of normality
          - Robust to outliers (e.g., earnings surprises)
          - Captures non-linear monotonic relationships

        Returns: (correlation, p-value)
        """
        rx = np.log(x / x.shift(1)).dropna()
        ry = np.log(y / y.shift(1)).dropna()
        rx, ry = rx.align(ry, join="inner")
        if len(rx) < 30:
            return np.nan, np.nan
        corr, pval = stats.spearmanr(rx, ry)
        return corr, pval

    def rolling_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Rolling Pearson correlation on returns.

        A stable rolling correlation (low standard deviation over time)
        suggests a persistent structural relationship, not a temporary
        co-movement that will break down.

        Args:
            x, y:   price series
            window: lookback window for each correlation estimate
        """
        window = window or self.rolling_window
        rx = np.log(x / x.shift(1))
        ry = np.log(y / y.shift(1))
        rx, ry = rx.align(ry, join="inner")

        # Rolling correlation using pandas built-in
        rolling_corr = rx.rolling(window).corr(ry)
        return rolling_corr

    def compute(
        self,
        prices: pd.DataFrame,
        symbol_a: str,
        symbol_b: str,
    ) -> Optional[CorrelationResult]:
        """Compute all correlation metrics for a single pair."""
        if symbol_a not in prices.columns or symbol_b not in prices.columns:
            return None

        x = prices[symbol_a].dropna()
        y = prices[symbol_b].dropna()
        x, y = x.align(y, join="inner")

        if len(x) < max(self.rolling_window * 2, 60):
            logger.debug(f"Insufficient data for {symbol_a}/{symbol_b}")
            return None

        pearson, pearson_p = self.pearson_correlation(x, y)
        spearman, spearman_p = self.spearman_correlation(x, y)
        rolling = self.rolling_correlation(x, y)
        roll_valid = rolling.dropna()

        if len(roll_valid) == 0:
            return None

        roll_mean = float(roll_valid.mean())
        roll_std = float(roll_valid.std())
        roll_min = float(roll_valid.min())

        passes = (
            not np.isnan(pearson) and
            pearson >= self.min_pearson and
            spearman >= self.min_spearman and
            roll_mean >= self.min_rolling_mean and
            roll_std <= self.max_rolling_std
        )

        return CorrelationResult(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            pearson=pearson,
            spearman=spearman,
            pearson_pvalue=pearson_p,
            spearman_pvalue=spearman_p,
            rolling_mean=roll_mean,
            rolling_std=roll_std,
            rolling_min=roll_min,
            passes_filter=passes,
        )

    def filter(
        self,
        prices: pd.DataFrame,
        candidate_pairs: List[Tuple[str, str]],
    ) -> List[CorrelationResult]:
        """
        Run correlation filter on all candidate pairs.

        Returns list of CorrelationResult for pairs that pass the filter,
        sorted by Pearson correlation descending.
        """
        results = []
        for sym_a, sym_b in candidate_pairs:
            result = self.compute(prices, sym_a, sym_b)
            if result and result.passes_filter:
                results.append(result)

        results.sort(key=lambda r: r.pearson, reverse=True)
        logger.info(
            f"Correlation filter: {len(results)}/{len(candidate_pairs)} pairs passed"
        )
        return results

    def correlation_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute full correlation matrix of returns.
        Useful for heatmap visualization and cluster analysis.
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns.corr(method="pearson")
