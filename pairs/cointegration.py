"""
Co-integration Testing — the core of pair selection.

═══════════════════════════════════════════════════════════════════════════════

MATHEMATICAL FOUNDATION
═══════════════════════

Two time series X_t and Y_t are co-integrated if:
  1. Each is individually non-stationary (I(1) — integrated of order 1)
  2. A linear combination Z_t = Y_t - β·X_t is stationary (I(0))

Intuition:
  Both series "wander" (random walk) but they are tethered to each other.
  The spread Z_t = Y_t - β·X_t is the "rubber band" — it can stretch,
  but it always snaps back to its mean.

Steps:
  1. Test each series for non-stationarity (ADF test)
  2. Estimate hedge ratio β by OLS: Y = α + β·X + ε
  3. Test residuals ε for stationarity (Engle-Granger test)
  4. Optionally: Johansen test for multi-variate co-integration

═══════════════════════════════════════════════════════════════════════════════

1. AUGMENTED DICKEY-FULLER (ADF) TEST
═══════════════════════════════════════

Tests null hypothesis H₀: unit root exists (series is non-stationary)
vs. H₁: series is stationary.

Model: ΔX_t = α + βX_{t-1} + Σγ_i ΔX_{t-i} + ε_t

If β < 0 and statistically significant → reject H₀ → stationary.

Test statistic: τ = β̂ / SE(β̂)  (follows Dickey-Fuller distribution, not t)

Decision rule:
  p-value < 0.05 → reject H₀ → series IS stationary
  p-value > 0.05 → fail to reject H₀ → series has unit root (non-stationary)

For pair trading:
  Each individual price series should have p > 0.05 (non-stationary)
  The spread should have p < 0.05 (stationary) → co-integrated!

═══════════════════════════════════════════════════════════════════════════════

2. ENGLE-GRANGER CO-INTEGRATION TEST
═══════════════════════════════════════

Two-step procedure:
  Step 1: Regress Y on X: Y_t = α + β·X_t + ε_t  (OLS)
  Step 2: Test residuals ε_t for stationarity using ADF

If residuals are stationary → (Y, X) are co-integrated with hedge ratio β.

Limitation: order matters (Y on X ≠ X on Y), only handles pairwise.

═══════════════════════════════════════════════════════════════════════════════

3. JOHANSEN CO-INTEGRATION TEST
═══════════════════════════════════════

More general: tests for co-integration among N > 2 series simultaneously.
Also determines the co-integration rank r (number of co-integrating vectors).

VAR(p) representation: ΔX_t = ΠX_{t-1} + Σ Γ_i ΔX_{t-i} + ε_t

If rank(Π) = r:
  r = 0 → no co-integration
  0 < r < N → r co-integrating relationships
  r = N → all series are stationary

Two test statistics:

  Trace statistic: λ_trace(r) = -T Σ_{i=r+1}^{N} ln(1 - λ̂_i)
  Tests H₀: rank ≤ r vs. H₁: rank > r

  Maximum eigenvalue: λ_max(r) = -T ln(1 - λ̂_{r+1})
  Tests H₀: rank = r vs. H₁: rank = r+1

Advantages over Engle-Granger:
  - Order-invariant (symmetric)
  - Handles multiple series
  - More powerful in finite samples
  - Provides multiple co-integrating vectors

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class ADFResult:
    """Results from Augmented Dickey-Fuller unit root test."""
    statistic: float
    pvalue: float
    critical_values: Dict[str, float]
    is_stationary: bool    # True if p < 0.05 (reject unit root)
    lags: int

    def __str__(self):
        status = "STATIONARY" if self.is_stationary else "NON-STATIONARY"
        return f"ADF({status}) stat={self.statistic:.4f} p={self.pvalue:.4f}"


@dataclass
class EngleGrangerResult:
    """Results from Engle-Granger co-integration test."""
    symbol_a: str
    symbol_b: str
    # OLS regression: Y = α + β·X + ε
    alpha: float           # intercept
    beta: float            # hedge ratio (co-integrating coefficient)
    # ADF on residuals
    adf_statistic: float
    pvalue: float
    critical_values: Dict[str, float]
    is_cointegrated: bool  # True if p < 0.05
    residuals: Optional[pd.Series] = None

    def __str__(self):
        status = "CO-INT" if self.is_cointegrated else "NOT CO-INT"
        return (f"EngleGranger({status}) {self.symbol_a}/{self.symbol_b} "
                f"β={self.beta:.4f} p={self.pvalue:.4f}")


@dataclass
class JohansenResult:
    """Results from Johansen co-integration test."""
    symbols: List[str]
    # Trace test
    trace_stats: np.ndarray
    trace_pvalues: np.ndarray
    trace_crit_vals: np.ndarray
    # Eigenvalue test
    max_eig_stats: np.ndarray
    max_eig_pvalues: np.ndarray
    max_eig_crit_vals: np.ndarray
    # Co-integrating vectors (columns of eigenvectors)
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    # Rank
    cointegration_rank: int
    is_cointegrated: bool

    def get_hedge_ratios(self) -> Optional[np.ndarray]:
        """Extract hedge ratios from first co-integrating vector."""
        if self.cointegration_rank == 0:
            return None
        # Normalize by first element
        vec = self.eigenvectors[:, 0]
        return vec / vec[0]


class CointegrationTester:
    """
    Implements Engle-Granger and Johansen co-integration tests.

    Usage:
        tester = CointegrationTester(pvalue_threshold=0.05)
        # Test single pair
        eg_result = tester.engle_granger(prices['JPM'], prices['BAC'])
        # Test multiple series
        joh_result = tester.johansen(prices[['JPM', 'BAC', 'WFC']])
    """

    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        adf_lags: Optional[int] = None,   # None → auto select by AIC
        trend: str = "c",                  # 'c' = constant, 'ct' = trend
    ):
        self.pvalue_threshold = pvalue_threshold
        self.adf_lags = adf_lags
        self.trend = trend

    def adf_test(self, series: pd.Series) -> ADFResult:
        """
        Augmented Dickey-Fuller test for unit root.

        H₀: unit root (non-stationary)
        H₁: stationary

        Args:
            series: price or spread time series

        Returns:
            ADFResult with test statistic, p-value, critical values
        """
        clean = series.dropna()
        result = adfuller(
            clean,
            maxlag=self.adf_lags,
            autolag="AIC" if self.adf_lags is None else None,
            regression=self.trend,
        )
        stat, pval, lags, _, crit, _ = result
        return ADFResult(
            statistic=float(stat),
            pvalue=float(pval),
            critical_values={k: float(v) for k, v in crit.items()},
            is_stationary=pval < self.pvalue_threshold,
            lags=int(lags),
        )

    def engle_granger(
        self,
        y: pd.Series,
        x: pd.Series,
        symbol_a: str = "A",
        symbol_b: str = "B",
    ) -> EngleGrangerResult:
        """
        Engle-Granger two-step co-integration test.

        Step 1: Regress Y = α + β·X + ε  (OLS)
        Step 2: ADF test on residuals ε

        If residuals are stationary → co-integrated.

        Mathematical note:
          The OLS estimator β̂ is super-consistent for co-integrated series:
          √T-consistent instead of the usual √n-consistent.
          This means the hedge ratio converges faster to the true value.

        Args:
            y: dependent price series (Y_t)
            x: independent price series (X_t)

        Returns:
            EngleGrangerResult with hedge ratio and co-integration p-value
        """
        # Align series
        y, x = y.align(x, join="inner")
        y, x = y.dropna(), x.dropna()
        y, x = y.align(x, join="inner")

        if len(y) < 60:
            logger.warning(f"Too few observations for {symbol_a}/{symbol_b}: {len(y)}")
            return EngleGrangerResult(
                symbol_a=symbol_a, symbol_b=symbol_b,
                alpha=np.nan, beta=np.nan,
                adf_statistic=np.nan, pvalue=1.0,
                critical_values={}, is_cointegrated=False,
            )

        # Step 1: OLS regression Y = α + β·X + ε
        X_mat = np.column_stack([np.ones(len(x)), x.values])
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, y.values, rcond=None)
        alpha, beta = coeffs[0], coeffs[1]
        residuals = pd.Series(
            y.values - alpha - beta * x.values,
            index=y.index,
            name=f"spread_{symbol_a}_{symbol_b}",
        )

        # Step 2: ADF test on residuals
        # Note: statsmodels coint() also does this but gives slightly
        #       different critical values (uses MacKinnon 2010)
        t_stat, pval, crit = coint(y.values, x.values, trend=self.trend)

        return EngleGrangerResult(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            alpha=float(alpha),
            beta=float(beta),
            adf_statistic=float(t_stat),
            pvalue=float(pval),
            critical_values={k: float(v) for k, v in zip(["1%", "5%", "10%"], crit)},
            is_cointegrated=pval < self.pvalue_threshold,
            residuals=residuals,
        )

    def johansen(
        self,
        prices: pd.DataFrame,
        det_order: int = 0,    # 0=const, 1=linear trend
        k_ar_diff: int = 1,    # VAR lag order
    ) -> JohansenResult:
        """
        Johansen trace and maximum eigenvalue co-integration test.

        Tests H₀: co-integration rank ≤ r for r = 0, 1, ..., N-1

        Mathematical derivation:
          1. Fit VAR(p) model to differences ΔX_t
          2. Extract residual matrices R₀ (from regressing ΔX on lags)
             and R₁ (from regressing X_{t-1} on lags)
          3. Solve generalized eigenvalue problem:
             |λ S₁₁ - S₁₀ S₀₀⁻¹ S₀₁| = 0
             where S_ij = T⁻¹ Σ R_i R_j'
          4. Eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_N measure
             "correlation" between R₀ and R₁
          5. Test statistics derived from eigenvalues

        Args:
            prices:      DataFrame with N price series (columns = symbols)
            det_order:   deterministic trend (-1=none, 0=const, 1=linear)
            k_ar_diff:   number of lagged differences in VAR

        Returns:
            JohansenResult with test statistics and co-integrating vectors
        """
        data = prices.dropna()
        symbols = list(prices.columns)
        n = len(symbols)

        if len(data) < 60:
            logger.warning(f"Insufficient data for Johansen test: {len(data)} rows")
            return JohansenResult(
                symbols=symbols,
                trace_stats=np.array([]),
                trace_pvalues=np.array([]),
                trace_crit_vals=np.array([]),
                max_eig_stats=np.array([]),
                max_eig_pvalues=np.array([]),
                max_eig_crit_vals=np.array([]),
                eigenvectors=np.array([]),
                eigenvalues=np.array([]),
                cointegration_rank=0,
                is_cointegrated=False,
            )

        result = coint_johansen(data.values, det_order, k_ar_diff)

        # Determine rank by trace test at 5% significance
        # Critical values at 5%: result.cvt[:, 1]
        rank = 0
        for i in range(n):
            if result.lr1[i] > result.cvt[i, 1]:  # trace > 5% critical value
                rank += 1
            else:
                break

        # Compute approximate p-values (Johansen doesn't give exact p-values;
        # we compare test statistics to critical values)
        trace_pvals = np.array([
            1.0 - min(1.0, result.lr1[i] / result.cvt[i, 1]) * 0.05
            for i in range(n)
        ])
        max_eig_pvals = np.array([
            1.0 - min(1.0, result.lr2[i] / result.cvm[i, 1]) * 0.05
            for i in range(n)
        ])

        return JohansenResult(
            symbols=symbols,
            trace_stats=result.lr1,
            trace_pvalues=trace_pvals,
            trace_crit_vals=result.cvt,
            max_eig_stats=result.lr2,
            max_eig_pvalues=max_eig_pvals,
            max_eig_crit_vals=result.cvm,
            eigenvectors=result.evec,
            eigenvalues=result.eig,
            cointegration_rank=rank,
            is_cointegrated=rank > 0,
        )

    def test_all_pairs(
        self,
        prices: pd.DataFrame,
        candidate_pairs: List[Tuple[str, str]],
    ) -> List[EngleGrangerResult]:
        """
        Run Engle-Granger test on all candidate pairs.
        Returns co-integrated pairs sorted by p-value ascending.
        """
        results = []
        n = len(candidate_pairs)
        for i, (sym_a, sym_b) in enumerate(candidate_pairs):
            if sym_a not in prices.columns or sym_b not in prices.columns:
                continue
            result = self.engle_granger(
                prices[sym_a], prices[sym_b], sym_a, sym_b
            )
            if result.is_cointegrated:
                results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"Co-integration: tested {i+1}/{n} pairs, {len(results)} co-integrated")

        results.sort(key=lambda r: r.pvalue)
        logger.info(f"Co-integration complete: {len(results)}/{n} pairs co-integrated")
        return results
