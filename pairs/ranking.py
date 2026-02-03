"""
Pair Ranking Score — composite quality metric for co-integrated pairs.

Not all co-integrated pairs are equally tradeable.
We need a ranking system that favors pairs with:
  - Strong co-integration (low p-value, high test statistic)
  - Fast mean reversion (short half-life → more trades per year)
  - Low spread volatility (lower risk per trade)
  - Good historical Sharpe ratio (evidence of profitability)
  - High liquidity in both legs

═══════════════════════════════════════════════════════════════════════════════
COMPOSITE RANKING EQUATION
═══════════════════════════════════════════════════════════════════════════════

Score(pair) = w₁·S_coint + w₂·S_halflife + w₃·S_sharpe
             + w₄·S_liquidity + w₅·S_stability

Where each component is normalized to [0, 1]:

  S_coint    = 1 - pvalue           (lower p → stronger co-integration)
  S_halflife = 1 - clip(HL/HL_max)  (shorter half-life → higher score)
  S_sharpe   = clip(Sharpe / 3.0)   (Sharpe of 3.0 → max score)
  S_liquidity= log(avg_volume) / log(vol_max)
  S_stability= 1 - rolling_corr_std  (stable correlation → higher score)

Default weights: w = [0.30, 0.25, 0.25, 0.10, 0.10]

This rewards pairs that:
  1. Have strong statistical co-integration evidence
  2. Mean-revert quickly (better risk-adjusted return per year)
  3. Have historically generated good risk-adjusted returns
  4. Are liquid enough to trade without excessive impact
  5. Have stable (not regime-shifting) relationships
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from pairs.cointegration import EngleGrangerResult
from pairs.correlation import CorrelationResult


@dataclass
class PairScore:
    """Complete quality assessment for a co-integrated pair."""
    symbol_a: str
    symbol_b: str
    # Co-integration
    coint_pvalue: float
    hedge_ratio: float
    # Mean reversion
    half_life_days: float
    ou_theta: float          # OU speed of mean reversion
    ou_mu: float             # OU long-term mean
    ou_sigma: float          # OU volatility
    # Spread statistics
    spread_mean: float
    spread_std: float
    spread_sharpe: float     # Sharpe ratio of spread returns
    # Component scores (0-1)
    score_coint: float
    score_halflife: float
    score_sharpe: float
    score_liquidity: float
    score_stability: float
    # Final composite
    composite_score: float

    def __repr__(self):
        return (f"PairScore({self.symbol_a}/{self.symbol_b} "
                f"score={self.composite_score:.3f} HL={self.half_life_days:.1f}d "
                f"p={self.coint_pvalue:.4f})")


class PairRanker:
    """
    Compute composite quality scores for co-integrated pairs.

    Usage:
        ranker = PairRanker()
        scores = ranker.rank_pairs(prices, eg_results, corr_results)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_halflife: float = 2.0,
        max_halflife: float = 60.0,
        max_sharpe_cap: float = 3.0,
    ):
        self.weights = weights or {
            "coint": 0.30,
            "halflife": 0.25,
            "sharpe": 0.25,
            "liquidity": 0.10,
            "stability": 0.10,
        }
        self.min_halflife = min_halflife
        self.max_halflife = max_halflife
        self.max_sharpe_cap = max_sharpe_cap

    def compute_half_life(self, spread: pd.Series) -> float:
        """
        Half-life of mean reversion from AR(1) regression.

        Model: Δspread_t = κ · (μ - spread_{t-1}) + ε_t

        Discretized AR(1): spread_t = α + β · spread_{t-1} + ε_t
        β̂ = regression coefficient
        κ̂ = -(1 - β̂) / Δt

        Half-life = log(2) / κ̂ = -log(2) / log(β̂)

        Intuition:
          β < 1 → mean-reverting (|spread| decays toward μ over time)
          β = 1 → random walk (no mean reversion)
          β > 1 → explosive (diverging)

        Half-life = time for spread to close half the distance to mean.
        Short half-life (2-10 days) → fast mean reversion → more signals.
        Long half-life (>60 days) → too slow → capital tied up too long.
        """
        clean = spread.dropna()
        if len(clean) < 30:
            return np.inf

        y = clean.diff().dropna()     # Δspread_t
        x = clean.shift(1).dropna()   # spread_{t-1}
        x, y = x.align(y, join="inner")

        # OLS: Δspread = α + β · spread_{t-1} + ε
        X = np.column_stack([np.ones(len(x)), x.values])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
        except Exception:
            return np.inf

        beta = coeffs[1]  # coefficient on lagged spread
        if beta >= 0:
            return np.inf  # not mean-reverting
        half_life = -np.log(2) / np.log(1 + beta)
        return float(half_life)

    def fit_ou_process(
        self,
        spread: pd.Series,
    ) -> Tuple[float, float, float]:
        """
        Fit Ornstein-Uhlenbeck process to spread.

        OU SDE: dS_t = θ(μ - S_t)dt + σ dW_t

        Parameters:
          θ (theta) = speed of mean reversion  [1/time]
          μ (mu)    = long-term mean (equilibrium level)
          σ (sigma) = volatility / diffusion coefficient

        Discrete approximation (Euler):
          ΔS_t = θ·μ·Δt - θ·S_{t-1}·Δt + σ·√Δt·ε_t
          ΔS_t = a + b·S_{t-1} + ε_t

        OLS regression: ΔS = a + b·S_{t-1} + ε
          b = -θ·Δt  → θ = -b/Δt
          a = θ·μ·Δt → μ = a / (θ·Δt) = -a/b
          σ = std(ε) / √Δt

        Returns: (theta, mu, sigma)
        """
        clean = spread.dropna()
        if len(clean) < 30:
            return np.nan, float(clean.mean()), float(clean.std())

        dt = 1.0  # 1 trading day
        y = clean.diff().dropna()
        x = clean.shift(1).dropna()
        x, y = x.align(y, join="inner")

        X = np.column_stack([np.ones(len(x)), x.values])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
        except Exception:
            return np.nan, float(clean.mean()), float(clean.std())

        a, b = coeffs[0], coeffs[1]
        theta = -b / dt
        if theta <= 0:
            theta = 1e-6
        mu = a / (theta * dt)
        residual_std = float(np.std(y.values - a - b * x.values))
        sigma = residual_std / np.sqrt(dt)

        return float(theta), float(mu), float(sigma)

    def compute_spread_sharpe(
        self,
        spread: pd.Series,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        annualize: float = 252.0,
    ) -> float:
        """
        Compute Sharpe ratio of a simple z-score mean-reversion strategy
        on the spread. Used as a quality metric.
        """
        clean = spread.dropna()
        if len(clean) < 60:
            return 0.0

        roll_mean = clean.rolling(60).mean()
        roll_std = clean.rolling(60).std()
        zscore = (clean - roll_mean) / (roll_std + 1e-10)

        # Simulate simple long/short signal
        position = pd.Series(0.0, index=clean.index)
        in_trade = False
        direction = 0

        for i in range(len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue
            if not in_trade:
                if z > entry_threshold:
                    direction = -1; in_trade = True
                elif z < -entry_threshold:
                    direction = 1; in_trade = True
            else:
                if abs(z) < exit_threshold:
                    in_trade = False; direction = 0
                elif z > abs(entry_threshold) * 2:
                    in_trade = False; direction = 0
            position.iloc[i] = direction

        spread_returns = clean.diff() * position.shift(1)
        spread_returns = spread_returns.dropna()

        if spread_returns.std() < 1e-10:
            return 0.0
        sharpe = (spread_returns.mean() / spread_returns.std()) * np.sqrt(annualize)
        return float(sharpe)

    def score_pair(
        self,
        eg_result: EngleGrangerResult,
        prices: pd.DataFrame,
        corr_result: Optional[CorrelationResult] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> Optional[PairScore]:
        """
        Compute composite score for a single pair.

        Returns None if pair fails basic quality checks.
        """
        sym_a, sym_b = eg_result.symbol_a, eg_result.symbol_b
        if eg_result.residuals is None:
            return None

        spread = eg_result.residuals.dropna()
        if len(spread) < 60:
            return None

        # Compute metrics
        half_life = self.compute_half_life(spread)
        theta, mu, sigma = self.fit_ou_process(spread)
        sharpe = self.compute_spread_sharpe(spread)

        # Filter out non-tradeable pairs
        if half_life < self.min_halflife or half_life > self.max_halflife:
            logger.debug(f"{sym_a}/{sym_b}: half-life {half_life:.1f}d out of range")
            return None

        # ─── Normalize component scores to [0, 1] ────────────────────────

        # Co-integration strength: lower p → higher score
        s_coint = 1.0 - eg_result.pvalue

        # Half-life score: shorter (within range) → higher score
        # Map [min_hl, max_hl] → [1, 0]
        s_halflife = 1.0 - (half_life - self.min_halflife) / (self.max_halflife - self.min_halflife)
        s_halflife = float(np.clip(s_halflife, 0, 1))

        # Sharpe score: cap at max_sharpe_cap
        s_sharpe = float(np.clip(sharpe / self.max_sharpe_cap, 0, 1))

        # Liquidity score: log of average daily dollar volume
        s_liquidity = 0.5  # default if no volume data
        if volumes is not None and sym_a in volumes and sym_b in volumes:
            vol_a = float(volumes[sym_a].mean()) if sym_a in volumes.columns else 1e6
            vol_b = float(volumes[sym_b].mean()) if sym_b in volumes.columns else 1e6
            # Use minimum (bottleneck leg)
            min_vol = min(vol_a, vol_b)
            # Score: log scale, normalize to [0,1] assuming [1e4, 1e8] range
            s_liquidity = float(np.clip(
                (np.log10(max(min_vol, 1)) - 4) / 4.0, 0, 1
            ))

        # Stability score: stable rolling correlation → higher score
        s_stability = 0.5  # default
        if corr_result:
            s_stability = float(np.clip(1.0 - corr_result.rolling_std / 0.3, 0, 1))

        # ─── Composite score ─────────────────────────────────────────────
        w = self.weights
        composite = (
            w["coint"] * s_coint +
            w["halflife"] * s_halflife +
            w["sharpe"] * s_sharpe +
            w["liquidity"] * s_liquidity +
            w["stability"] * s_stability
        )

        return PairScore(
            symbol_a=sym_a,
            symbol_b=sym_b,
            coint_pvalue=eg_result.pvalue,
            hedge_ratio=eg_result.beta,
            half_life_days=half_life,
            ou_theta=theta if not np.isnan(theta) else 0.0,
            ou_mu=mu,
            ou_sigma=sigma,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            spread_sharpe=sharpe,
            score_coint=s_coint,
            score_halflife=s_halflife,
            score_sharpe=s_sharpe,
            score_liquidity=s_liquidity,
            score_stability=s_stability,
            composite_score=composite,
        )

    def rank_pairs(
        self,
        prices: pd.DataFrame,
        eg_results: List[EngleGrangerResult],
        corr_results: Optional[List[CorrelationResult]] = None,
        volumes: Optional[pd.DataFrame] = None,
        top_n: int = 50,
    ) -> List[PairScore]:
        """
        Score and rank all co-integrated pairs.

        Returns top_n pairs by composite score, descending.
        """
        corr_map = {}
        if corr_results:
            for cr in corr_results:
                key = (cr.symbol_a, cr.symbol_b)
                corr_map[key] = cr
                corr_map[(cr.symbol_b, cr.symbol_a)] = cr

        scored = []
        for eg in eg_results:
            corr = corr_map.get((eg.symbol_a, eg.symbol_b))
            score = self.score_pair(eg, prices, corr, volumes)
            if score:
                scored.append(score)

        scored.sort(key=lambda s: s.composite_score, reverse=True)
        top = scored[:top_n]

        logger.info(
            f"Pair ranking: {len(scored)} scored, returning top {len(top)}"
        )
        if top:
            logger.info(f"Top pair: {top[0]}")
        return top
