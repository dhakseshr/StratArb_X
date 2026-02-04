"""
Kalman Filter for Dynamic Hedge Ratio Estimation.

═══════════════════════════════════════════════════════════════════════════════
WHY OLS IS INSUFFICIENT
═══════════════════════════════════════════════════════════════════════════════

Ordinary Least Squares (OLS) gives a static hedge ratio:
  β_OLS = Cov(Y, X) / Var(X)

Problems:
  1. STATIC: Uses the entire sample, assumes β doesn't change over time.
     In reality, hedge ratios shift with earnings cycles, market regimes,
     capital structure changes, etc.

  2. REGIME SHIFTS: If a company's business mix changes (e.g., a bank
     entering credit cards), its sensitivity to broad banking factors changes.
     Static OLS misses this → wrong hedge ratio → non-stationary spread.

  3. STRUCTURAL BREAKS: M&A activity, spinoffs, regulatory changes can
     permanently shift the relationship between two stocks.

  4. LOOK-AHEAD BIAS: Full-sample OLS uses future data to estimate past β.
     This inflates backtest performance. Kalman filter is causal (online).

═══════════════════════════════════════════════════════════════════════════════
STATE-SPACE REPRESENTATION
═══════════════════════════════════════════════════════════════════════════════

We model the hedge ratio β_t as a latent state that evolves over time.

Observation Equation (what we see):
  Y_t = H_t · θ_t + ε_t,    ε_t ~ N(0, R)

  Where:
    Y_t = price of asset Y at time t               [scalar]
    H_t = [1, X_t]  (observation matrix)           [1 × 2]
    θ_t = [α_t, β_t]'  (state: intercept + hedge)  [2 × 1]
    R = observation noise variance (measurement error)

State Equation (how the state evolves):
  θ_t = F · θ_{t-1} + η_t,  η_t ~ N(0, Q)

  Where:
    F = I (identity) — random walk prior for hedge ratio
    Q = state transition covariance (how fast θ can change)

  The random walk prior says: β_t ≈ β_{t-1} + small_shock
  Q controls how quickly the filter adapts to new observations.
  Large Q → fast adaptation, but noisy estimates.
  Small Q → slow adaptation, but smoother estimates.

═══════════════════════════════════════════════════════════════════════════════
KALMAN FILTER ALGORITHM
═══════════════════════════════════════════════════════════════════════════════

Initialize:
  θ̂₀|₀ = initial state estimate  (e.g., OLS estimate over first 60 days)
  P₀|₀  = initial state covariance (large uncertainty)

For each t = 1, 2, ..., T:

  PREDICTION STEP (prior):
  ─────────────────────────
  θ̂_{t|t-1} = F · θ̂_{t-1|t-1}          (predicted state)
  P_{t|t-1}  = F · P_{t-1|t-1} · F' + Q  (predicted covariance)

  INNOVATION (prediction error):
  ─────────────────────────────────
  ŷ_t = Y_t - H_t · θ̂_{t|t-1}           (innovation = actual - predicted)
  S_t = H_t · P_{t|t-1} · H_t' + R      (innovation covariance)

  UPDATE STEP (posterior):
  ─────────────────────────
  K_t = P_{t|t-1} · H_t' · S_t⁻¹        (Kalman gain)
  θ̂_{t|t} = θ̂_{t|t-1} + K_t · ŷ_t     (updated state estimate)
  P_{t|t}  = (I - K_t · H_t) · P_{t|t-1}  (updated covariance)

Kalman Gain Intuition:
  K_t balances "trust in the model" vs. "trust in the observation".
  K → 0: trust the model prediction, ignore new observation (small R)
  K → 1: trust the new observation, ignore model prediction (small P)

═══════════════════════════════════════════════════════════════════════════════
DYNAMIC SPREAD CONSTRUCTION
═══════════════════════════════════════════════════════════════════════════════

  Spread(t) = Y_t - β̂_t · X_t - α̂_t

Where β̂_t is the Kalman-filtered hedge ratio at time t.
This spread is (approximately) stationary even as β changes over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class KalmanState:
    """Current state of the Kalman filter."""
    theta: np.ndarray      # state estimate [alpha, beta]
    P: np.ndarray          # state covariance matrix (2×2)
    innovation: float      # latest prediction error
    innovation_cov: float  # innovation covariance
    kalman_gain: np.ndarray


class KalmanHedgeRatio:
    """
    Online Kalman filter for time-varying hedge ratio estimation.

    Models:
      Observation:  Y_t = α_t + β_t · X_t + ε_t
      State:        [α_t, β_t]' = [α_{t-1}, β_{t-1}]' + η_t

    Usage:
        kf = KalmanHedgeRatio(delta=1e-4)
        alphas, betas, spreads = kf.fit(prices['Y'], prices['X'])
    """

    def __init__(
        self,
        delta: float = 1e-4,     # state noise magnitude (Q = delta * I)
        obs_noise: float = 1e-3,  # observation noise R
        init_var: float = 1.0,    # initial state uncertainty P₀
    ):
        """
        Args:
            delta:     controls how fast the hedge ratio can change.
                       Larger δ → more responsive but noisier.
                       Typical range: 1e-5 to 1e-3
            obs_noise: measurement noise variance R
            init_var:  initial diagonal value of P₀ (large = uncertain start)
        """
        self.delta = delta
        self.R = obs_noise
        self.init_var = init_var

        # State transition matrix (identity = random walk)
        self.F = np.eye(2)
        # State noise covariance: Q = delta/(1-delta) * I  (standard tuning)
        self.Q = (delta / (1 - delta)) * np.eye(2)

    def _initialize(self, y_init: float, x_init: float) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize state θ₀ and covariance P₀."""
        theta_0 = np.array([0.0, y_init / (x_init + 1e-10)])  # [alpha=0, beta=Y/X]
        P_0 = self.init_var * np.eye(2)
        return theta_0, P_0

    def step(
        self,
        y: float,
        x: float,
        theta: np.ndarray,
        P: np.ndarray,
    ) -> KalmanState:
        """
        Single Kalman filter update step.

        Args:
            y:     observed price Y_t
            x:     observed price X_t
            theta: current state estimate θ_{t-1|t-1}
            P:     current covariance P_{t-1|t-1}

        Returns:
            KalmanState with updated theta, P, innovation, gain
        """
        H = np.array([[1.0, x]])   # observation matrix [1, X_t]

        # ── Prediction Step ──────────────────────────────────────────────
        theta_pred = self.F @ theta          # θ̂_{t|t-1} = F · θ̂_{t-1|t-1}
        P_pred = self.F @ P @ self.F.T + self.Q  # P_{t|t-1} = F·P·F' + Q

        # ── Innovation ───────────────────────────────────────────────────
        y_pred = (H @ theta_pred)[0]         # ŷ_t = H_t · θ̂_{t|t-1}
        innovation = y - y_pred              # ẽ_t = Y_t - ŷ_t
        S = (H @ P_pred @ H.T)[0, 0] + self.R  # S_t = H·P·H' + R

        # ── Update Step ──────────────────────────────────────────────────
        K = (P_pred @ H.T / S).flatten()    # K_t = P_{t|t-1}·H'·S⁻¹   [2×1]
        theta_new = theta_pred + K * innovation    # θ̂_{t|t} = θ̂_{t|t-1} + K·ẽ
        P_new = (np.eye(2) - np.outer(K, H)) @ P_pred  # P_{t|t} = (I-KH)·P

        return KalmanState(
            theta=theta_new,
            P=P_new,
            innovation=innovation,
            innovation_cov=S,
            kalman_gain=K,
        )

    def fit(
        self,
        y: pd.Series,
        x: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Run Kalman filter over entire price history.

        Args:
            y: price series of asset Y
            x: price series of asset X

        Returns:
            (alphas, betas, spreads)
              alphas: time-varying intercept α̂_t
              betas:  time-varying hedge ratio β̂_t
              spreads: Y_t - β̂_t·X_t - α̂_t
        """
        # Align and drop NaN
        y, x = y.align(x, join="inner")
        y = y.dropna(); x = x.dropna()
        y, x = y.align(x, join="inner")

        n = len(y)
        alphas = np.zeros(n)
        betas = np.zeros(n)
        spreads = np.zeros(n)
        innovations = np.zeros(n)

        # Initialize
        theta, P = self._initialize(float(y.iloc[0]), float(x.iloc[0]))

        for i, (yi, xi) in enumerate(zip(y.values, x.values)):
            state = self.step(float(yi), float(xi), theta, P)
            theta = state.theta
            P = state.P
            alphas[i] = theta[0]
            betas[i] = theta[1]
            spreads[i] = yi - theta[0] - theta[1] * xi
            innovations[i] = state.innovation

        alpha_series = pd.Series(alphas, index=y.index, name="alpha")
        beta_series = pd.Series(betas, index=y.index, name="beta")
        spread_series = pd.Series(spreads, index=y.index, name="spread_kalman")

        logger.info(
            f"Kalman filter: β range [{beta_series.min():.3f}, {beta_series.max():.3f}], "
            f"β mean={beta_series.mean():.3f}"
        )
        return alpha_series, beta_series, spread_series

    def get_current_estimate(
        self,
        y_history: pd.Series,
        x_history: pd.Series,
    ) -> KalmanState:
        """
        Run filter and return only the final (current) state.
        Useful for live trading — get the latest hedge ratio estimate.
        """
        alpha, beta, spread = self.fit(y_history, x_history)
        # Re-run one step to get full state
        theta = np.array([float(alpha.iloc[-1]), float(beta.iloc[-1])])
        P = (self.delta / (1 - self.delta)) * np.eye(2)  # approximate
        state = self.step(
            float(y_history.iloc[-1]),
            float(x_history.iloc[-1]),
            theta, P,
        )
        return state
