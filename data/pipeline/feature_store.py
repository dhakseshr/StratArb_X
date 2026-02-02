"""
Feature Store — computed features for the ML pipeline.

The feature store sits between raw market data and the ML models.
It pre-computes expensive features and caches them for fast access.

Features stored:
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Microstructure features (spread, volume imbalance)
  - Pair features (z-score, spread, half-life)
  - OU process parameters (theta, mu, sigma)
  - Regime indicators (HMM state, volatility regime)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureStore:
    """
    Computes and caches features for the ML signal enhancement layer.
    """

    def compute_technical_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute standard technical indicators.

        Returns DataFrame with features aligned to prices index.
        """
        df = pd.DataFrame(index=prices.index)

        # Returns
        df["ret_1d"] = prices.pct_change(1)
        df["ret_5d"] = prices.pct_change(5)
        df["ret_21d"] = prices.pct_change(21)

        # Volatility
        df["vol_21d"] = df["ret_1d"].rolling(21).std() * np.sqrt(252)
        df["vol_63d"] = df["ret_1d"].rolling(63).std() * np.sqrt(252)

        # RSI (14-period)
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        bb_mid = prices.rolling(20).mean()
        bb_std = prices.rolling(20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_pct"] = (prices - bb_lower) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # ATR (proxy using close-to-close)
        df["atr_14"] = df["ret_1d"].abs().rolling(14).mean() * prices

        # Volume-price trend
        df["price_momentum_5d"] = prices / prices.shift(5) - 1
        df["price_momentum_21d"] = prices / prices.shift(21) - 1

        return df

    def compute_spread_features(
        self,
        spread: pd.Series,
        hedge_ratio: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Features computed from the spread time series.
        Used for predicting spread mean reversion probability.
        """
        df = pd.DataFrame(index=spread.index)

        # Z-score at multiple lookbacks
        for window in [20, 60, 126]:
            roll_mean = spread.rolling(window).mean()
            roll_std = spread.rolling(window).std()
            df[f"zscore_{window}d"] = (spread - roll_mean) / (roll_std + 1e-10)

        # Spread momentum
        df["spread_ret_1d"] = spread.pct_change(1)
        df["spread_ret_5d"] = spread.pct_change(5)

        # Spread volatility
        df["spread_vol_20d"] = spread.pct_change().rolling(20).std()

        # Distance from zero (normalized)
        spread_std = spread.rolling(60).std()
        df["spread_dist_from_zero"] = spread.abs() / (spread_std + 1e-10)

        # Regime: is spread in "active zone" (abs zscore > 1)?
        zscore = df.get("zscore_60d", df["zscore_20d"])
        df["in_active_zone"] = (zscore.abs() > 1.0).astype(float)

        # Autocorrelation (negative AC → mean reverting)
        df["spread_ac_1"] = spread.autocorr(lag=1)  # scalar — will be constant
        # Better: rolling autocorrelation
        def rolling_ac(s, window=20, lag=1):
            return s.rolling(window).apply(
                lambda x: pd.Series(x).autocorr(lag=lag), raw=False
            )
        df["rolling_ac_20d"] = rolling_ac(spread, 20, 1)

        # Half-life proxy (from AR(1) regression)
        if len(spread.dropna()) > 30:
            spread_lag = spread.shift(1)
            valid = spread.dropna() & spread_lag.dropna()
            if len(spread.dropna()) > 30:
                from scipy import stats
                try:
                    slope, _, _, _, _ = stats.linregress(
                        spread_lag.dropna().values,
                        spread.dropna().values,
                    )
                    if 0 < slope < 1:
                        half_life = -np.log(2) / np.log(slope)
                        df["half_life_proxy"] = half_life
                    else:
                        df["half_life_proxy"] = np.nan
                except Exception:
                    df["half_life_proxy"] = np.nan
            else:
                df["half_life_proxy"] = np.nan
        else:
            df["half_life_proxy"] = np.nan

        # Hedge ratio features (if dynamic hedge ratio provided)
        if hedge_ratio is not None:
            df["hedge_ratio_change"] = hedge_ratio.pct_change()
            df["hedge_ratio_vol"] = hedge_ratio.rolling(20).std()

        return df

    def compute_order_flow_features(
        self,
        df: pd.DataFrame,
        volume_col: str = "volume",
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Order flow imbalance features.
        Used to detect aggressive buying/selling pressure.
        """
        features = pd.DataFrame(index=df.index)

        if volume_col in df.columns and price_col in df.columns:
            returns = df[price_col].pct_change()
            volume = df[volume_col]

            # Volume ratio (today vs 20d avg)
            features["volume_ratio"] = volume / volume.rolling(20).mean()

            # Volume-weighted return (proxy for order imbalance)
            features["vwr_5d"] = (returns * volume).rolling(5).sum() / volume.rolling(5).sum()

            # Turnover
            features["turnover_1d"] = volume * df[price_col]
            features["turnover_ratio"] = features["turnover_1d"] / features["turnover_1d"].rolling(20).mean()

            # Up/Down volume ratio
            up_vol = (volume * (returns > 0)).rolling(10).sum()
            down_vol = (volume * (returns < 0)).rolling(10).sum()
            features["ud_ratio"] = up_vol / (down_vol + 1e-10)

        return features

    def build_feature_matrix(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        spread: pd.Series,
        volume_a: Optional[pd.Series] = None,
        volume_b: Optional[pd.Series] = None,
        hedge_ratio: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build complete feature matrix for ML signal model.
        Combines all feature groups into one DataFrame.
        """
        parts = []

        # Asset A features
        feat_a = self.compute_technical_features(prices_a).add_suffix("_a")
        parts.append(feat_a)

        # Asset B features
        feat_b = self.compute_technical_features(prices_b).add_suffix("_b")
        parts.append(feat_b)

        # Spread features
        spread_feat = self.compute_spread_features(spread, hedge_ratio)
        parts.append(spread_feat)

        # Order flow features (if volume available)
        if volume_a is not None:
            of_a = self.compute_order_flow_features(
                pd.DataFrame({"close": prices_a, "volume": volume_a})
            ).add_suffix("_a")
            parts.append(of_a)
        if volume_b is not None:
            of_b = self.compute_order_flow_features(
                pd.DataFrame({"close": prices_b, "volume": volume_b})
            ).add_suffix("_b")
            parts.append(of_b)

        combined = pd.concat(parts, axis=1)
        combined = combined.replace([np.inf, -np.inf], np.nan)
        return combined
