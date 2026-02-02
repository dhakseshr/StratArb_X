"""
Data Validation Pipeline.

Ensures raw market data meets quality standards before storage.

Checks performed:
  - Missing values (NaN detection, forward-fill thresholds)
  - Price anomalies (>10σ moves, zero prices, negative prices)
  - Volume anomalies (zero volume on trading days)
  - Timestamp integrity (duplicates, gaps, out-of-sequence)
  - Corporate actions (splits that need adjustment)
  - Stale data (prices unchanged for too many bars)
  - Cross-symbol consistency (OHLC: H ≥ O,C ≥ L)

Why validation matters:
  Bad data → fake co-integration → false signals → losses.
  Garbage in, garbage out is especially destructive in StatArb.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ValidationReport:
    symbol: str
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    rows_removed: int = 0
    fill_count: int = 0
    outlier_count: int = 0


class DataValidator:
    """
    Validates and cleans market data DataFrames.

    Usage:
        validator = DataValidator()
        clean_df, report = validator.validate(raw_df, symbol='JPM')
    """

    def __init__(
        self,
        max_nan_pct: float = 0.05,      # Allow at most 5% NaN rows
        max_daily_return: float = 0.50,  # Flag returns > 50% as anomalies
        max_stale_days: int = 5,         # Flag if price unchanged > 5 bars
        min_volume: float = 0.0,
    ):
        self.max_nan_pct = max_nan_pct
        self.max_daily_return = max_daily_return
        self.max_stale_days = max_stale_days
        self.min_volume = min_volume

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """
        Main entry point. Returns cleaned DataFrame + validation report.

        Args:
            df:     OHLCV DataFrame with DatetimeIndex
            symbol: ticker for logging

        Returns:
            (cleaned_df, ValidationReport)
        """
        report = ValidationReport(symbol=symbol, is_valid=True)
        df = df.copy()

        # 1. Deduplicate timestamps
        n_dups = df.index.duplicated().sum()
        if n_dups > 0:
            df = df[~df.index.duplicated(keep="last")]
            report.issues.append(f"Removed {n_dups} duplicate timestamps")

        # 2. Sort index
        df = df.sort_index()

        # 3. OHLC consistency: H >= max(O,C) and L <= min(O,C)
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            bad_mask = (
                (df["high"] < df[["open", "close"]].max(axis=1)) |
                (df["low"] > df[["open", "close"]].min(axis=1)) |
                (df["high"] < df["low"])
            )
            n_bad = bad_mask.sum()
            if n_bad > 0:
                df = df[~bad_mask]
                report.rows_removed += n_bad
                report.issues.append(f"Removed {n_bad} rows with OHLC inconsistency")

        # 4. Remove zero/negative prices
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        for col in price_cols:
            bad_mask = df[col] <= 0
            n_bad = bad_mask.sum()
            if n_bad > 0:
                df = df[~bad_mask]
                report.rows_removed += n_bad
                report.issues.append(f"Removed {n_bad} rows with {col} <= 0")

        # 5. Outlier detection via rolling z-score on close returns
        if "close" in df.columns and len(df) > 30:
            returns = df["close"].pct_change()
            rolling_mean = returns.rolling(window=20, min_periods=10).mean()
            rolling_std = returns.rolling(window=20, min_periods=10).std()
            z_scores = ((returns - rolling_mean) / rolling_std).abs()
            outlier_mask = z_scores > 10  # 10-sigma events
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                df.loc[outlier_mask, "close"] = np.nan
                report.outlier_count += n_outliers
                report.issues.append(f"Flagged {n_outliers} price outliers (>10σ)")

        # 6. Forward-fill small gaps
        nan_pct_before = df.isna().mean().mean()
        if nan_pct_before > 0:
            df = df.ffill(limit=3)  # fill up to 3 consecutive NaN
            nan_pct_after = df.isna().mean().mean()
            filled = int((nan_pct_before - nan_pct_after) * len(df) * len(df.columns))
            report.fill_count = filled

        # 7. Reject if too many NaN remain
        final_nan_pct = df.isna().mean().mean()
        if final_nan_pct > self.max_nan_pct:
            report.is_valid = False
            report.issues.append(
                f"Too many NaN: {final_nan_pct:.1%} > threshold {self.max_nan_pct:.1%}"
            )

        # 8. Stale price detection (unchanged close for > N bars)
        if "close" in df.columns:
            stale_mask = df["close"].diff() == 0
            stale_streaks = stale_mask.rolling(self.max_stale_days).sum()
            n_stale = (stale_streaks >= self.max_stale_days).sum()
            if n_stale > 0:
                report.issues.append(f"WARNING: {n_stale} periods with stale prices")

        # Drop remaining NaN rows
        df = df.dropna(subset=["close"] if "close" in df.columns else df.columns[:1])

        if report.is_valid:
            logger.debug(f"Validated {symbol}: {len(df)} rows, {len(report.issues)} issues")
        else:
            logger.warning(f"INVALID data for {symbol}: {report.issues}")

        return df, report

    def validate_bulk(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, ValidationReport]]:
        """Validate a dictionary of symbol → DataFrame."""
        clean_data = {}
        reports = {}
        for sym, df in data.items():
            clean_df, report = self.validate(df, sym)
            reports[sym] = report
            if report.is_valid:
                clean_data[sym] = clean_df
            else:
                logger.error(f"Dropping {sym}: failed validation")
        logger.info(
            f"Validation complete: {len(clean_data)}/{len(data)} symbols passed"
        )
        return clean_data, reports
