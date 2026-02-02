"""
Alpha Vantage data client.

Alpha Vantage provides:
  - Intraday OHLCV (1min to 60min) up to 2 years
  - Daily, weekly, monthly bars
  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Fundamental data (earnings, balance sheet, income statement)
  - Forex and crypto data
  - Economic indicators (CPI, GDP, etc.)

Advantages: free tier (25 req/day), rich fundamental + technical data
Limitations: rate limited on free tier, smaller history vs Polygon
Best use: fundamental features, technical indicator cross-validation
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings


class AlphaVantageClient:
    """Alpha Vantage REST API wrapper."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.data_sources.alpha_vantage_key
        self._request_count = 0
        self._last_request_time = 0.0

    def _throttle(self, calls_per_minute: int = 5):
        """Respect rate limits (free tier: 5 calls/min, 500/day)."""
        min_interval = 60.0 / calls_per_minute
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch(self, params: dict) -> dict:
        import requests
        self._throttle()
        params["apikey"] = self.api_key
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        return data

    def get_daily(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """
        Daily adjusted OHLCV.

        Args:
            symbol:     ticker
            outputsize: 'compact' (100 days) or 'full' (20+ years)
        """
        logger.info(f"AlphaVantage: daily bars for {symbol}")
        data = self._fetch({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
        })
        ts = data.get("Time Series (Daily)", {})
        records = []
        for date_str, vals in ts.items():
            records.append({
                "timestamp": pd.Timestamp(date_str),
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "adj_close": float(vals["5. adjusted close"]),
                "volume": float(vals["6. volume"]),
                "dividend": float(vals["7. dividend amount"]),
                "split_coeff": float(vals["8. split coefficient"]),
            })
        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        return df

    def get_intraday(
        self,
        symbol: str,
        interval: str = "5min",
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """Intraday OHLCV bars (1min, 5min, 15min, 30min, 60min)."""
        logger.info(f"AlphaVantage: {interval} intraday bars for {symbol}")
        data = self._fetch({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "adjusted": "true",
            "extended_hours": "false",
        })
        key = f"Time Series ({interval})"
        ts = data.get(key, {})
        records = []
        for dt_str, vals in ts.items():
            records.append({
                "timestamp": pd.Timestamp(dt_str),
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "volume": float(vals["5. volume"]),
            })
        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        return df

    def get_rsi(self, symbol: str, period: int = 14, interval: str = "daily") -> pd.Series:
        """Fetch RSI technical indicator."""
        data = self._fetch({
            "function": "RSI",
            "symbol": symbol,
            "interval": interval,
            "time_period": period,
            "series_type": "close",
        })
        ts = data.get("Technical Analysis: RSI", {})
        series = {pd.Timestamp(k): float(v["RSI"]) for k, v in ts.items()}
        return pd.Series(series, name=f"RSI_{period}").sort_index()

    def get_earnings(self, symbol: str) -> pd.DataFrame:
        """Fetch quarterly earnings (EPS actual vs. estimate)."""
        data = self._fetch({"function": "EARNINGS", "symbol": symbol})
        quarterly = data.get("quarterlyEarnings", [])
        df = pd.DataFrame(quarterly)
        if not df.empty and "fiscalDateEnding" in df.columns:
            df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
            df = df.set_index("fiscalDateEnding").sort_index()
        return df
