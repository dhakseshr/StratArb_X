"""
Binance cryptocurrency data client.

Binance provides:
  - Spot, futures, perpetual swap markets
  - Tick-by-tick trade data
  - Level 2 order book (up to 5000 levels)
  - OHLCV klines (1s, 1m, ... 1M)
  - Funding rates (for perpetual futures)
  - Open interest, liquidation data

Advantages: deepest crypto liquidity, rich data, free REST/WebSocket
Use case: crypto pair trading (BTC/ETH, SOL/AVAX, etc.), basis trading
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings


class BinanceDataClient:
    """
    Binance data client for crypto pair trading research.
    Supports spot and futures markets.
    """

    SPOT_BASE = "https://api.binance.com"
    FUTURES_BASE = "https://fapi.binance.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.api_key = api_key or settings.data_sources.binance_api_key
        self.secret_key = secret_key or settings.data_sources.binance_secret_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from binance.client import Client
                self._client = Client(api_key=self.api_key, api_secret=self.secret_key)
            except ImportError:
                raise ImportError("python-binance required: pip install python-binance")
        return self._client

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines.

        Args:
            symbol:   e.g. 'BTCUSDT', 'ETHUSDT'
            interval: '1m','3m','5m','15m','30m','1h','4h','1d','1w','1M'
            start:    'YYYY-MM-DD' or timestamp ms
            end:      'YYYY-MM-DD' or timestamp ms
            limit:    max 1000 per request (paginate for more)
        """
        client = self._get_client()
        logger.info(f"Binance: {interval} klines for {symbol}")

        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start,
            end_str=end,
            limit=limit,
        )

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(klines, columns=columns)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                        "taker_buy_base", "taker_buy_quote"]
        for col in numeric_cols:
            df[col] = df[col].astype(float)
        df["trade_count"] = df["trade_count"].astype(int)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume", "trade_count"]]
        return df.sort_index()

    def get_bulk_klines(
        self,
        symbols: List[str],
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch klines for multiple crypto symbols."""
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get_klines(sym, interval, start, end)
            except Exception as e:
                logger.warning(f"Binance: failed {sym}: {e}")
        return result

    def get_order_book(self, symbol: str, depth: int = 100) -> dict:
        """
        Fetch L2 order book snapshot.

        Returns dict with 'bids' and 'asks' as DataFrames (price, quantity).
        Depth options: 5, 10, 20, 50, 100, 500, 1000, 5000
        """
        client = self._get_client()
        book = client.get_order_book(symbol=symbol, limit=depth)
        bids = pd.DataFrame(book["bids"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(book["asks"], columns=["price", "qty"], dtype=float)
        return {
            "symbol": symbol,
            "lastUpdateId": book["lastUpdateId"],
            "bids": bids.sort_values("price", ascending=False),
            "asks": asks.sort_values("price", ascending=True),
        }

    def get_funding_rates(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch perpetual futures funding rate history.
        Funding rate is paid every 8 hours — positive = longs pay shorts.
        """
        client = self._get_client()
        rates = client.futures_funding_rate(symbol=symbol, limit=limit)
        df = pd.DataFrame(rates)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        return df.set_index("fundingTime").sort_index()

    def get_crypto_pair_universe(self) -> List[str]:
        """Return list of USDT-margined perpetual symbols for pair discovery."""
        client = self._get_client()
        info = client.futures_exchange_info()
        symbols = [
            s["symbol"] for s in info["symbols"]
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
        ]
        return sorted(symbols)
