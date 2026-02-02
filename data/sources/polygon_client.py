"""
Polygon.io market data client.

Polygon provides:
  - Tick data (every trade)
  - L1 quotes (NBBO)
  - L2 order book snapshots
  - OHLCV aggregates (from 1-second to 1-year)
  - WebSocket streaming for real-time data
  - REST API for historical data

Advantages: institutional-grade, full tick history, options chains
Cost: paid tiers required for tick data
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings
from data.models.market_data import OHLCV, Quote, Tick


class PolygonClient:
    """
    Wrapper around polygon-api-client.
    Provides both REST (historical) and WebSocket (streaming) access.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.data_sources.polygon_api_key
        self._client = None
        self._ws_client = None

    def _get_client(self):
        if self._client is None:
            try:
                from polygon import RESTClient
                self._client = RESTClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("polygon-api-client required: pip install polygon-api-client")
        return self._client

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        timespan: str = "day",
        multiplier: int = 1,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from Polygon.

        Args:
            symbol:     ticker (e.g. 'AAPL')
            start:      'YYYY-MM-DD'
            end:        'YYYY-MM-DD'
            timespan:   'minute', 'hour', 'day', 'week', 'month'
            multiplier: bar width multiplier (e.g. multiplier=5, timespan='minute' = 5-min bars)
            adjusted:   split/dividend adjusted

        Returns:
            DataFrame with columns: [open, high, low, close, volume, vwap, timestamp]
        """
        client = self._get_client()
        logger.info(f"Polygon: fetching {multiplier}{timespan} bars for {symbol} {start}→{end}")

        aggs = client.get_aggs(
            ticker=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
            adjusted=adjusted,
            sort="asc",
            limit=50000,
        )

        records = []
        for bar in aggs:
            records.append({
                "timestamp": pd.Timestamp(bar.timestamp, unit="ms", tz="UTC"),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": getattr(bar, "vwap", None),
                "trade_count": getattr(bar, "transactions", None),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("timestamp").sort_index()
        return df

    def get_ticks(self, symbol: str, trade_date: str) -> List[Tick]:
        """Fetch all trades for a symbol on a given date (tick data)."""
        client = self._get_client()
        logger.info(f"Polygon: fetching ticks for {symbol} on {trade_date}")

        trades = client.list_trades(ticker=symbol, timestamp=trade_date, limit=50000)
        ticks = []
        for t in trades:
            ticks.append(Tick(
                symbol=symbol,
                timestamp=pd.Timestamp(t.sip_timestamp, unit="ns", tz="UTC").to_pydatetime(),
                price=t.price,
                size=t.size,
                exchange=getattr(t, "exchange", ""),
                conditions=getattr(t, "conditions", []) or [],
            ))
        return ticks

    def get_quotes(self, symbol: str, trade_date: str) -> List[Quote]:
        """Fetch L1 NBBO quotes for a symbol on a given date."""
        client = self._get_client()
        logger.info(f"Polygon: fetching quotes for {symbol} on {trade_date}")

        quotes_raw = client.list_quotes(ticker=symbol, timestamp=trade_date, limit=50000)
        quotes = []
        for q in quotes_raw:
            quotes.append(Quote(
                symbol=symbol,
                timestamp=pd.Timestamp(q.sip_timestamp, unit="ns", tz="UTC").to_pydatetime(),
                bid_price=q.bid_price,
                bid_size=q.bid_size,
                ask_price=q.ask_price,
                ask_size=q.ask_size,
            ))
        return quotes

    def get_snapshot(self, symbols: List[str]) -> dict:
        """Get current L1 snapshot for multiple symbols (real-time)."""
        client = self._get_client()
        snapshots = client.get_snapshot_all(market_type="stocks", tickers=symbols)
        result = {}
        for snap in snapshots:
            result[snap.ticker] = {
                "price": snap.day.close if snap.day else None,
                "bid": snap.last_quote.bid_price if snap.last_quote else None,
                "ask": snap.last_quote.ask_price if snap.last_quote else None,
                "volume": snap.day.volume if snap.day else None,
            }
        return result

    def get_bulk_ohlcv(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timespan: str = "day",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols. Returns dict[symbol → DataFrame]."""
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get_ohlcv(sym, start, end, timespan)
            except Exception as e:
                logger.warning(f"Polygon: failed to fetch {sym}: {e}")
        return result
