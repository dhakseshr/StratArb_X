"""
Alpaca Markets data client.

Alpaca provides:
  - Commission-free brokerage (also used for execution)
  - Historical OHLCV via Data API v2
  - Real-time streaming via WebSocket
  - Paper trading environment (great for testing)
  - Options data (newer feature)

Advantages: free tier available, same API for data + execution,
            excellent for backtesting → live trading pipeline
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings


class AlpacaClient:
    """
    Alpaca Data API v2 wrapper.
    Supports both paper and live environments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.data_sources.alpaca_api_key
        self.secret_key = secret_key or settings.data_sources.alpaca_secret_key
        self.base_url = base_url or settings.data_sources.alpaca_base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                )
            except ImportError:
                raise ImportError("alpaca-py required: pip install alpaca-py")
        return self._client

    def get_ohlcv(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timeframe: str = "1Day",
        feed: str = "iex",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for multiple symbols.

        Args:
            symbols:   list of tickers
            start:     'YYYY-MM-DD'
            end:       'YYYY-MM-DD'
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            feed:      'iex' (free) or 'sip' (paid, full NBBO)

        Returns:
            dict[symbol → DataFrame]
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

        client = self._get_client()
        logger.info(f"Alpaca: fetching {timeframe} bars for {symbols} {start}→{end}")

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
            feed=feed,
        )
        bars = client.get_stock_bars(request)
        result = {}
        for sym in symbols:
            try:
                df = bars[sym].df
                df.index = df.index.tz_convert("UTC")
                result[sym] = df[["open", "high", "low", "close", "volume", "vwap", "trade_count"]]
            except Exception as e:
                logger.warning(f"Alpaca: no data for {sym}: {e}")
        return result

    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        """Get latest L1 bid/ask for multiple symbols."""
        from alpaca.data.requests import StockLatestQuoteRequest

        client = self._get_client()
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = client.get_stock_latest_quote(request)
        result = {}
        for sym, q in quotes.items():
            result[sym] = {
                "bid": q.bid_price,
                "bid_size": q.bid_size,
                "ask": q.ask_price,
                "ask_size": q.ask_size,
                "timestamp": q.timestamp,
            }
        return result

    def stream_bars(self, symbols: List[str], handler) -> None:
        """
        Stream real-time 1-minute bars via WebSocket.

        Args:
            symbols: list of tickers to subscribe
            handler: async callback(bar) function
        """
        from alpaca.data.live import StockDataStream

        stream = StockDataStream(api_key=self.api_key, secret_key=self.secret_key)

        async def _handler(bar):
            await handler(bar)

        stream.subscribe_bars(_handler, *symbols)
        logger.info(f"Alpaca: streaming bars for {symbols}")
        stream.run()
