"""
Interactive Brokers (IBKR) data client via ib_insync.

IBKR provides:
  - Real-time and historical data for stocks, options, futures, forex, bonds
  - Level 2 order book (market depth)
  - Tick data (time & sales)
  - Options chains with Greeks
  - Fundamental data (Reuters/Morningstar)
  - Also handles order execution (same connection)

Advantages: institutional-grade, breadth of instruments, low latency
Requirements: TWS or IB Gateway running locally, API enabled in settings
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings


class IBKRClient:
    """
    Interactive Brokers client via ib_insync.
    Requires TWS or IB Gateway running on localhost.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ):
        self.host = host or settings.data_sources.ibkr_host
        self.port = port or settings.data_sources.ibkr_port
        self.client_id = client_id or settings.data_sources.ibkr_client_id
        self._ib = None

    def connect(self) -> None:
        """Connect to TWS/Gateway."""
        try:
            import ib_insync
            self._ib = ib_insync.IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"IBKR: connected to {self.host}:{self.port}")
        except ImportError:
            raise ImportError("ib_insync required: pip install ib_insync")

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("IBKR: disconnected")

    def _ensure_connected(self):
        if self._ib is None or not self._ib.isConnected():
            self.connect()

    def _stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        import ib_insync
        return ib_insync.Stock(symbol, exchange, currency)

    def get_historical_bars(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "ADJUSTED_LAST",
    ) -> pd.DataFrame:
        """
        Fetch historical bars from IBKR.

        Args:
            symbol:       ticker
            duration:     '1 D', '1 W', '1 M', '1 Y', '5 Y'
            bar_size:     '1 secs','5 secs','1 min','5 mins','1 hour','1 day'
            what_to_show: 'TRADES','MIDPOINT','BID','ASK','ADJUSTED_LAST'
        """
        self._ensure_connected()
        contract = self._stock_contract(symbol)
        self._ib.qualifyContracts(contract)

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
        )

        df = pd.DataFrame([{
            "timestamp": b.date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "average": b.average,
            "bar_count": b.barCount,
        } for b in bars])

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
        return df

    def get_market_depth(self, symbol: str, num_rows: int = 20) -> dict:
        """
        Subscribe to Level 2 market depth (order book).

        Args:
            symbol:   ticker
            num_rows: depth levels to request

        Returns:
            dict with 'bids' and 'asks' DataFrames
        """
        self._ensure_connected()
        contract = self._stock_contract(symbol)
        self._ib.qualifyContracts(contract)

        ticker = self._ib.reqMktDepth(contract, numRows=num_rows)
        self._ib.sleep(2)  # wait for data

        bids = pd.DataFrame([
            {"price": d.price, "size": d.size, "marketMaker": d.marketMaker}
            for d in ticker.domBids
        ])
        asks = pd.DataFrame([
            {"price": d.price, "size": d.size, "marketMaker": d.marketMaker}
            for d in ticker.domAsks
        ])
        self._ib.cancelMktDepth(contract)
        return {"bids": bids, "asks": asks}

    def get_scanner_results(
        self,
        scan_code: str = "TOP_PERC_GAIN",
        num_rows: int = 50,
        above_price: float = 5.0,
        below_price: float = 500.0,
        above_volume: int = 100_000,
    ) -> List[str]:
        """Run IBKR scanner to discover liquid, active stocks."""
        self._ensure_connected()
        import ib_insync
        sub = ib_insync.ScannerSubscription(
            instrument="STK",
            locationCode="STK.US.MAJOR",
            scanCode=scan_code,
            abovePrice=above_price,
            belowPrice=below_price,
            aboveVolume=above_volume,
            numberOfRows=num_rows,
        )
        results = self._ib.reqScannerData(sub)
        symbols = [r.contractDetails.contract.symbol for r in results]
        logger.info(f"IBKR scanner ({scan_code}): found {len(symbols)} symbols")
        return symbols
