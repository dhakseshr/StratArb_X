"""
Yahoo Finance client (via yfinance).

Yahoo Finance provides:
  - Free daily OHLCV going back decades
  - Adjusted close prices (splits + dividends)
  - Fundamental data (P/E, market cap, etc.)
  - Options chains
  - Forex and crypto data

Advantages: completely free, huge history, simple API
Limitations: no tick data, occasional data gaps, no official SLA
Best use: research, backtesting, pair universe construction
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger


class YahooClient:
    """
    yfinance wrapper optimized for multi-ticker batch downloads
    and pair-trading research workflows.
    """

    def get_ohlcv(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
        auto_adjust: bool = True,
        threads: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch download OHLCV for multiple symbols.

        Args:
            symbols:     list of Yahoo tickers (e.g. ['JPM', 'BAC', 'GS'])
            start:       'YYYY-MM-DD'
            end:         'YYYY-MM-DD'
            interval:    '1m','2m','5m','15m','30m','60m','1h','1d','1wk','1mo'
            auto_adjust: adjust for splits and dividends
            threads:     parallel download

        Returns:
            dict[symbol → DataFrame with OHLCV columns]
        """
        logger.info(f"Yahoo: downloading {len(symbols)} symbols {start}→{end} ({interval})")

        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            threads=threads,
            progress=False,
        )

        result = {}
        if isinstance(raw.columns, pd.MultiIndex):
            # Multiple symbols: columns are (metric, symbol)
            for sym in symbols:
                try:
                    df = raw.xs(sym, level=1, axis=1).copy()
                    df.columns = [c.lower() for c in df.columns]
                    df = df.dropna(how="all")
                    if not df.empty:
                        result[sym] = df
                except Exception as e:
                    logger.warning(f"Yahoo: no data for {sym}: {e}")
        else:
            # Single symbol
            sym = symbols[0]
            df = raw.copy()
            df.columns = [c.lower() for c in df.columns]
            result[sym] = df.dropna(how="all")

        return result

    def get_prices(
        self,
        symbols: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Returns adjusted close price matrix (columns = symbols).
        Most convenient format for pair research.
        """
        data = self.get_ohlcv(symbols, start, end)
        prices = {}
        for sym, df in data.items():
            col = "adj close" if "adj close" in df.columns else "close"
            prices[sym] = df[col]
        return pd.DataFrame(prices).dropna(how="all")

    def get_fundamentals(self, symbol: str) -> dict:
        """Fetch fundamental data (P/E, market cap, sector, etc.)."""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "beta": info.get("beta"),
            "short_ratio": info.get("shortRatio"),
        }

    def get_universe_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch fundamentals for a list of symbols."""
        records = []
        for sym in symbols:
            try:
                records.append(self.get_fundamentals(sym))
            except Exception as e:
                logger.warning(f"Yahoo fundamentals: {sym} failed: {e}")
        return pd.DataFrame(records).set_index("symbol")
