"""
Market data schemas.

Data types explained:
  Tick        — individual trade event (price, size, timestamp)
  Level1      — best bid/ask (NBBO)
  Level2      — full order book depth
  OHLCV       — Open/High/Low/Close/Volume bar (1m, 5m, 1d, etc.)

Tick data is the most granular — every single trade.
L1 data gives the current best bid/ask spread.
L2 data shows the full depth of the book (multiple price levels).
OHLCV aggregates tick data into time-bucketed bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Tick:
    """Single trade tick. The atomic unit of market data."""
    symbol: str
    timestamp: datetime
    price: float
    size: float          # shares / contracts / base currency
    exchange: str = ""
    conditions: List[str] = field(default_factory=list)
    # Tick type: T=trade, Q=quote
    tick_type: str = "T"


@dataclass
class Quote:
    """Level 1 — National Best Bid/Offer (NBBO)."""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    exchange: str = ""

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid_price) * 10_000


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    size: float
    order_count: int = 1


@dataclass
class OrderBook:
    """Level 2 — full order book depth."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)  # sorted descending
    asks: List[OrderBookLevel] = field(default_factory=list)  # sorted ascending

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    def bid_depth(self, levels: int = 5) -> float:
        """Total bid liquidity in top N levels."""
        return sum(l.size for l in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        return sum(l.size for l in self.asks[:levels])

    def order_imbalance(self, levels: int = 5) -> float:
        """
        Order imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
        Range: [-1, 1]. Positive → more buy pressure.
        """
        b = self.bid_depth(levels)
        a = self.ask_depth(levels)
        total = b + a
        if total == 0:
            return 0.0
        return (b - a) / total


@dataclass
class OHLCV:
    """
    OHLCV bar — the workhorse of quantitative research.
    Aggregates raw ticks into time buckets.

    Fields:
      open   — first trade price in period
      high   — highest trade price in period
      low    — lowest trade price in period
      close  — last trade price in period
      volume — total shares/contracts traded
      vwap   — volume-weighted average price
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    timeframe: str = "1d"  # 1m, 5m, 15m, 1h, 1d

    @property
    def returns(self) -> float:
        return (self.close - self.open) / self.open

    @property
    def hl_range(self) -> float:
        return self.high - self.low


@dataclass
class MarketDataSnapshot:
    """Complete market data snapshot for a symbol at a point in time."""
    symbol: str
    timestamp: datetime
    last_trade: Optional[Tick] = None
    quote: Optional[Quote] = None
    order_book: Optional[OrderBook] = None
    ohlcv_1m: Optional[OHLCV] = None
    ohlcv_1d: Optional[OHLCV] = None
