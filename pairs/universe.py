"""
Candidate Universe Selection.

The goal is to find pairs that have a plausible structural reason
to share a long-run price relationship. Purely statistical pairs
(no fundamental link) tend to be spurious.

Universe categories:

  Banking Stocks
  ─────────────
  Large US banks are subject to identical macro forces: Fed rate
  decisions, credit cycle, regulatory environment. JPM vs. BAC
  is a classic pair — both driven by the same interest rate regime.

  Energy Stocks
  ─────────────
  Exploration companies share crude oil price exposure. Refiners
  share crack spread exposure. XOM vs. CVX, SLB vs. HAL.

  ETF Pairs
  ─────────
  ETFs tracking the same or similar index (e.g., SPY vs. IVV,
  GLD vs. IAU) are nearly identical products with arbitrage bounds.
  Also sector ETF vs. its top holdings.

  Crypto Pairs
  ─────────────
  BTC/ETH are driven by the same crypto market sentiment. Layer-1
  blockchains (SOL, AVAX, ADA) share user growth and TVL dynamics.

  ADRs
  ─────
  A stock listed on both NYSE and a foreign exchange (e.g., BABA
  on NYSE and 9988.HK on HKEX). Same underlying company, but prices
  can diverge due to FX, market hours, and local liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ─── Pre-defined universes ────────────────────────────────────────────────────

BANKING_STOCKS = [
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC",
    "TFC", "COF", "SCHW", "BK", "STT", "FITB", "KEY", "RF",
]

ENERGY_STOCKS = [
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "PSX",
    "VLO", "MPC", "OXY", "DVN", "FANG", "PXD", "HES",
]

TECH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD",
    "INTC", "QCOM", "TXN", "AVGO", "MU", "LRCX", "KLAC",
]

PHARMA_STOCKS = [
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD",
    "BIIB", "REGN", "VRTX", "LLY", "ZTS",
]

ETF_PAIRS = [
    # Same-index ETFs (almost perfect co-integration)
    ("SPY", "IVV"), ("SPY", "VOO"), ("IVV", "VOO"),
    # Gold ETFs
    ("GLD", "IAU"), ("GLD", "SGOL"),
    # Sector vs. broad market
    ("XLF", "KBE"), ("XLE", "OIH"),
    ("XLK", "QQQ"), ("XLV", "IHF"),
    # Bond ETFs
    ("TLT", "IEF"), ("HYG", "JNK"),
    # Emerging market
    ("EEM", "VWO"),
]

CRYPTO_PAIRS = [
    ("BTCUSDT", "ETHUSDT"),
    ("SOLUSDT", "AVAXUSDT"),
    ("SOLUSDT", "NEARUSDT"),
    ("BNBUSDT", "MATICUSDT"),
    ("ADAUSDT", "DOTUSDT"),
    ("LINKUSDT", "UNIUSDT"),
]

ADR_PAIRS = [
    # ADR vs. foreign-listed equivalent
    ("BABA", "9988.HK"),
    ("TSM", "2330.TW"),
    ("NIO", "NIO"),  # NYSE + HK crosslist
    ("JD", "9618.HK"),
]

SECTOR_MAP = {
    "banking": BANKING_STOCKS,
    "energy": ENERGY_STOCKS,
    "tech": TECH_STOCKS,
    "pharma": PHARMA_STOCKS,
}


@dataclass
class CandidateUniverse:
    """
    Manages the universe of candidate pairs for co-integration testing.

    Strategy:
    1. Start with sector-constrained list (fundamental similarity)
    2. Generate all N×(N-1)/2 unique pairs within sector
    3. Pass to CorrelationFilter (fast pre-screen)
    4. Pass survivors to CointegrationTester (expensive, run once/week)
    """

    sectors: List[str] = field(default_factory=lambda: ["banking", "energy"])
    include_etfs: bool = True
    include_crypto: bool = False
    custom_symbols: List[str] = field(default_factory=list)

    def get_symbols(self) -> List[str]:
        """All individual symbols in the universe."""
        symbols = []
        for sector in self.sectors:
            symbols.extend(SECTOR_MAP.get(sector, []))
        symbols.extend(self.custom_symbols)
        if self.include_etfs:
            # Flatten ETF pairs list
            for a, b in ETF_PAIRS:
                if a not in symbols:
                    symbols.append(a)
                if b not in symbols:
                    symbols.append(b)
        return list(dict.fromkeys(symbols))  # deduplicate, preserve order

    def get_candidate_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate all candidate pairs.
        Within-sector pairs + pre-defined ETF pairs + crypto pairs.
        """
        pairs = []

        # Within-sector pairs (strong fundamental link)
        for sector in self.sectors:
            sector_stocks = SECTOR_MAP.get(sector, [])
            for i, a in enumerate(sector_stocks):
                for b in sector_stocks[i + 1:]:
                    pairs.append((a, b))

        # Pre-defined ETF pairs
        if self.include_etfs:
            pairs.extend(ETF_PAIRS)

        # Crypto pairs
        if self.include_crypto:
            pairs.extend(CRYPTO_PAIRS)

        # Custom symbols: all combinations
        n = len(self.custom_symbols)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self.custom_symbols[i], self.custom_symbols[j]))

        return list(dict.fromkeys(pairs))  # deduplicate

    def filter_by_liquidity(
        self,
        prices: pd.DataFrame,
        min_price: float = 5.0,
        min_avg_volume: Optional[float] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        Remove illiquid or penny stocks from universe.
        Low-price stocks have wide spreads and high impact costs.
        """
        valid = []
        for sym in prices.columns:
            avg_price = prices[sym].mean()
            if avg_price < min_price:
                continue
            if min_avg_volume and volumes is not None and sym in volumes:
                avg_vol = volumes[sym].mean()
                if avg_vol < min_avg_volume:
                    continue
            valid.append(sym)
        return valid
