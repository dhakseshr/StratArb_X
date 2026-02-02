"""
Data Storage Architecture.

Storage technology choices:

  PostgreSQL + TimescaleDB
  ─────────────────────────
  Why: TimescaleDB is a PostgreSQL extension that adds hypertables —
       automatic time-partitioned tables with built-in compression.
       SQL interface for complex analytical queries.
       ACID compliance for trade/position records.
       Best for: OHLCV history, trade logs, backtesting results.

  Redis
  ─────
  Why: In-memory key-value store. Sub-millisecond latency.
       Used for real-time signal caching, latest quotes, portfolio state.
       Pub/Sub for event distribution.
       Best for: hot path data, real-time state, feature cache.

  Parquet (on disk / S3)
  ───────────────────────
  Why: Columnar storage format with snappy/zstd compression.
       10-100x smaller than CSV. Preserves dtype metadata.
       Fast column-wise reads (critical for feature computation).
       Best for: raw data archive, research datasets, ML features.

Pipeline flow:
  Raw → Parquet (archive) → TimescaleDB (queryable) → Redis (hot cache)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings


class ParquetStorage:
    """
    Local Parquet archive for raw market data.
    Organized as: data/storage/{symbol}/{year}/{month}.parquet
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or settings.data_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, year: int, month: int) -> Path:
        p = self.base_dir / symbol / str(year)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{month:02d}.parquet"

    def write(self, symbol: str, df: pd.DataFrame) -> None:
        """Write DataFrame partitioned by year/month."""
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        df["_year"] = df.index.year
        df["_month"] = df.index.month
        for (year, month), group in df.groupby(["_year", "_month"]):
            path = self._path(symbol, year, month)
            group = group.drop(columns=["_year", "_month"])
            group.to_parquet(path, engine="pyarrow", compression="snappy")
            logger.debug(f"Parquet: wrote {len(group)} rows → {path}")

    def read(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read Parquet files for a symbol, optionally filtered by date range."""
        sym_dir = self.base_dir / symbol
        if not sym_dir.exists():
            return pd.DataFrame()

        frames = []
        for parquet_file in sorted(sym_dir.rglob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_file, engine="pyarrow")
                frames.append(df)
            except Exception as e:
                logger.warning(f"Parquet read error {parquet_file}: {e}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        if start:
            combined = combined[combined.index >= start]
        if end:
            combined = combined[combined.index <= end]
        return combined

    def read_bulk(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        column: str = "close",
    ) -> pd.DataFrame:
        """Read a single column for multiple symbols. Returns price matrix."""
        series = {}
        for sym in symbols:
            df = self.read(sym, start, end)
            if not df.empty and column in df.columns:
                series[sym] = df[column]
        return pd.DataFrame(series)


class TimescaleDBStorage:
    """
    TimescaleDB (PostgreSQL + hypertables) for queryable OHLCV storage.

    Schema:
      market_data (time TIMESTAMPTZ, symbol TEXT, open, high, low, close, volume, vwap)
      — partitioned by time (daily chunks)
      — indexed on (symbol, time)

    Compression: automatic after 7 days (typically 10-20x compression ratio)
    """

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine
            self._engine = create_engine(
                settings.database.postgres_dsn,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        return self._engine

    def initialize_schema(self) -> None:
        """Create hypertables and indexes if not exist."""
        ddl = """
        CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

        CREATE TABLE IF NOT EXISTS market_data (
            time        TIMESTAMPTZ NOT NULL,
            symbol      TEXT        NOT NULL,
            open        DOUBLE PRECISION,
            high        DOUBLE PRECISION,
            low         DOUBLE PRECISION,
            close       DOUBLE PRECISION,
            volume      DOUBLE PRECISION,
            vwap        DOUBLE PRECISION,
            trade_count INTEGER,
            timeframe   TEXT DEFAULT '1d'
        );

        SELECT create_hypertable(
            'market_data', 'time',
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '1 month'
        );

        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time
            ON market_data (symbol, time DESC);

        -- Enable compression after 7 days
        ALTER TABLE market_data SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol'
        );

        SELECT add_compression_policy('market_data',
            INTERVAL '7 days', if_not_exists => TRUE);

        -- Research output tables
        CREATE TABLE IF NOT EXISTS pairs (
            id              SERIAL PRIMARY KEY,
            symbol_a        TEXT,
            symbol_b        TEXT,
            hedge_ratio     DOUBLE PRECISION,
            half_life_days  DOUBLE PRECISION,
            coint_pvalue    DOUBLE PRECISION,
            correlation     DOUBLE PRECISION,
            sharpe_spread   DOUBLE PRECISION,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS signals (
            time        TIMESTAMPTZ NOT NULL,
            pair_id     INTEGER REFERENCES pairs(id),
            zscore      DOUBLE PRECISION,
            signal      TEXT,          -- 'long_spread','short_spread','exit','stop'
            confidence  DOUBLE PRECISION,
            hedge_ratio DOUBLE PRECISION
        );

        SELECT create_hypertable(
            'signals', 'time',
            if_not_exists => TRUE
        );

        CREATE TABLE IF NOT EXISTS trades (
            id              SERIAL PRIMARY KEY,
            time            TIMESTAMPTZ NOT NULL,
            pair_id         INTEGER REFERENCES pairs(id),
            side            TEXT,       -- 'entry', 'exit', 'stop'
            symbol          TEXT,
            direction       TEXT,       -- 'long', 'short'
            quantity        DOUBLE PRECISION,
            fill_price      DOUBLE PRECISION,
            commission      DOUBLE PRECISION,
            slippage        DOUBLE PRECISION,
            pnl             DOUBLE PRECISION
        );
        """
        engine = self._get_engine()
        with engine.connect() as conn:
            conn.execute(ddl)
            conn.commit()
        logger.info("TimescaleDB schema initialized")

    def write_ohlcv(self, symbol: str, df: pd.DataFrame, timeframe: str = "1d") -> None:
        """Upsert OHLCV data into the hypertable."""
        if df.empty:
            return
        df = df.copy()
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        df.index.name = "time"
        df = df.reset_index()
        engine = self._get_engine()
        df.to_sql(
            "market_data",
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
        logger.debug(f"TimescaleDB: wrote {len(df)} rows for {symbol}")

    def read_ohlcv(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Read OHLCV data from TimescaleDB."""
        engine = self._get_engine()
        sym_list = ", ".join(f"'{s}'" for s in symbols)
        query = f"""
            SELECT time, symbol, open, high, low, close, volume, vwap
            FROM market_data
            WHERE symbol IN ({sym_list})
              AND time BETWEEN '{start}' AND '{end}'
              AND timeframe = '{timeframe}'
            ORDER BY symbol, time ASC
        """
        df = pd.read_sql(query, engine, parse_dates=["time"])
        result = {}
        for sym in symbols:
            sym_df = df[df["symbol"] == sym].set_index("time").drop(columns=["symbol"])
            if not sym_df.empty:
                result[sym] = sym_df
        return result


class RedisCache:
    """
    Redis for real-time hot-path data:
      - Latest quotes (TTL: 1 second)
      - Current z-scores (TTL: 1 minute)
      - Portfolio state (TTL: no expiry, updated on change)
      - Feature vectors (TTL: 5 minutes)
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis
            self._client = redis.Redis(
                host=settings.database.redis_host,
                port=settings.database.redis_port,
                db=settings.database.redis_db,
                decode_responses=True,
            )
        return self._client

    def set_latest_price(self, symbol: str, price: float, ttl: int = 5) -> None:
        r = self._get_client()
        r.setex(f"price:{symbol}", ttl, str(price))

    def get_latest_price(self, symbol: str) -> Optional[float]:
        r = self._get_client()
        val = r.get(f"price:{symbol}")
        return float(val) if val else None

    def set_zscore(self, pair_key: str, zscore: float, ttl: int = 60) -> None:
        r = self._get_client()
        r.setex(f"zscore:{pair_key}", ttl, str(zscore))

    def get_zscore(self, pair_key: str) -> Optional[float]:
        r = self._get_client()
        val = r.get(f"zscore:{pair_key}")
        return float(val) if val else None

    def publish_signal(self, channel: str, message: str) -> None:
        """Publish trading signal via Redis pub/sub."""
        r = self._get_client()
        r.publish(channel, message)

    def subscribe_signals(self, channel: str):
        """Subscribe to signal channel."""
        r = self._get_client()
        pubsub = r.pubsub()
        pubsub.subscribe(channel)
        return pubsub


class DataStorage:
    """
    Unified storage interface combining all backends.

    Write path: data → Parquet (archive) + TimescaleDB (queryable) + Redis (hot)
    Read path: Redis (if fresh) → TimescaleDB → Parquet (fallback)
    """

    def __init__(self):
        self.parquet = ParquetStorage()
        self.timescale = TimescaleDBStorage()
        self.redis = RedisCache()

    def store_ohlcv(self, symbol: str, df: pd.DataFrame, timeframe: str = "1d") -> None:
        """Store OHLCV in all backends."""
        self.parquet.write(symbol, df)
        try:
            self.timescale.write_ohlcv(symbol, df, timeframe)
        except Exception as e:
            logger.warning(f"TimescaleDB write failed for {symbol}: {e}")

    def load_prices(
        self,
        symbols: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Load price matrix (close prices). Falls back to Parquet if DB unavailable."""
        try:
            data = self.timescale.read_ohlcv(symbols, start, end)
            return pd.DataFrame({s: df["close"] for s, df in data.items()})
        except Exception as e:
            logger.warning(f"TimescaleDB read failed, falling back to Parquet: {e}")
            return self.parquet.read_bulk(symbols, start, end, column="close")
