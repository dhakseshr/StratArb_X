"""
Central configuration for StatArb_X.
Reads from environment variables / .env file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataSourceConfig:
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_secret_key: str = os.getenv("BINANCE_SECRET_KEY", "")
    ibkr_host: str = os.getenv("IBKR_HOST", "127.0.0.1")
    ibkr_port: int = int(os.getenv("IBKR_PORT", "7497"))
    ibkr_client_id: int = int(os.getenv("IBKR_CLIENT_ID", "1"))


@dataclass
class DatabaseConfig:
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "statarb_x")
    postgres_user: str = os.getenv("POSTGRES_USER", "statarb")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "statarb_secret")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_market_data: str = os.getenv("KAFKA_TOPIC_MARKET_DATA", "market_data")
    topic_signals: str = os.getenv("KAFKA_TOPIC_SIGNALS", "signals")
    topic_orders: str = os.getenv("KAFKA_TOPIC_ORDERS", "orders")


@dataclass
class StrategyConfig:
    universe: List[str] = field(
        default_factory=lambda: os.getenv("UNIVERSE", "banking,energy,etf").split(",")
    )
    lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "252"))
    zscore_entry: float = float(os.getenv("ZSCORE_ENTRY", "2.0"))
    zscore_exit: float = float(os.getenv("ZSCORE_EXIT", "0.5"))
    zscore_stop: float = float(os.getenv("ZSCORE_STOP", "-4.0"))
    max_pairs: int = int(os.getenv("MAX_PAIRS", "50"))
    min_halflife_days: int = int(os.getenv("MIN_HALFLIFE_DAYS", "2"))
    max_halflife_days: int = int(os.getenv("MAX_HALFLIFE_DAYS", "60"))
    min_correlation: float = 0.70
    min_coint_pvalue: float = 0.05
    rolling_window: int = 60


@dataclass
class RiskConfig:
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.05"))
    max_leverage: float = float(os.getenv("MAX_LEVERAGE", "2.0"))
    max_sector_exposure: float = float(os.getenv("MAX_SECTOR_EXPOSURE", "0.30"))
    daily_loss_limit: float = float(os.getenv("DAILY_LOSS_LIMIT", "0.02"))
    var_confidence: float = float(os.getenv("VAR_CONFIDENCE", "0.99"))
    max_drawdown_limit: float = 0.15


@dataclass
class ExecutionConfig:
    paper_trading: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "5"))
    commission_per_share: float = float(os.getenv("COMMISSION_PER_SHARE", "0.005"))
    latency_ms: int = int(os.getenv("LATENCY_MS", "10"))


@dataclass
class Settings:
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "storage")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
