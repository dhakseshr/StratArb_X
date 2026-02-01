# StatArb_X — Institutional Statistical Arbitrage Engine

A production-grade statistical arbitrage system implementing pair trading strategies
with dynamic hedge ratios, machine learning signal enhancement, and real-time execution.

---

## What is Statistical Arbitrage?

Statistical arbitrage (StatArb) exploits temporary price divergences between related
securities, betting that prices will revert to their historical equilibrium.

**Core Mechanism:**
1. Find pairs of assets with stable long-run price relationship (co-integration)
2. Measure deviation from equilibrium → construct **spread**
3. Enter market-neutral position when spread diverges beyond threshold
4. Exit when spread reverts to mean → capture profit

Unlike pure directional bets, StatArb is **market-neutral**: simultaneous long/short
positions cancel out broad market moves, leaving only the spread dynamics.

---

## Why Co-Integration (Not Just Correlation)?

| Property | Correlation | Co-Integration |
|---|---|---|
| Measures | Linear co-movement | Long-run equilibrium |
| Stationary | No | Residuals are stationary |
| Regime-stable | Breaks down | More stable |
| Actionable | Weak signal | Strong mean-reversion signal |

**Correlation** measures how two series move together *right now*.  
**Co-integration** means two non-stationary series share a common stochastic trend —
their linear combination is stationary, enabling mean-reversion trading.

Mathematically: If `X_t` and `Y_t` are I(1), and `∃β s.t. Y_t - βX_t ~ I(0)`,
then `(X_t, Y_t)` are co-integrated with co-integrating vector `(1, -β)`.

---

## Alpha Generation Mechanism

```
Price Divergence → Z-score Spike → Entry Signal → Position → Mean Reversion → Profit
```

**Edge sources:**
- Structural relationships (same business, same supply chain)
- Index rebalancing effects
- Liquidity mismatches between related instruments
- Short-term overreaction / underreaction
- Microstructure-driven temporary dislocations

---

## How Top Funds Use This

| Fund | Approach |
|---|---|
| Renaissance Medallion | High-frequency statistical patterns, ML-driven signal discovery |
| Two Sigma | Systematic factor models + co-integration across thousands of pairs |
| Citadel | Multi-strategy: StatArb + options market-making + macro overlays |
| DE Shaw | Deep quantitative research, cross-asset pairs |

All share the same foundation: **data edge + model edge + execution edge**.

---

## System Architecture

```
Market Data (Polygon/Alpaca/Yahoo/Binance/IBKR)
         ↓
    Data Pipeline (Validation → Feature Store → Research DB)
         ↓
    Pair Selection Engine (Correlation → Cointegration → Ranking)
         ↓
    Kalman Filter (Dynamic Hedge Ratio Estimation)
         ↓
    Signal Engine (Z-score → OU Process → Half-Life)
         ↓
    ML Enhancement (XGBoost/LSTM/Transformer signal filtering)
         ↓
    Portfolio Construction (Risk Parity / Kelly / Factor Neutral)
         ↓
    Risk Management (VaR/CVaR/Regime Detection/HMM)
         ↓
    Backtesting Engine (Event-Driven / Walk-Forward Validation)
         ↓
    Production (Kafka → Docker → Kubernetes → Prometheus/Grafana)
         ↓
    Real-Time Dashboard (Streamlit)
```

---

## Project Structure

```
StatArb_X/
├── config/           # Settings, API keys, parameters
├── data/
│   ├── sources/      # Polygon, Alpaca, Yahoo, Binance, IBKR clients
│   ├── pipeline/     # Validation, feature store, storage
│   └── models/       # Market data schemas
├── pairs/            # Universe selection, correlation, co-integration, ranking
├── hedge/            # Kalman filter, OLS, rolling hedge ratios
├── signals/          # Z-score, OU process, half-life, signal engine
├── microstructure/   # Bid-ask spread, slippage, latency, order book
├── portfolio/        # Capital allocation, exposure controls, position sizing
├── risk/             # VaR/CVaR/drawdown, regime detection, constraints
├── backtest/         # Event-driven engine, PnL, walk-forward validation
├── ml/               # Feature engineering, XGBoost/LSTM/Transformer models
├── production/       # Kafka producer/consumer, orchestration pipeline
├── dashboard/        # Real-time Streamlit dashboard
└── tests/            # Unit + integration tests
```

---

## Installation

```bash
git clone https://github.com/dhakseshr/StatArb_X.git
cd StatArb_X
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API credentials

# Start infrastructure
docker-compose up -d

# Run pair discovery
python -m pairs.cointegration --universe banking

# Run backtester
python -m backtest.engine --start 2020-01-01 --end 2024-01-01

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Risk Disclaimer

This software is for **research and educational purposes only**.  
Past performance does not guarantee future results.  
Always paper-trade before live deployment.

---

## License

MIT License — see LICENSE file.
