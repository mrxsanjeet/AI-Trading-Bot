# AI Trading Bot — Project Prompt & Specification

## Overview

Build a **production-grade AI-powered cryptocurrency/stock trading bot** that uses machine learning models to analyze market data, generate trading signals, and execute trades automatically. The bot should support backtesting, paper trading, and live trading modes with robust risk management.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AI Trading Bot                        │
├──────────┬──────────┬───────────┬───────────┬───────────┤
│  Data    │ Strategy │    AI/ML  │ Execution │ Dashboard │
│  Engine  │  Engine  │   Engine  │  Engine   │   (UI)    │
├──────────┼──────────┼───────────┼───────────┼───────────┤
│ Market   │ Signal   │ Price     │ Order     │ Real-time │
│ Data     │ Generate │ Predict   │ Manage    │ Charts    │
│ Fetch    │          │           │           │           │
│ Clean    │ Risk     │ Sentiment │ Portfolio │ Alerts &  │
│ Store    │ Assess   │ Analysis  │ Balance   │ Logs      │
└──────────┴──────────┴───────────┴───────────┴───────────┘
```

---

## Tech Stack

| Layer           | Technology                                      |
| --------------- | ----------------------------------------------- |
| Language        | Python 3.11+                                    |
| ML Framework    | PyTorch / scikit-learn                           |
| Data            | pandas, numpy, ta-lib (technical indicators)    |
| Market Data API | ccxt (crypto), yfinance (stocks), Alpha Vantage  |
| Database        | SQLite (dev) / PostgreSQL (prod)                 |
| Task Queue      | Celery + Redis (scheduled jobs)                  |
| Dashboard       | Streamlit or FastAPI + React                     |
| Testing         | pytest, pytest-cov                               |
| Containerization| Docker + docker-compose                          |

---

## Module Breakdown

### 1. Data Engine (`src/data/`)

- **Market data fetcher** — Connect to exchange APIs via `ccxt` to pull OHLCV (Open, High, Low, Close, Volume) candles at multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d).
- **Data cleaner** — Handle missing values, outliers, and timezone normalization.
- **Feature engineer** — Compute technical indicators: RSI, MACD, Bollinger Bands, EMA, ATR, OBV, Fibonacci levels.
- **Data storage** — Store historical and live data in a local database with efficient querying.
- **News/sentiment feed** — Scrape or fetch news headlines and compute sentiment scores (using a pre-trained NLP model or API).

### 2. Strategy Engine (`src/strategy/`)

- **Signal generator** — Combine AI predictions with technical indicator rules to produce BUY / SELL / HOLD signals.
- **Strategy library** — Implement multiple configurable strategies:
  - Trend Following (EMA crossover, MACD divergence)
  - Mean Reversion (Bollinger Band bounce, RSI extremes)
  - Momentum (breakout, volume spike)
  - AI/ML-based (model-driven predictions)
  - Hybrid (ensemble of the above)
- **Risk manager** — Enforce:
  - Max position size (% of portfolio)
  - Stop-loss & take-profit levels
  - Max drawdown limit (auto-pause trading)
  - Daily loss limit
  - Max number of concurrent open positions
  - Cooldown period after consecutive losses

### 3. AI/ML Engine (`src/ml/`)

- **Price prediction model** — LSTM or Transformer-based model trained on historical OHLCV + technical indicators to predict next-candle direction or price range.
- **Sentiment model** — Classify news/social media sentiment as bullish, bearish, or neutral.
- **Feature pipeline** — Automated feature selection, normalization, and train/test splitting.
- **Model registry** — Version and store trained models with metadata (accuracy, training date, hyperparameters).
- **Training pipeline** — Scheduled retraining with new data; track performance metrics (accuracy, Sharpe ratio, max drawdown on validation set).

### 4. Execution Engine (`src/execution/`)

- **Order manager** — Place market, limit, and stop-limit orders via exchange API.
- **Portfolio tracker** — Track balances, open positions, P&L, and trade history.
- **Paper trading mode** — Simulate order fills against live market data without real money.
- **Live trading mode** — Execute real orders with safety checks and confirmation.
- **Backtesting engine** — Run strategies against historical data with realistic slippage and fee modeling.

### 5. Dashboard (`src/dashboard/`)

- **Real-time dashboard** showing:
  - Portfolio value & P&L chart
  - Open positions and recent trades
  - Active signals and strategy status
  - AI model confidence scores
  - Risk metrics (drawdown, Sharpe, win rate)
- **Alerting** — Send notifications via Telegram, Discord, or email on trade execution, stop-loss triggers, or anomalies.

---

## Project Structure

```
AI-Trading-Bot/
├── config/
│   ├── settings.yaml          # Global config (API keys, DB, trading params)
│   ├── strategies.yaml        # Strategy-specific parameters
│   └── logging.yaml           # Logging configuration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py         # Market data fetching (ccxt, yfinance)
│   │   ├── cleaner.py         # Data cleaning & normalization
│   │   ├── features.py        # Technical indicator computation
│   │   ├── storage.py         # Database read/write operations
│   │   └── sentiment.py       # News sentiment analysis
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract base strategy class
│   │   ├── trend.py           # Trend-following strategies
│   │   ├── mean_reversion.py  # Mean reversion strategies
│   │   ├── momentum.py        # Momentum/breakout strategies
│   │   ├── ml_strategy.py     # ML model-driven strategy
│   │   ├── hybrid.py          # Ensemble/hybrid strategy
│   │   ├── signals.py         # Signal generation logic
│   │   └── risk.py            # Risk management rules
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── models.py          # Model architectures (LSTM, Transformer)
│   │   ├── train.py           # Training pipeline
│   │   ├── predict.py         # Inference pipeline
│   │   ├── features.py        # ML feature engineering
│   │   └── registry.py        # Model versioning & storage
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── orders.py          # Order placement & management
│   │   ├── portfolio.py       # Portfolio tracking
│   │   ├── paper.py           # Paper trading simulator
│   │   ├── live.py            # Live trading executor
│   │   └── backtest.py        # Backtesting engine
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py             # Streamlit/FastAPI dashboard
│   │   └── alerts.py          # Notification system
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Centralized logging
│       └── helpers.py         # Shared utility functions
├── tests/
│   ├── test_data.py
│   ├── test_strategy.py
│   ├── test_ml.py
│   ├── test_execution.py
│   └── test_backtest.py
├── notebooks/
│   └── exploration.ipynb      # Data exploration & model prototyping
├── models/                    # Saved trained models
├── data/                      # Local data cache
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── setup.py
├── .env.example               # Environment variable template
├── .gitignore
└── README.md
```

---

## Configuration Example (`config/settings.yaml`)

```yaml
trading:
  mode: paper                  # paper | live | backtest
  exchange: binance
  symbols:
    - BTC/USDT
    - ETH/USDT
  timeframe: 1h
  initial_capital: 10000

risk:
  max_position_pct: 5          # Max 5% of portfolio per trade
  stop_loss_pct: 2             # 2% stop loss
  take_profit_pct: 6           # 6% take profit
  max_drawdown_pct: 15         # Pause trading at 15% drawdown
  daily_loss_limit: 500        # Max $500 loss per day
  max_open_positions: 3

ml:
  model_type: lstm             # lstm | transformer
  lookback_window: 60          # Number of candles for prediction
  retrain_interval: 7d         # Retrain every 7 days
  confidence_threshold: 0.65   # Minimum confidence to act on signal

alerts:
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}
  discord:
    enabled: false
    webhook_url: ${DISCORD_WEBHOOK}

database:
  url: sqlite:///data/trading.db
```

---

## Key Workflows

### Backtesting Flow
1. Load historical OHLCV data for selected symbol & timeframe
2. Compute technical indicators and ML features
3. Run strategy logic candle-by-candle, generating signals
4. Simulate order execution with slippage & fees
5. Calculate performance metrics: total return, Sharpe ratio, max drawdown, win rate, profit factor
6. Output report with equity curve chart

### Live/Paper Trading Flow
1. On each new candle close → fetch latest market data
2. Update technical indicators and run ML inference
3. Strategy engine produces signal (BUY/SELL/HOLD)
4. Risk manager validates signal against portfolio rules
5. If approved → execution engine places order
6. Portfolio tracker updates positions and P&L
7. Dashboard refreshes; alerts fire if conditions met

---

## Safety & Best Practices

- **Never hardcode API keys** — use environment variables or a secrets manager.
- **Rate limiting** — respect exchange API rate limits with exponential backoff.
- **Graceful shutdown** — handle SIGINT/SIGTERM to close open positions or cancel pending orders safely.
- **Logging** — structured logging at every decision point (signal generated, order placed, risk check passed/failed).
- **Idempotency** — ensure order placement is idempotent to avoid duplicate trades on retries.
- **Circuit breaker** — automatically halt trading if unusual market conditions are detected (flash crash, extreme volatility).
- **Audit trail** — log every trade with timestamp, signal reason, model confidence, and execution price for post-analysis.

---

## Getting Started (Implementation Order)

1. **Phase 1 — Foundation**: Project setup, config system, data fetcher, database, basic logging
2. **Phase 2 — Indicators & Strategy**: Technical indicators, base strategy class, simple trend-following strategy
3. **Phase 3 — Backtesting**: Backtesting engine with performance metrics and equity curve
4. **Phase 4 — ML Models**: LSTM/Transformer price prediction, training pipeline, model registry
5. **Phase 5 — Execution**: Paper trading mode, order management, portfolio tracking
6. **Phase 6 — Dashboard & Alerts**: Streamlit dashboard, Telegram/Discord notifications
7. **Phase 7 — Live Trading**: Live execution with full risk management and circuit breakers
8. **Phase 8 — Polish**: Docker setup, tests, documentation, CI/CD

---

## Disclaimer

> This bot is for **educational and research purposes**. Trading cryptocurrencies and stocks involves significant financial risk. Always start with paper trading, thoroughly backtest strategies, and never trade with money you cannot afford to lose. Past performance does not guarantee future results.
