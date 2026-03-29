# AI Trading Bot

An AI-powered cryptocurrency and stock trading bot with machine learning price prediction, multiple trading strategies, backtesting, paper trading, and live trading support.

## Features

- **ML Price Prediction** — LSTM and Transformer models trained on historical data
- **Multiple Strategies** — Trend following, mean reversion, momentum, ML-driven, and hybrid ensemble
- **Risk Management** — Position sizing, stop-loss/take-profit, max drawdown limits, cooldown periods
- **Backtesting Engine** — Test strategies on historical data with realistic slippage and fees
- **Paper Trading** — Simulate trades with live market data (no real money)
- **Live Trading** — Execute real trades on exchanges via CCXT
- **Streamlit Dashboard** — Real-time portfolio monitoring, charts, and trade history
- **Alerts** — Telegram and Discord notifications for trades and risk events
- **Sentiment Analysis** — NLP-based news sentiment scoring

## Quick Start

### 1. Install

```bash
git clone https://github.com/mrxsanjeet/AI-Trading-Bot.git
cd AI-Trading-Bot
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
# Edit config/settings.yaml for trading parameters
```

### 3. Run

```bash
# Check status
python -m src.main status

# Run backtest
python -m src.main backtest -s BTC/USDT -t 1h --strategy trend

# Paper trading
python -m src.main paper -s BTC/USDT -t 1h

# Train ML model
python -m src.main train -s BTC/USDT -t 1h --epochs 50

# Launch dashboard
python -m src.main dashboard
```

## Architecture

```
src/
├── data/          # Market data fetching, cleaning, indicators, storage
├── strategy/      # Trading strategies, signal generation, risk management
├── ml/            # LSTM/Transformer models, training, prediction, registry
├── execution/     # Order management, portfolio, paper/live trading, backtesting
├── dashboard/     # Streamlit UI, alerts (Telegram/Discord)
└── utils/         # Logging, config, helpers
```

## Docker

```bash
docker-compose up -d
```

- **Trading bot** runs on the `trading-bot` service
- **Dashboard** available at `http://localhost:8501`

## Testing

```bash
pytest tests/ -v
```

## Configuration

Edit `config/settings.yaml` for trading parameters and `config/strategies.yaml` for strategy-specific settings. See `PROMPT.md` for the full specification.

## Disclaimer

This bot is for **educational and research purposes only**. Trading involves significant financial risk. Always start with paper trading and never trade with money you cannot afford to lose.
