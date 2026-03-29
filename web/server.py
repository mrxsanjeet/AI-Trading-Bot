"""FastAPI web server for AI Trading Bot — access from anywhere via browser."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaner import DataCleaner
from src.data.features import FeatureEngineer
from src.data.storage import DataStorage
from src.execution.backtest import BacktestEngine
from src.execution.paper import PaperTrader
from src.execution.portfolio import Portfolio
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.risk import RiskManager
from src.strategy.signals import SignalAggregator
from src.strategy.trend import TrendFollowingStrategy
from src.utils.helpers import load_config, load_strategies_config

app = FastAPI(title="AI Trading Bot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
config = load_config()
strategies_config = load_strategies_config()
cleaner = DataCleaner()
feature_eng = FeatureEngineer()
storage = DataStorage(config["database"]["url"])

# Paper trading state
portfolio = Portfolio(initial_capital=config["trading"]["initial_capital"])
risk_manager = RiskManager(config["risk"])
strat_config = strategies_config.get("strategies", {})
strategies = [
    TrendFollowingStrategy(strat_config.get("trend_following", {})),
    MeanReversionStrategy(strat_config.get("mean_reversion", {})),
    MomentumStrategy(strat_config.get("momentum", {})),
]
aggregator = SignalAggregator(strategies, min_agreement=1)
paper_trader = PaperTrader(portfolio, risk_manager, storage, fee_rate=0.001)

# Indian stocks list
INDIAN_STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "SBIN.NS": "State Bank of India",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Limited",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "WIPRO.NS": "Wipro",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharma",
    "TITAN.NS": "Titan Company",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ADANIENT.NS": "Adani Enterprises",
}


def _fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch stock data via yfinance."""
    from src.data.fetcher import IndianStockFetcher
    df = IndianStockFetcher.fetch_ohlcv(symbol, "1d", period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    df = cleaner.clean(df)
    df = feature_eng.add_all_indicators(df)
    return df


def _df_to_json(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    """Convert DataFrame to JSON-serializable list."""
    if columns:
        df = df[columns]
    records = []
    for ts, row in df.iterrows():
        rec = {"timestamp": ts.isoformat()}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            else:
                rec[col] = float(val)
        records.append(rec)
    return records


# ==================== API ENDPOINTS ====================


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard."""
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/status")
async def get_status():
    """Get bot status and configuration."""
    return {
        "market": config["trading"].get("market", "indian"),
        "exchange": config["trading"].get("exchange", "NSE"),
        "mode": config["trading"]["mode"],
        "symbols": config["trading"]["symbols"],
        "timeframe": config["trading"]["timeframe"],
        "initial_capital": config["trading"]["initial_capital"],
        "risk": config["risk"],
    }


@app.get("/api/stocks")
async def list_stocks():
    """List available Indian stocks."""
    return {"stocks": INDIAN_STOCKS}


@app.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str, period: str = "1y"):
    """Fetch OHLCV + indicators for a stock."""
    try:
        df = _fetch_stock_data(symbol, period)
        ohlcv = _df_to_json(df, ["open", "high", "low", "close", "volume"])
        indicators = _df_to_json(df, ["ema_12", "ema_26", "rsi", "macd", "macd_signal",
                                       "macd_histogram", "bb_upper", "bb_middle", "bb_lower"])
        return {
            "symbol": symbol,
            "name": INDIAN_STOCKS.get(symbol, symbol),
            "candles": len(ohlcv),
            "ohlcv": ohlcv,
            "indicators": indicators,
            "last_price": ohlcv[-1]["close"] if ohlcv else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signal/{symbol}")
async def get_signal(symbol: str):
    """Generate trading signal for a stock."""
    try:
        df = _fetch_stock_data(symbol)
        result = aggregator.generate_signals(df, symbol)
        agg = result["aggregated_signal"]
        individual = {
            name: {
                "signal": sig.signal.value,
                "confidence": round(sig.confidence, 3),
                "reason": sig.reason,
            }
            for name, sig in result["individual_signals"].items()
        }
        return {
            "symbol": symbol,
            "price": round(df["close"].iloc[-1], 2),
            "signal": agg.signal.value,
            "confidence": round(agg.confidence, 3),
            "reason": agg.reason,
            "stop_loss": round(agg.stop_loss, 2) if agg.stop_loss else None,
            "take_profit": round(agg.take_profit, 2) if agg.take_profit else None,
            "strategies": individual,
            "indicators": {
                "rsi": round(float(df["rsi"].iloc[-1]), 2) if not pd.isna(df["rsi"].iloc[-1]) else None,
                "macd": round(float(df["macd"].iloc[-1]), 4) if not pd.isna(df["macd"].iloc[-1]) else None,
                "ema_12": round(float(df["ema_12"].iloc[-1]), 2) if not pd.isna(df["ema_12"].iloc[-1]) else None,
                "ema_26": round(float(df["ema_26"].iloc[-1]), 2) if not pd.isna(df["ema_26"].iloc[-1]) else None,
                "bb_upper": round(float(df["bb_upper"].iloc[-1]), 2) if not pd.isna(df["bb_upper"].iloc[-1]) else None,
                "bb_lower": round(float(df["bb_lower"].iloc[-1]), 2) if not pd.isna(df["bb_lower"].iloc[-1]) else None,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BacktestRequest(BaseModel):
    symbol: str = "RELIANCE.NS"
    strategy: str = "trend"
    capital: float = 10000
    period: str = "2y"
    fee_rate: float = 0.001
    slippage: float = 0.0005


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    """Run a backtest on historical data."""
    strategy_map = {
        "trend": TrendFollowingStrategy(strat_config.get("trend_following", {})),
        "mean_reversion": MeanReversionStrategy(strat_config.get("mean_reversion", {})),
        "momentum": MomentumStrategy(strat_config.get("momentum", {})),
    }

    if req.strategy not in strategy_map:
        raise HTTPException(400, f"Unknown strategy: {req.strategy}")

    try:
        df = _fetch_stock_data(req.symbol, req.period)

        engine = BacktestEngine(
            initial_capital=req.capital,
            fee_rate=req.fee_rate,
            slippage_pct=req.slippage,
            risk_config=config["risk"],
        )
        result = engine.run(df, strategy_map[req.strategy], req.symbol)
        metrics = result["metrics"]

        # Equity curve
        eq = result.get("equity_curve", pd.DataFrame())
        equity = []
        if not eq.empty:
            for ts, row in eq.iterrows():
                equity.append({"timestamp": ts.isoformat(), "value": round(row["value"], 2)})

        # Trades
        trades = []
        for t in result.get("trades", []):
            trade = {}
            for k, v in t.items():
                if isinstance(v, (datetime, pd.Timestamp)):
                    trade[k] = v.isoformat()
                elif isinstance(v, float):
                    trade[k] = round(v, 2)
                else:
                    trade[k] = v
            trades.append(trade)

        return {
            "symbol": req.symbol,
            "strategy": req.strategy,
            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            "equity_curve": equity,
            "trades": trades,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio")
async def get_portfolio():
    """Get current paper trading portfolio status."""
    # Get latest prices for open positions
    prices = {}
    for sym in portfolio.positions:
        try:
            from src.data.fetcher import IndianStockFetcher
            ticker = IndianStockFetcher.fetch_ticker(sym)
            prices[sym] = ticker["last"]
        except Exception:
            pos = portfolio.positions[sym]
            prices[sym] = pos.entry_price

    stats = portfolio.get_stats(prices)
    positions = {}
    for sym, pos in portfolio.positions.items():
        curr = prices.get(sym, pos.entry_price)
        positions[sym] = {
            **pos.to_dict(),
            "current_price": curr,
            "unrealized_pnl": round(pos.unrealized_pnl(curr), 2),
            "unrealized_pnl_pct": round(pos.unrealized_pnl_pct(curr), 2),
        }

    return {
        "portfolio": {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()},
        "positions": positions,
        "risk_status": risk_manager.get_status(),
    }


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get trade history."""
    trades = storage.get_trades(mode="paper", limit=limit)
    for t in trades:
        for k, v in t.items():
            if isinstance(v, datetime):
                t[k] = v.isoformat()
    return {"trades": trades}


@app.post("/api/paper/execute/{symbol}")
async def paper_execute(symbol: str):
    """Generate signal and execute paper trade for a symbol."""
    try:
        df = _fetch_stock_data(symbol)
        result = aggregator.generate_signals(df, symbol)
        signal = result["aggregated_signal"]
        price = df["close"].iloc[-1]
        current_prices = {symbol: price}

        trade = paper_trader.execute_signal(signal, current_prices)

        return {
            "signal": signal.signal.value,
            "confidence": round(signal.confidence, 3),
            "reason": signal.reason,
            "price": round(price, 2),
            "trade_executed": trade is not None,
            "trade": trade if trade else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio/reset")
async def reset_portfolio():
    """Reset paper trading portfolio."""
    global portfolio, risk_manager, paper_trader
    capital = config["trading"]["initial_capital"]
    portfolio = Portfolio(initial_capital=capital)
    risk_manager = RiskManager(config["risk"])
    paper_trader = PaperTrader(portfolio, risk_manager, storage, fee_rate=0.001)
    return {"message": "Portfolio reset", "capital": capital}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("  AI Trading Bot — Web Dashboard")
    print("=" * 55)
    print("  Open in browser: http://localhost:8000")
    print("  API docs:        http://localhost:8000/docs")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
