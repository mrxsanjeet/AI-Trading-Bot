"""Backtesting engine — run strategies against historical data."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.execution.portfolio import Portfolio
from src.strategy.base import BaseStrategy, Signal
from src.strategy.risk import RiskManager
from src.utils.logger import get_logger

logger = get_logger("execution.backtest")


class BacktestEngine:
    """Run strategies against historical data with realistic simulation."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        slippage_pct: float = 0.0005,
        risk_config: dict | None = None,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.risk_config = risk_config or {
            "max_position_pct": 5,
            "stop_loss_pct": 2,
            "take_profit_pct": 6,
            "max_drawdown_pct": 15,
            "daily_loss_limit": 500,
            "max_open_positions": 3,
        }

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str,
    ) -> dict:
        """Run a backtest on historical data.

        Args:
            df: DataFrame with OHLCV + technical indicators.
            strategy: The strategy to test.
            symbol: Trading pair symbol.

        Returns:
            dict with performance metrics and equity curve.
        """
        portfolio = Portfolio(self.initial_capital)
        risk_manager = RiskManager(self.risk_config)

        equity_curve = []
        signals_log = []
        trades_log = []

        logger.info(
            f"Starting backtest: {strategy.name} on {symbol} | "
            f"{len(df)} candles | Capital: ${self.initial_capital:,.2f}"
        )

        for i in range(50, len(df)):  # Start after enough data for indicators
            current_slice = df.iloc[:i + 1]
            current = df.iloc[i]
            price = current["close"]
            timestamp = df.index[i]
            current_prices = {symbol: price}

            # Check stop-loss / take-profit
            closed = portfolio.check_stop_loss_take_profit(current_prices)
            for trade in closed:
                risk_manager.record_trade_result(trade["pnl"])
                trades_log.append(trade)

            # Generate signal
            signal = strategy.generate_signal(current_slice, symbol)
            signals_log.append({
                "timestamp": timestamp,
                "signal": signal.signal.value,
                "confidence": signal.confidence,
                "reason": signal.reason,
                "price": price,
            })

            if signal.signal == Signal.HOLD:
                equity_curve.append({
                    "timestamp": timestamp,
                    "value": portfolio.total_value(current_prices),
                })
                continue

            # Risk check
            risk_result = risk_manager.evaluate(
                signal=signal,
                portfolio_value=portfolio.total_value(current_prices),
                cash_balance=portfolio.cash_balance,
                open_positions=len(portfolio.positions),
                current_drawdown_pct=portfolio.current_drawdown_pct(current_prices),
            )

            if risk_result["approved"]:
                if signal.signal == Signal.BUY and symbol not in portfolio.positions:
                    # Apply slippage
                    fill_price = price * (1 + self.slippage_pct)
                    position_size = risk_result["position_size"]
                    quantity = position_size / fill_price

                    position = portfolio.open_position(
                        symbol=symbol,
                        side="long",
                        price=fill_price,
                        amount=quantity,
                        stop_loss=risk_result.get("stop_loss"),
                        take_profit=risk_result.get("take_profit"),
                        fee_rate=self.fee_rate,
                    )
                    if position:
                        trades_log.append({
                            "timestamp": timestamp,
                            "side": "BUY",
                            "price": fill_price,
                            "quantity": quantity,
                            "reason": signal.reason,
                        })

                elif signal.signal == Signal.SELL and symbol in portfolio.positions:
                    fill_price = price * (1 - self.slippage_pct)
                    result = portfolio.close_position(
                        symbol=symbol,
                        price=fill_price,
                        fee_rate=self.fee_rate,
                        reason=signal.reason,
                    )
                    if result:
                        risk_manager.record_trade_result(result["pnl"])
                        trades_log.append(result)

            equity_curve.append({
                "timestamp": timestamp,
                "value": portfolio.total_value(current_prices),
            })

        # Close any remaining positions at last price
        last_price = df["close"].iloc[-1]
        for sym in list(portfolio.positions.keys()):
            result = portfolio.close_position(sym, last_price, reason="backtest_end")
            if result:
                trades_log.append(result)

        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)

        metrics = self._calculate_metrics(equity_df, trades_log, portfolio)

        logger.info(
            f"Backtest complete: {strategy.name} | "
            f"Return: {metrics['total_return_pct']:.2f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"Win Rate: {metrics['win_rate']:.1f}% | "
            f"Max DD: {metrics['max_drawdown_pct']:.2f}%"
        )

        return {
            "strategy": strategy.name,
            "symbol": symbol,
            "metrics": metrics,
            "equity_curve": equity_df,
            "trades": trades_log,
            "signals": signals_log,
        }

    def _calculate_metrics(
        self, equity_df: pd.DataFrame, trades: list, portfolio: Portfolio
    ) -> dict:
        """Calculate comprehensive backtest performance metrics."""
        if equity_df.empty:
            return self._empty_metrics()

        final_value = equity_df["value"].iloc[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100

        # Returns for Sharpe calculation
        returns = equity_df["value"].pct_change().dropna()
        sharpe_ratio = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized

        # Max drawdown
        cummax = equity_df["value"].cummax()
        drawdown = (equity_df["value"] - cummax) / cummax
        max_drawdown = abs(drawdown.min()) * 100

        # Trade stats
        closed_trades = [t for t in trades if "pnl" in t]
        winning = [t for t in closed_trades if t["pnl"] > 0]
        losing = [t for t in closed_trades if t["pnl"] < 0]

        win_rate = len(winning) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = sum(t["pnl"] for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t["pnl"] for t in losing) / len(losing) if losing else 0
        total_wins = sum(t["pnl"] for t in winning)
        total_losses = abs(sum(t["pnl"] for t in losing)) if losing else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "total_trades": len(closed_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_fees": sum(t.get("fee", 0) for t in trades),
        }

    def _empty_metrics(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "final_value": self.initial_capital,
            "total_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "total_fees": 0,
        }
