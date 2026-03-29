"""Paper trading simulator — executes trades against live data without real money."""

from datetime import datetime, timezone

from src.data.storage import DataStorage
from src.execution.orders import OrderManager
from src.execution.portfolio import Portfolio
from src.strategy.base import Signal, TradeSignal
from src.strategy.risk import RiskManager
from src.utils.logger import get_logger

logger = get_logger("execution.paper")


class PaperTrader:
    """Simulates trading using live market data without real money."""

    def __init__(
        self,
        portfolio: Portfolio,
        risk_manager: RiskManager,
        storage: DataStorage | None = None,
        fee_rate: float = 0.001,
    ):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.order_manager = OrderManager()  # No exchange = simulated
        self.storage = storage
        self.fee_rate = fee_rate

        logger.info(
            f"PaperTrader initialized | Capital: ${portfolio.initial_capital:,.2f} | "
            f"Fee rate: {fee_rate:.1%}"
        )

    def execute_signal(self, signal: TradeSignal, current_prices: dict[str, float]) -> dict | None:
        """Execute a trade signal in paper trading mode.

        Returns:
            Trade result dict or None if not executed.
        """
        if signal.signal == Signal.HOLD:
            return None

        # Check stop-loss and take-profit on existing positions first
        closed = self.portfolio.check_stop_loss_take_profit(current_prices)
        for trade in closed:
            self.risk_manager.record_trade_result(trade["pnl"])
            if self.storage:
                self._save_trade(trade, signal.strategy)

        # Risk evaluation
        risk_result = self.risk_manager.evaluate(
            signal=signal,
            portfolio_value=self.portfolio.total_value(current_prices),
            cash_balance=self.portfolio.cash_balance,
            open_positions=len(self.portfolio.positions),
            current_drawdown_pct=self.portfolio.current_drawdown_pct(current_prices),
        )

        if not risk_result["approved"]:
            logger.info(f"Signal rejected by risk manager: {risk_result['reason']}")
            return None

        # Execute the trade
        if signal.signal == Signal.BUY:
            return self._execute_buy(signal, risk_result, current_prices)
        elif signal.signal == Signal.SELL:
            return self._execute_sell(signal, current_prices)

        return None

    def _execute_buy(
        self, signal: TradeSignal, risk_result: dict, current_prices: dict[str, float]
    ) -> dict | None:
        """Execute a buy order."""
        # Skip if already have position in this symbol
        if signal.symbol in self.portfolio.positions:
            logger.debug(f"Already have position in {signal.symbol}, skipping BUY")
            return None

        position_size = risk_result["position_size"]
        quantity = position_size / signal.price

        # Place simulated market order
        order = self.order_manager.create_market_order(
            symbol=signal.symbol,
            side="buy",
            amount=quantity,
            price=signal.price,
        )

        # Open position in portfolio
        position = self.portfolio.open_position(
            symbol=signal.symbol,
            side="long",
            price=signal.price,
            amount=quantity,
            stop_loss=risk_result.get("stop_loss"),
            take_profit=risk_result.get("take_profit"),
            fee_rate=self.fee_rate,
        )

        if position is None:
            return None

        trade_record = {
            "symbol": signal.symbol,
            "side": "BUY",
            "order_type": "market",
            "price": signal.price,
            "quantity": quantity,
            "total": position_size,
            "fee": position_size * self.fee_rate,
            "timestamp": datetime.now(timezone.utc),
            "strategy": signal.strategy,
            "signal_reason": signal.reason,
            "confidence": signal.confidence,
            "mode": "paper",
        }

        if self.storage:
            self._save_trade(trade_record, signal.strategy)

        return trade_record

    def _execute_sell(
        self, signal: TradeSignal, current_prices: dict[str, float]
    ) -> dict | None:
        """Execute a sell order (close position)."""
        if signal.symbol not in self.portfolio.positions:
            logger.debug(f"No position in {signal.symbol} to sell")
            return None

        result = self.portfolio.close_position(
            symbol=signal.symbol,
            price=signal.price,
            fee_rate=self.fee_rate,
            reason=signal.reason,
        )

        if result:
            self.risk_manager.record_trade_result(result["pnl"])
            if self.storage:
                self._save_trade(result, signal.strategy)

        return result

    def _save_trade(self, trade: dict, strategy: str) -> None:
        """Save trade to database."""
        try:
            record = {
                "symbol": trade.get("symbol", ""),
                "side": trade.get("side", ""),
                "order_type": trade.get("order_type", "market"),
                "price": trade.get("price", trade.get("exit_price", 0)),
                "quantity": trade.get("quantity", 0),
                "total": trade.get("total", 0),
                "fee": trade.get("fee", 0),
                "timestamp": trade.get("timestamp", datetime.now(timezone.utc)),
                "strategy": strategy,
                "signal_reason": trade.get("signal_reason", trade.get("reason", "")),
                "confidence": trade.get("confidence", 0),
                "mode": "paper",
            }
            self.storage.save_trade(record)
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    def get_status(self, current_prices: dict[str, float]) -> dict:
        """Get current paper trading status."""
        stats = self.portfolio.get_stats(current_prices)
        risk_status = self.risk_manager.get_status()
        return {
            "mode": "paper",
            "portfolio": stats,
            "risk": risk_status,
            "positions": {
                sym: pos.to_dict() for sym, pos in self.portfolio.positions.items()
            },
        }
