"""Live trading executor — executes real trades on exchanges."""

from datetime import datetime, timezone

from src.data.storage import DataStorage
from src.execution.orders import OrderManager
from src.execution.portfolio import Portfolio
from src.strategy.base import Signal, TradeSignal
from src.strategy.risk import RiskManager
from src.utils.logger import get_logger

logger = get_logger("execution.live")


class LiveTrader:
    """Executes real trades on an exchange with full safety checks."""

    def __init__(
        self,
        exchange,
        portfolio: Portfolio,
        risk_manager: RiskManager,
        storage: DataStorage,
        fee_rate: float = 0.001,
        dry_run: bool = True,
    ):
        self.exchange = exchange
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.order_manager = OrderManager(exchange=exchange if not dry_run else None)
        self.storage = storage
        self.fee_rate = fee_rate
        self.dry_run = dry_run

        mode = "DRY RUN" if dry_run else "LIVE"
        logger.warning(f"LiveTrader initialized in {mode} mode")

    def execute_signal(self, signal: TradeSignal, current_prices: dict[str, float]) -> dict | None:
        """Execute a trade signal with full safety checks."""
        if signal.signal == Signal.HOLD:
            return None

        # Check stop-loss and take-profit
        closed = self.portfolio.check_stop_loss_take_profit(current_prices)
        for trade in closed:
            self.risk_manager.record_trade_result(trade["pnl"])
            self._save_trade(trade, signal.strategy)
            logger.warning(f"Auto-closed position: {trade['symbol']} | Reason: {trade['reason']}")

        # Risk evaluation
        risk_result = self.risk_manager.evaluate(
            signal=signal,
            portfolio_value=self.portfolio.total_value(current_prices),
            cash_balance=self.portfolio.cash_balance,
            open_positions=len(self.portfolio.positions),
            current_drawdown_pct=self.portfolio.current_drawdown_pct(current_prices),
        )

        if not risk_result["approved"]:
            logger.warning(f"Signal rejected by risk manager: {risk_result['reason']}")
            return None

        # Double-check: verify exchange balance before live execution
        if not self.dry_run:
            try:
                balance = self.exchange.fetch_balance()
                usdt_free = balance.get("USDT", {}).get("free", 0)
                if usdt_free < risk_result["position_size"]:
                    logger.error(
                        f"Insufficient exchange balance: ${usdt_free:.2f} < "
                        f"${risk_result['position_size']:.2f}"
                    )
                    return None
            except Exception as e:
                logger.error(f"Failed to check exchange balance: {e}")
                return None

        # Execute
        if signal.signal == Signal.BUY:
            return self._execute_buy(signal, risk_result, current_prices)
        elif signal.signal == Signal.SELL:
            return self._execute_sell(signal, current_prices)

        return None

    def _execute_buy(
        self, signal: TradeSignal, risk_result: dict, current_prices: dict[str, float]
    ) -> dict | None:
        """Execute a buy order."""
        if signal.symbol in self.portfolio.positions:
            return None

        position_size = risk_result["position_size"]
        quantity = position_size / signal.price

        order = self.order_manager.create_market_order(
            symbol=signal.symbol,
            side="buy",
            amount=quantity,
            price=signal.price,
        )

        if order.get("status") == "failed":
            logger.error(f"Buy order failed: {order.get('error')}")
            return None

        actual_price = order.get("price", signal.price)
        position = self.portfolio.open_position(
            symbol=signal.symbol,
            side="long",
            price=actual_price,
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
            "price": actual_price,
            "quantity": quantity,
            "total": position_size,
            "fee": position_size * self.fee_rate,
            "timestamp": datetime.now(timezone.utc),
            "strategy": signal.strategy,
            "signal_reason": signal.reason,
            "confidence": signal.confidence,
            "mode": "live" if not self.dry_run else "dry_run",
        }

        self._save_trade(trade_record, signal.strategy)
        return trade_record

    def _execute_sell(
        self, signal: TradeSignal, current_prices: dict[str, float]
    ) -> dict | None:
        """Execute a sell order."""
        if signal.symbol not in self.portfolio.positions:
            return None

        position = self.portfolio.positions[signal.symbol]

        order = self.order_manager.create_market_order(
            symbol=signal.symbol,
            side="sell",
            amount=position.quantity,
            price=signal.price,
        )

        if order.get("status") == "failed":
            logger.error(f"Sell order failed: {order.get('error')}")
            return None

        actual_price = order.get("price", signal.price)
        result = self.portfolio.close_position(
            symbol=signal.symbol,
            price=actual_price,
            fee_rate=self.fee_rate,
            reason=signal.reason,
        )

        if result:
            self.risk_manager.record_trade_result(result["pnl"])
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
                "mode": "live" if not self.dry_run else "dry_run",
            }
            self.storage.save_trade(record)
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
