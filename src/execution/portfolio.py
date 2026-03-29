"""Portfolio tracking — balances, positions, and P&L."""

from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger("execution.portfolio")


class Position:
    """Represents an open position."""

    def __init__(self, symbol: str, side: str, entry_price: float, quantity: float,
                 stop_loss: float | None = None, take_profit: float | None = None):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.opened_at = datetime.now(timezone.utc)

    @property
    def entry_value(self) -> float:
        return self.entry_price * self.quantity

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        pnl = self.unrealized_pnl(current_price)
        return (pnl / self.entry_value) * 100

    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop-loss has been triggered."""
        if self.stop_loss is None:
            return False
        if self.side == "long":
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        """Check if take-profit has been triggered."""
        if self.take_profit is None:
            return False
        if self.side == "long":
            return current_price >= self.take_profit
        return current_price <= self.take_profit

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_value": self.entry_value,
            "opened_at": self.opened_at.isoformat(),
        }


class Portfolio:
    """Tracks portfolio state, positions, and performance."""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_history: list[dict] = []
        self.realized_pnl = 0.0
        self._peak_value = initial_capital

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        fee_rate: float = 0.001,
    ) -> Position | None:
        """Open a new position."""
        total_cost = price * amount
        fee = total_cost * fee_rate

        if total_cost + fee > self.cash_balance:
            logger.warning(f"Insufficient funds: need ${total_cost + fee:.2f}, have ${self.cash_balance:.2f}")
            return None

        self.cash_balance -= (total_cost + fee)
        quantity = amount

        position = Position(symbol, side, price, quantity, stop_loss, take_profit)
        self.positions[symbol] = position

        self.trade_history.append({
            "symbol": symbol,
            "side": "BUY" if side == "long" else "SELL",
            "action": "OPEN",
            "price": price,
            "quantity": quantity,
            "total": total_cost,
            "fee": fee,
            "timestamp": datetime.now(timezone.utc),
        })

        logger.info(
            f"Opened {side} position: {symbol} | Qty: {quantity:.6f} | "
            f"Entry: ${price:.2f} | Cost: ${total_cost:.2f} | Fee: ${fee:.2f}"
        )
        return position

    def close_position(
        self, symbol: str, price: float, fee_rate: float = 0.001, reason: str = "manual"
    ) -> dict | None:
        """Close an existing position."""
        if symbol not in self.positions:
            logger.warning(f"No open position for {symbol}")
            return None

        position = self.positions[symbol]
        total_value = price * position.quantity
        fee = total_value * fee_rate
        pnl = position.unrealized_pnl(price) - fee

        self.cash_balance += (total_value - fee)
        self.realized_pnl += pnl

        result = {
            "symbol": symbol,
            "side": "SELL" if position.side == "long" else "BUY",
            "action": "CLOSE",
            "entry_price": position.entry_price,
            "exit_price": price,
            "quantity": position.quantity,
            "total": total_value,
            "fee": fee,
            "pnl": pnl,
            "pnl_pct": position.unrealized_pnl_pct(price),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc),
            "hold_duration": str(datetime.now(timezone.utc) - position.opened_at),
        }

        self.trade_history.append(result)
        del self.positions[symbol]

        logger.info(
            f"Closed {position.side} {symbol} | "
            f"Entry: ${position.entry_price:.2f} -> Exit: ${price:.2f} | "
            f"P&L: ${pnl:.2f} ({position.unrealized_pnl_pct(price):+.2f}%) | "
            f"Reason: {reason}"
        )
        return result

    def check_stop_loss_take_profit(self, prices: dict[str, float]) -> list[dict]:
        """Check all positions for stop-loss or take-profit triggers."""
        closed = []
        symbols_to_close = []

        for symbol, position in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue

            if position.should_stop_loss(price):
                symbols_to_close.append((symbol, price, "stop_loss"))
            elif position.should_take_profit(price):
                symbols_to_close.append((symbol, price, "take_profit"))

        for symbol, price, reason in symbols_to_close:
            result = self.close_position(symbol, price, reason=reason)
            if result:
                closed.append(result)

        return closed

    def total_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value (cash + positions)."""
        positions_value = sum(
            prices.get(symbol, pos.entry_price) * pos.quantity
            for symbol, pos in self.positions.items()
        )
        return self.cash_balance + positions_value

    def total_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Calculate total unrealized P&L across all positions."""
        return sum(
            pos.unrealized_pnl(prices.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )

    def current_drawdown_pct(self, prices: dict[str, float]) -> float:
        """Calculate current drawdown from peak value."""
        current_value = self.total_value(prices)
        if current_value > self._peak_value:
            self._peak_value = current_value
        if self._peak_value == 0:
            return 0.0
        return ((self._peak_value - current_value) / self._peak_value) * 100

    def get_stats(self, prices: dict[str, float]) -> dict:
        """Get portfolio statistics."""
        total_val = self.total_value(prices)
        total_return = ((total_val - self.initial_capital) / self.initial_capital) * 100

        winning_trades = [t for t in self.trade_history if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get("pnl", 0) < 0]
        closed_trades = [t for t in self.trade_history if "pnl" in t]

        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0

        avg_win = (
            sum(t["pnl"] for t in winning_trades) / len(winning_trades)
            if winning_trades else 0
        )
        avg_loss = (
            sum(t["pnl"] for t in losing_trades) / len(losing_trades)
            if losing_trades else 0
        )
        profit_factor = (
            abs(sum(t["pnl"] for t in winning_trades) / sum(t["pnl"] for t in losing_trades))
            if losing_trades and sum(t["pnl"] for t in losing_trades) != 0 else 0
        )

        return {
            "total_value": total_val,
            "cash_balance": self.cash_balance,
            "total_return_pct": total_return,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.total_unrealized_pnl(prices),
            "open_positions": len(self.positions),
            "total_trades": len(closed_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": self.current_drawdown_pct(prices),
        }
