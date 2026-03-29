"""Risk management rules and position sizing."""

from datetime import datetime, timedelta, timezone

from src.strategy.base import Signal, TradeSignal
from src.utils.logger import get_logger

logger = get_logger("strategy.risk")


class RiskManager:
    """Enforces risk management rules before trade execution."""

    def __init__(self, config: dict):
        self.max_position_pct = config.get("max_position_pct", 5)
        self.stop_loss_pct = config.get("stop_loss_pct", 2)
        self.take_profit_pct = config.get("take_profit_pct", 6)
        self.max_drawdown_pct = config.get("max_drawdown_pct", 15)
        self.daily_loss_limit = config.get("daily_loss_limit", 500)
        self.max_open_positions = config.get("max_open_positions", 3)
        self.cooldown_periods = config.get("cooldown_periods", 3)

        self._consecutive_losses = 0
        self._daily_loss = 0.0
        self._daily_loss_reset_date = None
        self._peak_portfolio_value = 0.0
        self._trading_paused = False
        self._pause_reason = ""

        logger.info(
            f"RiskManager initialized: max_position={self.max_position_pct}%, "
            f"stop_loss={self.stop_loss_pct}%, max_drawdown={self.max_drawdown_pct}%"
        )

    def evaluate(
        self,
        signal: TradeSignal,
        portfolio_value: float,
        cash_balance: float,
        open_positions: int,
        current_drawdown_pct: float,
    ) -> dict:
        """Evaluate a trade signal against risk rules.

        Returns:
            dict with: approved (bool), reason (str), position_size (float),
            stop_loss (float), take_profit (float).
        """
        # Reset daily loss counter at start of new day
        today = datetime.now(timezone.utc).date()
        if self._daily_loss_reset_date != today:
            self._daily_loss = 0.0
            self._daily_loss_reset_date = today

        # Update peak portfolio value
        if portfolio_value > self._peak_portfolio_value:
            self._peak_portfolio_value = portfolio_value

        # HOLD signals always pass
        if signal.signal == Signal.HOLD:
            return {"approved": True, "reason": "HOLD signal — no action", "position_size": 0}

        # Check if trading is paused
        if self._trading_paused:
            logger.warning(f"Trading paused: {self._pause_reason}")
            return {"approved": False, "reason": f"Trading paused: {self._pause_reason}", "position_size": 0}

        # Check cooldown after consecutive losses
        if self._consecutive_losses >= self.cooldown_periods:
            logger.warning(f"Cooldown active: {self._consecutive_losses} consecutive losses")
            return {
                "approved": False,
                "reason": f"Cooldown: {self._consecutive_losses} consecutive losses",
                "position_size": 0,
            }

        # Check max drawdown
        if current_drawdown_pct >= self.max_drawdown_pct:
            self._trading_paused = True
            self._pause_reason = f"Max drawdown reached: {current_drawdown_pct:.1f}%"
            logger.critical(self._pause_reason)
            return {"approved": False, "reason": self._pause_reason, "position_size": 0}

        # Check daily loss limit
        if self._daily_loss >= self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: ${self._daily_loss:.2f}")
            return {
                "approved": False,
                "reason": f"Daily loss limit reached: ${self._daily_loss:.2f}",
                "position_size": 0,
            }

        # Check max open positions (only for BUY signals)
        if signal.signal == Signal.BUY and open_positions >= self.max_open_positions:
            return {
                "approved": False,
                "reason": f"Max open positions reached: {open_positions}/{self.max_open_positions}",
                "position_size": 0,
            }

        # Calculate position size
        position_size = self.calculate_position_size(portfolio_value, signal.price)

        # Set stop-loss and take-profit
        stop_loss = signal.stop_loss or signal.price * (1 - self.stop_loss_pct / 100)
        take_profit = signal.take_profit or signal.price * (1 + self.take_profit_pct / 100)

        logger.info(
            f"Risk approved: {signal.signal.value} {signal.symbol} | "
            f"Size: ${position_size:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}"
        )

        return {
            "approved": True,
            "reason": "All risk checks passed",
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def calculate_position_size(self, portfolio_value: float, price: float) -> float:
        """Calculate the position size based on portfolio percentage."""
        max_amount = portfolio_value * (self.max_position_pct / 100)
        return max_amount

    def record_trade_result(self, pnl: float) -> None:
        """Record the P&L of a completed trade for risk tracking."""
        if pnl < 0:
            self._consecutive_losses += 1
            self._daily_loss += abs(pnl)
            logger.debug(
                f"Loss recorded: ${pnl:.2f} | "
                f"Consecutive losses: {self._consecutive_losses} | "
                f"Daily loss: ${self._daily_loss:.2f}"
            )
        else:
            self._consecutive_losses = 0
            logger.debug(f"Win recorded: ${pnl:.2f} | Consecutive losses reset")

    def reset_pause(self) -> None:
        """Manually reset the trading pause state."""
        self._trading_paused = False
        self._pause_reason = ""
        self._consecutive_losses = 0
        logger.info("Trading pause reset")

    def get_status(self) -> dict:
        """Get current risk manager status."""
        return {
            "trading_paused": self._trading_paused,
            "pause_reason": self._pause_reason,
            "consecutive_losses": self._consecutive_losses,
            "daily_loss": self._daily_loss,
            "peak_portfolio_value": self._peak_portfolio_value,
        }
