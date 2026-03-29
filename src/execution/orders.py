"""Order placement and management via exchange API."""

from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("execution.orders")


class OrderManager:
    """Manages order creation, tracking, and cancellation."""

    def __init__(self, exchange=None):
        self.exchange = exchange
        self.pending_orders: dict[str, dict] = {}
        self.filled_orders: list[dict] = []

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> dict:
        """Create a market order.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT').
            side: 'buy' or 'sell'.
            amount: Quantity to trade (in base currency).
            price: Current market price (for logging/tracking).
        """
        order = {
            "id": f"order_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "symbol": symbol,
            "type": "market",
            "side": side,
            "amount": amount,
            "price": price,
            "total": amount * price,
            "status": "filled",
            "timestamp": datetime.now(timezone.utc),
        }

        if self.exchange:
            try:
                result = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=amount,
                )
                order["id"] = result["id"]
                order["price"] = result.get("average", result.get("price", price))
                order["status"] = result.get("status", "filled")
                logger.info(f"Market order executed: {side.upper()} {amount} {symbol} @ {order['price']}")
            except Exception as e:
                order["status"] = "failed"
                order["error"] = str(e)
                logger.error(f"Market order failed: {e}")
        else:
            logger.info(f"Market order (simulated): {side.upper()} {amount} {symbol} @ {price}")

        self.filled_orders.append(order)
        return order

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> dict:
        """Create a limit order."""
        order = {
            "id": f"limit_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "symbol": symbol,
            "type": "limit",
            "side": side,
            "amount": amount,
            "price": price,
            "total": amount * price,
            "status": "pending",
            "timestamp": datetime.now(timezone.utc),
        }

        if self.exchange:
            try:
                result = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=amount,
                    price=price,
                )
                order["id"] = result["id"]
                order["status"] = result.get("status", "open")
            except Exception as e:
                order["status"] = "failed"
                order["error"] = str(e)
                logger.error(f"Limit order failed: {e}")
        else:
            logger.info(f"Limit order (simulated): {side.upper()} {amount} {symbol} @ {price}")

        self.pending_orders[order["id"]] = order
        return order

    def create_stop_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        stop_price: float,
    ) -> dict:
        """Create a stop-limit order."""
        order = {
            "id": f"stop_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "symbol": symbol,
            "type": "stop_limit",
            "side": side,
            "amount": amount,
            "price": price,
            "stop_price": stop_price,
            "total": amount * price,
            "status": "pending",
            "timestamp": datetime.now(timezone.utc),
        }

        if self.exchange:
            try:
                result = self.exchange.create_order(
                    symbol=symbol,
                    type="stop_limit" if self.exchange.has.get("createStopLimitOrder") else "limit",
                    side=side,
                    amount=amount,
                    price=price,
                    params={"stopPrice": stop_price},
                )
                order["id"] = result["id"]
                order["status"] = result.get("status", "open")
            except Exception as e:
                order["status"] = "failed"
                order["error"] = str(e)
                logger.error(f"Stop-limit order failed: {e}")

        self.pending_orders[order["id"]] = order
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a pending order."""
        if self.exchange:
            try:
                self.exchange.cancel_order(order_id, symbol)
                logger.info(f"Order cancelled: {order_id}")
            except Exception as e:
                logger.error(f"Cancel failed for {order_id}: {e}")
                return False

        if order_id in self.pending_orders:
            self.pending_orders[order_id]["status"] = "cancelled"
            del self.pending_orders[order_id]
        return True

    def get_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """Get all open/pending orders."""
        orders = list(self.pending_orders.values())
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]
        return orders

    def calculate_fee(self, total: float, fee_rate: float = 0.001) -> float:
        """Calculate trading fee."""
        return total * fee_rate
