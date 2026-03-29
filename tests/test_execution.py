"""Tests for the execution engine modules."""

import pytest

from src.execution.orders import OrderManager
from src.execution.portfolio import Portfolio, Position


class TestPosition:
    def test_long_position_pnl(self):
        pos = Position("BTC/USDT", "long", 50000, 0.1)
        assert pos.unrealized_pnl(55000) == pytest.approx(500.0)
        assert pos.unrealized_pnl(45000) == pytest.approx(-500.0)

    def test_short_position_pnl(self):
        pos = Position("BTC/USDT", "short", 50000, 0.1)
        assert pos.unrealized_pnl(45000) == pytest.approx(500.0)
        assert pos.unrealized_pnl(55000) == pytest.approx(-500.0)

    def test_stop_loss_trigger(self):
        pos = Position("BTC/USDT", "long", 50000, 0.1, stop_loss=48000)
        assert not pos.should_stop_loss(49000)
        assert pos.should_stop_loss(47000)

    def test_take_profit_trigger(self):
        pos = Position("BTC/USDT", "long", 50000, 0.1, take_profit=55000)
        assert not pos.should_take_profit(54000)
        assert pos.should_take_profit(56000)


class TestPortfolio:
    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=10000.0)

    def test_initial_state(self, portfolio):
        assert portfolio.cash_balance == 10000.0
        assert len(portfolio.positions) == 0
        assert portfolio.realized_pnl == 0.0

    def test_open_position(self, portfolio):
        pos = portfolio.open_position("BTC/USDT", "long", 50000, 0.1, fee_rate=0.001)
        assert pos is not None
        assert "BTC/USDT" in portfolio.positions
        assert portfolio.cash_balance < 10000.0  # Deducted cost + fee

    def test_insufficient_funds(self, portfolio):
        pos = portfolio.open_position("BTC/USDT", "long", 50000, 1.0, fee_rate=0.001)
        assert pos is None  # $50k > $10k

    def test_close_position_profit(self, portfolio):
        portfolio.open_position("BTC/USDT", "long", 50000, 0.1, fee_rate=0.001)
        result = portfolio.close_position("BTC/USDT", 55000, fee_rate=0.001)
        assert result is not None
        assert result["pnl"] > 0
        assert "BTC/USDT" not in portfolio.positions

    def test_close_position_loss(self, portfolio):
        portfolio.open_position("BTC/USDT", "long", 50000, 0.1, fee_rate=0.001)
        result = portfolio.close_position("BTC/USDT", 45000, fee_rate=0.001)
        assert result is not None
        assert result["pnl"] < 0

    def test_total_value(self, portfolio):
        portfolio.open_position("BTC/USDT", "long", 50000, 0.1, fee_rate=0.001)
        prices = {"BTC/USDT": 55000}
        total = portfolio.total_value(prices)
        assert total > 0

    def test_stop_loss_take_profit(self, portfolio):
        portfolio.open_position(
            "BTC/USDT", "long", 50000, 0.1,
            stop_loss=48000, take_profit=55000, fee_rate=0.001
        )
        closed = portfolio.check_stop_loss_take_profit({"BTC/USDT": 47000})
        assert len(closed) == 1
        assert closed[0]["reason"] == "stop_loss"

    def test_portfolio_stats(self, portfolio):
        prices = {"BTC/USDT": 50000}
        stats = portfolio.get_stats(prices)
        assert "total_value" in stats
        assert "win_rate" in stats
        assert "total_trades" in stats


class TestOrderManager:
    def test_create_market_order(self):
        om = OrderManager()
        order = om.create_market_order("BTC/USDT", "buy", 0.1, 50000)
        assert order["status"] == "filled"
        assert order["symbol"] == "BTC/USDT"
        assert order["side"] == "buy"

    def test_create_limit_order(self):
        om = OrderManager()
        order = om.create_limit_order("BTC/USDT", "buy", 0.1, 49000)
        assert order["status"] == "pending"
        assert order["price"] == 49000

    def test_cancel_order(self):
        om = OrderManager()
        order = om.create_limit_order("BTC/USDT", "buy", 0.1, 49000)
        result = om.cancel_order(order["id"], "BTC/USDT")
        assert result is True
        assert len(om.get_open_orders()) == 0
