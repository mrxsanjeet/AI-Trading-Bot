"""Notification system — Telegram, Discord, and email alerts."""

import json
from datetime import datetime, timezone

import aiohttp
import requests

from src.utils.logger import get_logger

logger = get_logger("dashboard.alerts")


class AlertManager:
    """Sends trading alerts via Telegram, Discord, and email."""

    def __init__(self, config: dict):
        self.config = config
        self.telegram_enabled = config.get("telegram", {}).get("enabled", False)
        self.discord_enabled = config.get("discord", {}).get("enabled", False)

        if self.telegram_enabled:
            self.telegram_token = config["telegram"]["bot_token"]
            self.telegram_chat_id = config["telegram"]["chat_id"]
            logger.info("Telegram alerts enabled")

        if self.discord_enabled:
            self.discord_webhook = config["discord"]["webhook_url"]
            logger.info("Discord alerts enabled")

    def send_trade_alert(self, trade: dict) -> None:
        """Send alert for a trade execution."""
        side = trade.get("side", "UNKNOWN")
        symbol = trade.get("symbol", "")
        price = trade.get("price", 0)
        quantity = trade.get("quantity", 0)
        total = trade.get("total", price * quantity)
        strategy = trade.get("strategy", "")
        confidence = trade.get("confidence", 0)
        pnl = trade.get("pnl")

        if pnl is not None:
            message = (
                f"{'🟢' if pnl > 0 else '🔴'} **TRADE CLOSED**\n"
                f"**{symbol}** | {side}\n"
                f"Price: ${price:,.2f}\n"
                f"P&L: ${pnl:,.2f} ({trade.get('pnl_pct', 0):+.2f}%)\n"
                f"Reason: {trade.get('reason', 'N/A')}\n"
                f"Strategy: {strategy}"
            )
        else:
            message = (
                f"{'📈' if side == 'BUY' else '📉'} **{side} ORDER**\n"
                f"**{symbol}**\n"
                f"Price: ${price:,.2f}\n"
                f"Amount: {quantity:.6f}\n"
                f"Total: ${total:,.2f}\n"
                f"Strategy: {strategy}\n"
                f"Confidence: {confidence:.1%}"
            )

        self._send(message)

    def send_risk_alert(self, message: str) -> None:
        """Send alert for risk management events."""
        alert = f"⚠️ **RISK ALERT**\n{message}"
        self._send(alert)

    def send_status_update(self, stats: dict) -> None:
        """Send periodic portfolio status update."""
        message = (
            f"📊 **Portfolio Update**\n"
            f"Value: ${stats.get('total_value', 0):,.2f}\n"
            f"Return: {stats.get('total_return_pct', 0):+.2f}%\n"
            f"Open Positions: {stats.get('open_positions', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1f}%\n"
            f"Drawdown: {stats.get('max_drawdown_pct', 0):.2f}%"
        )
        self._send(message)

    def send_error_alert(self, error: str) -> None:
        """Send alert for system errors."""
        alert = f"🚨 **SYSTEM ERROR**\n{error}"
        self._send(alert)

    def _send(self, message: str) -> None:
        """Send message to all enabled channels."""
        if self.telegram_enabled:
            self._send_telegram(message)
        if self.discord_enabled:
            self._send_discord(message)

    def _send_telegram(self, message: str) -> None:
        """Send message via Telegram Bot API."""
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Telegram alert sent")
        except requests.RequestException as e:
            logger.error(f"Telegram alert failed: {e}")

    def _send_discord(self, message: str) -> None:
        """Send message via Discord webhook."""
        payload = {"content": message}

        try:
            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Discord alert sent")
        except requests.RequestException as e:
            logger.error(f"Discord alert failed: {e}")
