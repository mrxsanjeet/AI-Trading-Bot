"""Shared utility functions for the trading bot."""

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load YAML config with environment variable substitution."""
    load_dotenv()

    with open(config_path) as f:
        content = f.read()

    # Replace ${VAR_NAME} with environment variable values
    def replace_env_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    content = re.sub(r"\$\{(\w+)\}", replace_env_var, content)
    return yaml.safe_load(content)


def load_strategies_config(config_path: str = "config/strategies.yaml") -> dict:
    """Load strategy configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def timestamp_to_datetime(ts: int) -> datetime:
    """Convert millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency string."""
    return f"${value:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a number as percentage string."""
    return f"{value:+.{decimals}f}%"


def calculate_pct_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100
