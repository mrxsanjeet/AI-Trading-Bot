"""Centralized logging configuration for the trading bot."""

import logging
import logging.config
import os
from pathlib import Path

import yaml


def setup_logging(config_path: str = "config/logging.yaml", default_level: int = logging.INFO) -> None:
    """Set up logging from YAML config file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        try:
            logging.config.dictConfig(config)
        except (ValueError, ImportError):
            # Fall back if colorlog is not installed
            logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the trading_bot namespace."""
    return logging.getLogger(f"trading_bot.{name}")
