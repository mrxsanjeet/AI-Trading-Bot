"""Market data fetching from exchanges and financial APIs."""

from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger("data.fetcher")


class MarketDataFetcher:
    """Fetches OHLCV market data from crypto exchanges and stock APIs."""

    VALID_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

    def __init__(self, exchange_id: str = "binance", api_key: str = "", api_secret: str = ""):
        self.exchange_id = exchange_id
        self.exchange = self._init_exchange(exchange_id, api_key, api_secret)
        logger.info(f"MarketDataFetcher initialized with exchange: {exchange_id}")

    def _init_exchange(self, exchange_id: str, api_key: str, api_secret: str) -> ccxt.Exchange:
        """Initialize a CCXT exchange instance."""
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        config = {"enableRateLimit": True}
        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret

        return exchange_class(config)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data from the exchange.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT').
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d).
            since: Start datetime for historical data.
            limit: Maximum number of candles to fetch.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {self.VALID_TIMEFRAMES}")

        since_ts = int(since.timestamp() * 1000) if since else None

        logger.info(f"Fetching {limit} {timeframe} candles for {symbol}")
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ts, limit=limit)

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)

        logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
        return df

    def fetch_all_history(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        max_candles: int = 5000,
    ) -> pd.DataFrame:
        """Fetch extended historical data by paginating through API calls."""
        all_data = []
        fetched = 0
        batch_size = 500
        current_since = int(since.timestamp() * 1000) if since else None

        while fetched < max_candles:
            remaining = min(batch_size, max_candles - fetched)
            raw = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=current_since, limit=remaining
            )
            if not raw:
                break

            all_data.extend(raw)
            fetched += len(raw)
            current_since = raw[-1][0] + 1  # Next millisecond after last candle

            if len(raw) < remaining:
                break

        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        df = df[~df.index.duplicated(keep="last")]

        logger.info(f"Fetched {len(df)} total historical candles for {symbol}")
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker data for a symbol."""
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "last": ticker.get("last"),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "volume": ticker.get("baseVolume"),
            "change_pct": ticker.get("percentage"),
            "timestamp": datetime.now(timezone.utc),
        }

    def get_available_symbols(self) -> list[str]:
        """Get list of available trading pairs on the exchange."""
        self.exchange.load_markets()
        return list(self.exchange.symbols)


class StockDataFetcher:
    """Fetches stock market data via yfinance."""

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
    }

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        period: str = "1y",
    ) -> pd.DataFrame:
        """Fetch stock OHLCV data.

        Args:
            symbol: Stock ticker (e.g., 'AAPL').
            timeframe: Data interval.
            period: How far back to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max).
        """
        interval = self.INTERVAL_MAP.get(timeframe, "1d")
        logger.info(f"Fetching stock data for {symbol} ({interval}, {period})")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]

        logger.info(f"Fetched {len(df)} stock candles for {symbol}")
        return df
