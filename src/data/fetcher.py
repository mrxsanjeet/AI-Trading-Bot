"""Market data fetching from exchanges and financial APIs."""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger("data.fetcher")


class MarketDataFetcher:
    """Fetches OHLCV market data from crypto exchanges."""

    VALID_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

    def __init__(self, exchange_id: str = "binance", api_key: str = "", api_secret: str = ""):
        self.exchange_id = exchange_id
        self.exchange = None

        # Only initialize CCXT for crypto exchanges
        if exchange_id.lower() not in ("nse", "bse", "indian"):
            try:
                import ccxt
                exchange_class = getattr(ccxt, exchange_id, None)
                if exchange_class:
                    config = {"enableRateLimit": True}
                    if api_key and api_secret:
                        config["apiKey"] = api_key
                        config["secret"] = api_secret
                    self.exchange = exchange_class(config)
            except ImportError:
                logger.warning("ccxt not installed, crypto exchange features unavailable")

        logger.info(f"MarketDataFetcher initialized: {exchange_id}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data.

        Automatically detects Indian stocks (.NS/.BO suffix) and uses yfinance,
        otherwise uses CCXT for crypto.
        """
        # Indian stock market — use yfinance
        if self._is_indian_stock(symbol):
            return IndianStockFetcher.fetch_ohlcv(symbol, timeframe, period="1y")

        # Crypto — use CCXT
        if self.exchange is None:
            raise ValueError(f"No exchange configured for {symbol}. Use .NS/.BO suffix for Indian stocks.")

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
        """Fetch extended historical data."""
        # Indian stock market
        if self._is_indian_stock(symbol):
            period = "5y" if max_candles > 1000 else "2y"
            return IndianStockFetcher.fetch_ohlcv(symbol, timeframe, period=period)

        # Crypto — paginate through CCXT
        if self.exchange is None:
            raise ValueError(f"No exchange configured for {symbol}")

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
            current_since = raw[-1][0] + 1
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
        """Fetch current ticker/price data."""
        if self._is_indian_stock(symbol):
            return IndianStockFetcher.fetch_ticker(symbol)

        if self.exchange is None:
            raise ValueError(f"No exchange configured for {symbol}")

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

    @staticmethod
    def _is_indian_stock(symbol: str) -> bool:
        """Check if the symbol is an Indian stock (NSE/BSE)."""
        return symbol.upper().endswith((".NS", ".BO"))


class IndianStockFetcher:
    """Fetches Indian stock market data (NSE/BSE) via yfinance."""

    # Popular Indian stocks for quick reference
    POPULAR_STOCKS = {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "SBIN": "SBIN.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "HINDUNILVR": "HINDUNILVR.NS",
        "BHARTIARTL": "BHARTIARTL.NS",
        "ITC": "ITC.NS",
        "KOTAKBANK": "KOTAKBANK.NS",
        "LT": "LT.NS",
        "AXISBANK": "AXISBANK.NS",
        "WIPRO": "WIPRO.NS",
        "ADANIENT": "ADANIENT.NS",
        "TATAMOTORS": "TATAMOTORS.NS",
        "TATASTEEL": "TATASTEEL.NS",
        "MARUTI": "MARUTI.NS",
        "SUNPHARMA": "SUNPHARMA.NS",
        "TITAN": "TITAN.NS",
        "BAJFINANCE": "BAJFINANCE.NS",
        "NIFTY50": "^NSEI",         # Nifty 50 Index
        "SENSEX": "^BSESN",         # BSE Sensex
    }

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
    }

    # yfinance period limits by interval
    PERIOD_LIMITS = {
        "1m": "7d",        # 1-minute data: max 7 days
        "5m": "60d",       # 5-minute: max 60 days
        "15m": "60d",
        "1h": "2y",        # hourly: max 2 years
        "1d": "max",       # daily: all available history
    }

    @classmethod
    def fetch_ohlcv(
        cls,
        symbol: str,
        timeframe: str = "1d",
        period: str | None = None,
    ) -> pd.DataFrame:
        """Fetch Indian stock OHLCV data.

        Args:
            symbol: NSE ticker (e.g., 'RELIANCE.NS', 'TCS.NS').
            timeframe: Data interval (1m, 5m, 15m, 1h, 1d).
            period: How far back (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max).
        """
        interval = cls.INTERVAL_MAP.get(timeframe, "1d")

        # Auto-select period based on interval if not specified
        if period is None:
            period = cls.PERIOD_LIMITS.get(timeframe, "1y")

        logger.info(f"Fetching Indian stock: {symbol} | interval={interval} | period={period}")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}. Check if the ticker is valid.")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Normalize columns
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]

        logger.info(f"Fetched {len(df)} candles for {symbol} ({interval})")
        return df

    @classmethod
    def fetch_ticker(cls, symbol: str) -> dict:
        """Fetch current price info for an Indian stock."""
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        return {
            "symbol": symbol,
            "last": info.get("lastPrice", info.get("previousClose", 0)),
            "bid": None,
            "ask": None,
            "volume": info.get("lastVolume", 0),
            "change_pct": None,
            "market_cap": info.get("marketCap", 0),
            "timestamp": datetime.now(timezone.utc),
        }

    @classmethod
    def fetch_stock_info(cls, symbol: str) -> dict:
        """Fetch detailed stock information."""
        ticker = yf.Ticker(symbol)
        try:
            info = ticker.info
        except Exception:
            info = {}

        return {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }

    @classmethod
    def search_stock(cls, query: str) -> str | None:
        """Search for Indian stock ticker by name."""
        query_upper = query.upper().replace(" ", "")
        if query_upper in cls.POPULAR_STOCKS:
            return cls.POPULAR_STOCKS[query_upper]
        # Try adding .NS suffix
        test_symbol = f"{query_upper}.NS"
        try:
            ticker = yf.Ticker(test_symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                return test_symbol
        except Exception:
            pass
        return None
