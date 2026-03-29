"""Database storage for market data and trade history."""

from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.logger import get_logger

logger = get_logger("data.storage")


class Base(DeclarativeBase):
    pass


class OHLCVRecord(Base):
    """Stores OHLCV candle data."""

    __tablename__ = "ohlcv"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class TradeRecord(Base):
    """Stores executed trade history."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(4), nullable=False)  # BUY or SELL
    order_type = Column(String(10), nullable=False)  # market, limit, stop
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    timestamp = Column(DateTime, nullable=False, index=True)
    strategy = Column(String(50))
    signal_reason = Column(Text)
    confidence = Column(Float)
    mode = Column(String(10), default="paper")  # paper, live, backtest


class PortfolioSnapshot(Base):
    """Stores periodic portfolio snapshots."""

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)


class DataStorage:
    """Handles all database operations for the trading bot."""

    def __init__(self, db_url: str = "sqlite:///data/trading.db"):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Database initialized: {db_url}")

    def _get_session(self) -> Session:
        return self.SessionLocal()

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """Save OHLCV data to database, avoiding duplicates."""
        session = self._get_session()
        try:
            saved = 0
            for timestamp, row in df.iterrows():
                exists = (
                    session.query(OHLCVRecord)
                    .filter_by(symbol=symbol, timeframe=timeframe, timestamp=timestamp)
                    .first()
                )
                if not exists:
                    record = OHLCVRecord(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                    )
                    session.add(record)
                    saved += 1

            session.commit()
            logger.info(f"Saved {saved} new OHLCV records for {symbol} ({timeframe})")
            return saved
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving OHLCV data: {e}")
            raise
        finally:
            session.close()

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from database."""
        session = self._get_session()
        try:
            query = session.query(OHLCVRecord).filter_by(symbol=symbol, timeframe=timeframe)
            if start:
                query = query.filter(OHLCVRecord.timestamp >= start)
            if end:
                query = query.filter(OHLCVRecord.timestamp <= end)
            query = query.order_by(OHLCVRecord.timestamp)

            records = query.all()
            if not records:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

            data = [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in records
            ]
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df
        finally:
            session.close()

    def save_trade(self, trade: dict) -> int:
        """Save a trade record to the database."""
        session = self._get_session()
        try:
            record = TradeRecord(**trade)
            session.add(record)
            session.commit()
            logger.info(f"Trade saved: {trade['side']} {trade['quantity']} {trade['symbol']} @ {trade['price']}")
            return record.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving trade: {e}")
            raise
        finally:
            session.close()

    def get_trades(
        self,
        symbol: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve trade history."""
        session = self._get_session()
        try:
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter_by(symbol=symbol)
            if mode:
                query = query.filter_by(mode=mode)
            query = query.order_by(TradeRecord.timestamp.desc()).limit(limit)

            return [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "order_type": t.order_type,
                    "price": t.price,
                    "quantity": t.quantity,
                    "total": t.total,
                    "fee": t.fee,
                    "timestamp": t.timestamp,
                    "strategy": t.strategy,
                    "signal_reason": t.signal_reason,
                    "confidence": t.confidence,
                    "mode": t.mode,
                }
                for t in query.all()
            ]
        finally:
            session.close()

    def save_portfolio_snapshot(self, snapshot: dict) -> None:
        """Save a portfolio snapshot."""
        session = self._get_session()
        try:
            record = PortfolioSnapshot(**snapshot)
            session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving portfolio snapshot: {e}")
            raise
        finally:
            session.close()

    def get_portfolio_history(self, limit: int = 500) -> pd.DataFrame:
        """Get portfolio value history."""
        session = self._get_session()
        try:
            records = (
                session.query(PortfolioSnapshot)
                .order_by(PortfolioSnapshot.timestamp.desc())
                .limit(limit)
                .all()
            )
            if not records:
                return pd.DataFrame()

            data = [
                {
                    "timestamp": r.timestamp,
                    "total_value": r.total_value,
                    "cash_balance": r.cash_balance,
                    "unrealized_pnl": r.unrealized_pnl,
                    "realized_pnl": r.realized_pnl,
                }
                for r in records
            ]
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df.sort_index()
        finally:
            session.close()
