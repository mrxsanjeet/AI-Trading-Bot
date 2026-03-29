"""Main entry point for the AI Trading Bot CLI."""

import signal
import sys
import time
from datetime import datetime, timezone

import click

from src.utils.helpers import load_config, load_strategies_config
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("main")

# Graceful shutdown flag
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received. Shutting down gracefully...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@click.group()
def cli():
    """AI Trading Bot — ML-powered Indian Stock & Crypto Trading."""
    pass


@cli.command()
@click.option("--symbol", "-s", default=None, help="Stock ticker (e.g., RELIANCE.NS) or crypto pair")
@click.option("--timeframe", "-t", default=None, help="Candle timeframe (1m, 5m, 1h, 1d)")
@click.option("--interval", "-i", default=60, help="Polling interval in seconds")
def paper(symbol, timeframe, interval):
    """Run the bot in paper trading mode (fake money)."""
    config = load_config()
    strategies_config = load_strategies_config()

    symbols = [symbol] if symbol else config["trading"]["symbols"]
    tf = timeframe or config["trading"]["timeframe"]
    initial_capital = config["trading"]["initial_capital"]

    from src.data.cleaner import DataCleaner
    from src.data.features import FeatureEngineer
    from src.data.fetcher import MarketDataFetcher
    from src.data.storage import DataStorage
    from src.dashboard.alerts import AlertManager
    from src.execution.paper import PaperTrader
    from src.execution.portfolio import Portfolio
    from src.strategy.mean_reversion import MeanReversionStrategy
    from src.strategy.momentum import MomentumStrategy
    from src.strategy.risk import RiskManager
    from src.strategy.signals import SignalAggregator
    from src.strategy.trend import TrendFollowingStrategy

    # Initialize components
    exchange_id = config["trading"].get("exchange", "NSE")
    fetcher = MarketDataFetcher(exchange_id)
    cleaner = DataCleaner()
    feature_eng = FeatureEngineer()
    storage = DataStorage(config["database"]["url"])
    portfolio = Portfolio(initial_capital)
    risk_manager = RiskManager(config["risk"])
    alert_manager = AlertManager(config.get("alerts", {}))

    # Initialize strategies
    strat_config = strategies_config.get("strategies", {})
    strategies = [
        TrendFollowingStrategy(strat_config.get("trend_following", {})),
        MeanReversionStrategy(strat_config.get("mean_reversion", {})),
        MomentumStrategy(strat_config.get("momentum", {})),
    ]
    aggregator = SignalAggregator(strategies, min_agreement=2)

    paper_trader = PaperTrader(portfolio, risk_manager, storage)

    click.echo(f"\n{'='*60}")
    click.echo(f"  PAPER TRADING MODE (Fake Money)")
    click.echo(f"  Market:    {config['trading'].get('market', 'indian').upper()}")
    click.echo(f"  Symbols:   {', '.join(symbols)}")
    click.echo(f"  Timeframe: {tf}")
    click.echo(f"  Capital:   ${initial_capital:,}")
    click.echo(f"  Interval:  {interval}s")
    click.echo(f"{'='*60}\n")

    while not _shutdown:
        for sym in symbols:
            try:
                # Fetch latest data
                df = fetcher.fetch_ohlcv(sym, tf, limit=200)
                df = cleaner.clean(df)
                df = feature_eng.add_all_indicators(df)

                # Store data
                storage.save_ohlcv(df.tail(10), sym, tf)

                # Generate signal
                result = aggregator.generate_signals(df, sym)
                sig = result["aggregated_signal"]
                logger.info(f"[{sym}] {sig}")

                # Execute
                current_prices = {sym: df["close"].iloc[-1]}
                trade = paper_trader.execute_signal(sig, current_prices)
                if trade:
                    alert_manager.send_trade_alert(trade)

            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")

        # Log portfolio status
        prices = {}
        for sym in symbols:
            try:
                ticker = fetcher.fetch_ticker(sym)
                prices[sym] = ticker["last"]
            except Exception:
                pass

        if prices:
            stats = portfolio.get_stats(prices)
            logger.info(
                f"Portfolio: ${stats['total_value']:,.2f} | "
                f"P&L: ${stats['realized_pnl']:+,.2f} | "
                f"Positions: {stats['open_positions']}"
            )

        time.sleep(interval)

    # Shutdown
    logger.info("Shutting down paper trader...")
    if prices:
        logger.info(f"Final portfolio value: ${portfolio.total_value(prices):,.2f}")


@cli.command()
@click.option("--symbol", "-s", default=None, help="Stock ticker (e.g., RELIANCE.NS)")
@click.option("--timeframe", "-t", default="1d", help="Candle timeframe")
@click.option("--strategy", "-st", default="trend", help="Strategy: trend, mean_reversion, momentum")
@click.option("--capital", "-c", default=10000.0, help="Initial capital")
@click.option("--chart", is_flag=True, help="Generate TradingView-style HTML chart")
def backtest(symbol, timeframe, strategy, capital, chart):
    """Run a backtest on historical data."""
    config = load_config()
    strategies_config = load_strategies_config()

    # Default to first configured symbol if not specified
    if symbol is None:
        symbol = config["trading"]["symbols"][0]

    from src.data.cleaner import DataCleaner
    from src.data.features import FeatureEngineer
    from src.data.fetcher import MarketDataFetcher
    from src.execution.backtest import BacktestEngine
    from src.strategy.mean_reversion import MeanReversionStrategy
    from src.strategy.momentum import MomentumStrategy
    from src.strategy.trend import TrendFollowingStrategy

    strat_config = strategies_config.get("strategies", {})

    strategy_map = {
        "trend": TrendFollowingStrategy(strat_config.get("trend_following", {})),
        "mean_reversion": MeanReversionStrategy(strat_config.get("mean_reversion", {})),
        "momentum": MomentumStrategy(strat_config.get("momentum", {})),
    }

    if strategy not in strategy_map:
        click.echo(f"Unknown strategy: {strategy}. Choose from: {list(strategy_map.keys())}")
        sys.exit(1)

    selected_strategy = strategy_map[strategy]

    # Fetch historical data
    click.echo(f"Fetching historical data for {symbol} ({timeframe})...")
    exchange_id = config["trading"].get("exchange", "NSE")
    fetcher = MarketDataFetcher(exchange_id)
    cleaner = DataCleaner()
    feature_eng = FeatureEngineer()

    df = fetcher.fetch_all_history(symbol, timeframe, max_candles=2000)
    df = cleaner.clean(df)
    df = feature_eng.add_all_indicators(df)

    click.echo(f"Loaded {len(df)} candles. Running backtest...")

    # Run backtest
    engine = BacktestEngine(
        initial_capital=capital,
        fee_rate=0.001,
        slippage_pct=0.0005,
        risk_config=config["risk"],
    )
    result = engine.run(df, selected_strategy, symbol)
    metrics = result["metrics"]

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo(f"  BACKTEST RESULTS — {selected_strategy.name}")
    click.echo("=" * 60)
    click.echo(f"  Symbol:           {symbol}")
    click.echo(f"  Timeframe:        {timeframe}")
    click.echo(f"  Candles:          {len(df)}")
    click.echo(f"  Initial Capital:  ${metrics['initial_capital']:,.2f}")
    click.echo(f"  Final Value:      ${metrics['final_value']:,.2f}")
    click.echo(f"  Total Return:     {metrics['total_return_pct']:+.2f}%")
    click.echo(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    click.echo(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    click.echo(f"  Total Trades:     {metrics['total_trades']}")
    click.echo(f"  Win Rate:         {metrics['win_rate']:.1f}%")
    click.echo(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
    click.echo(f"  Avg Win:          ${metrics['avg_win']:,.2f}")
    click.echo(f"  Avg Loss:         ${metrics['avg_loss']:,.2f}")
    click.echo(f"  Total Fees:       ${metrics['total_fees']:,.2f}")
    click.echo("=" * 60)

    # Generate TradingView-style chart
    if chart:
        from src.dashboard.charts import generate_backtest_chart
        chart_path = generate_backtest_chart(df, result, symbol)
        click.echo(f"\n  Chart saved: {chart_path}")
        click.echo(f"  Open in browser to view TradingView-style chart")


@cli.command()
@click.option("--symbol", "-s", default=None, help="Stock ticker (e.g., RELIANCE.NS)")
@click.option("--timeframe", "-t", default="1d", help="Candle timeframe")
@click.option("--epochs", "-e", default=50, help="Training epochs")
def train(symbol, timeframe, epochs):
    """Train the ML prediction model on historical data."""
    config = load_config()

    if symbol is None:
        symbol = config["trading"]["symbols"][0]

    from src.data.cleaner import DataCleaner
    from src.data.features import FeatureEngineer
    from src.data.fetcher import MarketDataFetcher
    from src.ml.features import MLFeatureEngineer
    from src.ml.registry import ModelRegistry
    from src.ml.train import ModelTrainer

    click.echo(f"Fetching training data for {symbol}...")
    exchange_id = config["trading"].get("exchange", "NSE")
    fetcher = MarketDataFetcher(exchange_id)
    cleaner = DataCleaner()
    feature_eng = FeatureEngineer()

    df = fetcher.fetch_all_history(symbol, timeframe, max_candles=5000)
    df = cleaner.clean(df)
    df = feature_eng.add_all_indicators(df)

    click.echo(f"Loaded {len(df)} candles. Preparing features...")
    ml_config = config.get("ml", {})
    ml_config["epochs"] = epochs
    ml_feature_eng = MLFeatureEngineer(lookback_window=ml_config.get("lookback_window", 60))

    features = ml_feature_eng.prepare_features(df)
    X_train, X_test, y_train, y_test = ml_feature_eng.train_test_split(
        features["X"], features["y"]
    )

    click.echo(f"Training {ml_config.get('model_type', 'lstm').upper()} model...")
    trainer = ModelTrainer(ml_config)
    result = trainer.train(X_train, y_train, X_test, y_test)

    click.echo(f"\nTraining complete!")
    click.echo(f"  Best Accuracy: {result['best_accuracy']:.2%}")
    click.echo(f"  Epochs: {result['epochs_trained']}")

    # Register model
    registry = ModelRegistry()
    version = registry.register_model(
        model=result["model"],
        feature_engineer=ml_feature_eng,
        metrics={
            "best_accuracy": result["best_accuracy"],
            "best_val_loss": result["best_val_loss"],
        },
        config=ml_config,
        symbol=symbol,
    )
    click.echo(f"  Model saved: {version}")


@cli.command()
@click.option("--symbol", "-s", default=None, help="Stock ticker (e.g., RELIANCE.NS)")
@click.option("--timeframe", "-t", default="1d", help="Candle timeframe")
@click.option("--period", "-p", default="1y", help="Data period (1mo, 3mo, 6mo, 1y, 2y, 5y)")
def chart(symbol, timeframe, period):
    """Generate a TradingView-style interactive chart (opens in browser)."""
    config = load_config()

    if symbol is None:
        symbol = config["trading"]["symbols"][0]

    from src.data.cleaner import DataCleaner
    from src.data.features import FeatureEngineer
    from src.data.fetcher import MarketDataFetcher
    from src.dashboard.charts import generate_tradingview_chart

    click.echo(f"Fetching data for {symbol}...")
    exchange_id = config["trading"].get("exchange", "NSE")
    fetcher = MarketDataFetcher(exchange_id)
    cleaner = DataCleaner()
    feature_eng = FeatureEngineer()

    df = fetcher.fetch_ohlcv(symbol, timeframe)
    df = cleaner.clean(df)
    df = feature_eng.add_all_indicators(df)

    chart_path = generate_tradingview_chart(df, symbol, timeframe)
    click.echo(f"Chart saved: {chart_path}")

    import webbrowser
    webbrowser.open(f"file://{chart_path}")


@cli.command()
def dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    click.echo("Launching dashboard at http://localhost:8501 ...")
    subprocess.run(["streamlit", "run", "src/dashboard/app.py"])


@cli.command()
def status():
    """Show current bot status and configuration."""
    config = load_config()

    market = config["trading"].get("market", "crypto")
    exchange = config["trading"].get("exchange", "binance")

    click.echo("\n" + "=" * 55)
    click.echo("  AI Trading Bot — Status")
    click.echo("=" * 55)
    click.echo(f"  Market:     {market.upper()}")
    click.echo(f"  Exchange:   {exchange}")
    click.echo(f"  Mode:       {config['trading']['mode']}")
    click.echo(f"  Symbols:    {', '.join(config['trading']['symbols'])}")
    click.echo(f"  Timeframe:  {config['trading']['timeframe']}")
    click.echo(f"  Capital:    ${config['trading']['initial_capital']:,}")
    click.echo(f"\n  Risk Settings:")
    click.echo(f"    Max Position: {config['risk']['max_position_pct']}%")
    click.echo(f"    Stop Loss:    {config['risk']['stop_loss_pct']}%")
    click.echo(f"    Take Profit:  {config['risk']['take_profit_pct']}%")
    click.echo(f"    Max Drawdown: {config['risk']['max_drawdown_pct']}%")

    click.echo(f"\n  Configured Stocks:")
    for sym in config["trading"]["symbols"]:
        name = sym.replace(".NS", " (NSE)").replace(".BO", " (BSE)")
        click.echo(f"    - {name}")

    click.echo("=" * 55)

    # Check for trained models
    try:
        from src.ml.registry import ModelRegistry
        registry = ModelRegistry()
        models = registry.list_models()
        if models:
            click.echo(f"\n  Trained Models: {len(models)}")
            for m in models[-3:]:
                acc = m["metrics"].get("best_accuracy", "N/A")
                if isinstance(acc, float):
                    acc = f"{acc:.2%}"
                click.echo(f"    {m['version']} — {m['model_type']} | {m['symbol']} | acc: {acc}")
        else:
            click.echo("\n  No trained models yet.")
            click.echo("  Run: python -m src.main train -s RELIANCE.NS")
    except ImportError:
        click.echo("\n  ML models: torch not installed (install with: pip install torch)")
        click.echo("  Run: pip install torch && python -m src.main train -s RELIANCE.NS")

    click.echo()


if __name__ == "__main__":
    cli()
