from setuptools import setup, find_packages

setup(
    name="ai-trading-bot",
    version="1.0.0",
    description="AI-powered cryptocurrency and stock trading bot",
    author="AI Trading Bot",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "ta>=0.11.0",
        "ccxt>=4.0.0",
        "torch>=2.1.0",
        "scikit-learn>=1.3.0",
        "sqlalchemy>=2.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "trading-bot=src.main:cli",
        ],
    },
)
