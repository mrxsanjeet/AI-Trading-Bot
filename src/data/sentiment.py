"""News sentiment analysis for trading signals."""

from datetime import datetime, timezone
from typing import Optional

import requests

from src.utils.logger import get_logger

logger = get_logger("data.sentiment")


class SentimentAnalyzer:
    """Analyzes news and social media sentiment for trading assets."""

    SENTIMENT_LABELS = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def __init__(self):
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-load the sentiment analysis pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,  # CPU
                )
                logger.info("Sentiment analysis pipeline loaded")
            except ImportError:
                logger.warning("transformers not installed; sentiment analysis unavailable")
                raise
        return self._pipeline

    def analyze_text(self, text: str) -> dict:
        """Analyze sentiment of a single text.

        Returns:
            dict with keys: label (POSITIVE/NEGATIVE), score (0-1), sentiment_value (-1 to 1).
        """
        pipe = self._get_pipeline()
        result = pipe(text[:512])[0]  # Truncate to model max length

        label = result["label"]
        score = result["score"]
        sentiment_value = score if label == "POSITIVE" else -score

        return {
            "text": text[:100],
            "label": label,
            "score": score,
            "sentiment_value": sentiment_value,
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze_text(t) for t in texts]

    def get_aggregate_sentiment(self, texts: list[str]) -> dict:
        """Get aggregate sentiment from multiple texts.

        Returns:
            dict with: avg_sentiment (-1 to 1), bullish_pct, bearish_pct, count.
        """
        if not texts:
            return {"avg_sentiment": 0.0, "bullish_pct": 0.0, "bearish_pct": 0.0, "count": 0}

        results = self.analyze_batch(texts)
        sentiments = [r["sentiment_value"] for r in results]

        bullish = sum(1 for s in sentiments if s > 0.1)
        bearish = sum(1 for s in sentiments if s < -0.1)
        total = len(sentiments)

        return {
            "avg_sentiment": sum(sentiments) / total,
            "bullish_pct": bullish / total * 100,
            "bearish_pct": bearish / total * 100,
            "neutral_pct": (total - bullish - bearish) / total * 100,
            "count": total,
            "timestamp": datetime.now(timezone.utc),
        }

    def fetch_crypto_news(self, symbol: str, api_key: Optional[str] = None) -> list[str]:
        """Fetch recent crypto news headlines for a symbol.

        Uses CryptoCompare News API (free tier).
        """
        # Extract base currency from pair (e.g., BTC from BTC/USDT)
        base_currency = symbol.split("/")[0] if "/" in symbol else symbol

        url = "https://min-api.cryptocompare.com/data/v2/news/"
        params = {"categories": base_currency, "lang": "EN"}
        if api_key:
            params["api_key"] = api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            headlines = [article["title"] for article in data.get("Data", [])[:20]]
            logger.info(f"Fetched {len(headlines)} news headlines for {base_currency}")
            return headlines
        except requests.RequestException as e:
            logger.error(f"Error fetching news: {e}")
            return []
