import re
from dataclasses import dataclass
from abc import ABC, abstractmethod
from utility.broker_apis.broker_ABS import Broker, TradeInfo
from utility.market_news.market_news import MarketNew
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class TradeRequest:
    symbol: str
    option: str
    quantity: int
    price: float
    confidence: float


class Trader(ABC):
    @abstractmethod
    def _analyze_news(self, news: MarketNew):
        pass

    @abstractmethod
    def trade(self, broker: Broker, news: MarketNew, current_price: float):
        pass


import re


class RegexTrader(Trader):
    def __init__(self, bullish_patterns: list = None, bearish_patterns: list = None, confidence_needed=0.1, ticker="AAPL"):
        self.bullish_patterns = ["buy", "bullish", "growth"]
        self.bearish_patterns = ["sell", "bearish", "decline"]
        self.confidence_needed_to_trade = confidence_needed
        self.ticker = ticker

        if bullish_patterns:
            self.bullish_patterns = bullish_patterns

        if bearish_patterns:
            self.bearish_patterns = bearish_patterns

    def _analyze_news(self, news: MarketNew) -> TradeRequest:
        text = (news.title + " " + news.content).lower()

        bullish_matches = sum(1 for pattern in self.bullish_patterns if re.search(pattern, text))

        bearish_matches = sum(1 for pattern in self.bearish_patterns if re.search(pattern, text))

        total_patterns = len(self.bullish_patterns) + len(self.bearish_patterns)
        confidence = (bullish_matches + bearish_matches) / max(total_patterns, 1)

        # default direction
        option = "hold"

        if bullish_matches > bearish_matches:
            option = "buy"
        elif bearish_matches > bullish_matches:
            option = "sell"

        return TradeRequest(
            symbol=self.ticker, option=option, quantity=1, price=150.0, confidence=confidence
        )  # how to get price and symbol? maybe classifier can also return these, or we can have a separate extractor for these, or we can use regex in trader to extract these, idk yet

    def trade(self, broker: Broker, news: MarketNew, current_price: float):
        print(f"Analyzing news: {news.title} - {news.content}")

        trade_request: TradeRequest = self._analyze_news(news)
        trade_request.price = current_price  # Set the actual price

        if trade_request.confidence < self.confidence_needed_to_trade:
            logging.info(f"Trade skipped | symbol={trade_request.symbol} " f"confidence={trade_request.confidence}")
            return

        if trade_request.option == "hold":
            logging.info(f"No signal | symbol={trade_request.symbol}")
            return

        trade_inf = TradeInfo(symbol=trade_request.symbol, entry_price=trade_request.price, date=news.date)

        broker.place_trade(trade_inf)

        logging.info(f"Trade executed | symbol={trade_request.symbol} " f"option={trade_request.option} " f"price={trade_request.price} " f"confidence={trade_request.confidence}")
