from abc import ABC, abstractmethod
from market_news_ABS import MarketNew, News_Classification
from news_classifier_ABS import News_Classifier
from broker_ABS import Broker, TradeInfo
import logging


def _trade_request():
    def __init__(self, symbol, option, quantity, price, confidence):
        self.symbol = symbol
        self.option = option
        self.quantity = quantity
        self.price = price
        self.confidence = confidence


class Trader(ABC):
    @abstractmethod
    def analyze_news(self, news: MarketNew):  # takes info regarding the news and analyzes it
        pass

    @abstractmethod
    def trade(self, news: MarketNew):  # takes news; analyzes it and decides a trade to place
        pass


class RegexTrader(Trader):
    def __init__(self, bullish_patterns: list = None, bearish_patterns: list = None, confidence_needed=0.5):
        self.bullish_patterns = [""]  # default bullish patterns
        self.bearish_patterns = [""]  # defauly bearish patterns
        self.confidence_needed_to_trade = confidence_needed

        if bullish_patterns:  # if patterns are given explicitly
            self.bullish_patterns = bullish_patterns

        if bearish_patterns:  # if patterns are given explicitly
            self.bearish_patterns = bearish_patterns

    def analyze_news(self, news: MarketNew) -> _trade_request:
        # takes news and news_classification; does regex on it. honestly doesn't even do news_classification

        pass

    def trade(self, broker: Broker, news: MarketNew):
        trade_request: _trade_request = self.analyze_news(news)
        trade_inf = TradeInfo(symbol=trade_request.symbol, entry_price=trade_request.price)  # this conversion isn't nice create new file for trade_requests and trade_info later
        broker.place_trade(trade_inf)
        logging.info(f"symbol={trade_request.symbol}, entry_price={trade_request.price}")
