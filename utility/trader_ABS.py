import re
from dataclasses import dataclass
from abc import ABC, abstractmethod
from utility.broker_apis.broker_ABS import Broker, TradeInfo
from utility.market_news.market_news import MarketNews
import logging
import pandas as pd

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
    def _analyze(self, news: MarketNews = None, marketData: pd.DataFrame = None):
        pass

    @abstractmethod
    def trade(self, broker: Broker, news: MarketNews = None, marketData: pd.DataFrame = None):
        pass
