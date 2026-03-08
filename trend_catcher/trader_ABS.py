from abc import ABC, abstractmethod
from market_news_ABS import MarketNew, News_Classification
from news_classifier_ABS import News_Classifier
from broker_ABS import Broker


def _trade_request():
    def __init__(self, option, quantity, price, confidence):
        self.option = option
        self.quantity = quantity
        self.price = price
        self.confidence = confidence

def trader(ABC):    
    @abstractmethod
    def analyze_news(self, news:MarketNew): # takes info regarding the news and analyzes it
        pass
    
    @abstractmethod
    def trade(self, news:MarketNew): # takes news; analyzes it and decides a trade to place
        pass

def regex_trader(trader):
    def analyze_news(self, news:MarketNew) -> _trade_request: # takes news and news_classification; does regex on it. honestly doesn't even do news_classification
 
        pass

    def __init__(self, bullish_patterns:list, bearish_patterns:list):
        pass


    def trade(self, news:MarketNew):

        # Implement regex-based trading logic here
        pass