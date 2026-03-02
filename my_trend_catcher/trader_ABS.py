from abc import ABC, abstractmethod
from market_news_ABS import MarketNew, News_Classification
from news_classifier_ABS import News_Classifier


def trader(ABC):
    @abstractmethod
    def analyze_news(self, news:MarketNew, news_classification:News_Classification): # takes info regarding the news and analyzes it
        pass
    @abstractmethod
    def trade(self): # takes news; analyzes it and decides a trade to place
        pass

def regex_trader(trader):
    def analyze_news(self, news:MarketNew, news_classification:News_Classification): # takes news and news_classification; does regex on it. honestly doesn't even do news_classification
        # Implement regex-based news analysis logic here
        pass

    def __init__(self):
        pass


    def trade(self):
        # Implement regex-based trading logic here
        pass