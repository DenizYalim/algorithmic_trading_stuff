from abc import ABC, abstractmethod
from trend import Trend
from utility.market_news.market_news import MarketNews
from news_classifier_ABS import News_Classification


class TrendCatcher(ABC):
    @abstractmethod
    def catch_trend(self, news: MarketNews, classification: News_Classification) -> Trend:
        pass


class LLMCatcher(TrendCatcher):
    def catch_trend(self, news: MarketNews, classification: News_Classification) -> Trend:
        pass


class REGEXCatcher(TrendCatcher):
    def catch_trend(self, news: MarketNews, classification: News_Classification) -> Trend:

        return Trend(["APPL", "TEST"], "bull", 0.8)  # Placeholder
