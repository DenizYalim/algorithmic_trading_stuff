from utility.market_news._finnhub_wrapper import ticker_news, get_news
from abc import ABC, abstractmethod
import requests


class NewsProviderABS(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_news_list(self, amount=1, ticker=None, from_date=None, to_date=None) -> list[dict]:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_news_by_ticker(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_site_source(self, url: str) -> str:
        text = requests.get(url).text
        return text


class FinnhubNewsProvider(NewsProviderABS):
    def get_news_list(self, amount=1, from_date=None, to_date=None) -> list[dict]:
        return get_news()[:amount]  # todo maybe add from_date and to_date to this as well

    def get_news_by_ticker(self, ticker: str, from_date: str, to_date: str, amount=1) -> list[dict]:
        return ticker_news(ticker, from_date, to_date)[:amount]


if __name__ == "__main__":
    provider = FinnhubNewsProvider()
    news: dict = provider.get_news_list(amount=5)[4]
