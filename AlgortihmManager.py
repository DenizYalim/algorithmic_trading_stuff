# This file will manage all active algorithms, and will be responsible for starting and stopping them as needed.
# Will probably have google sheets ui, way later on

from utility.trader_ABS import RegexTrader
from utility.market_news.market_news_provider_ABS import FinnhubNewsProvider
from utility.broker_apis.broker_ABS import Broker, SimulatedBroker
from utility.market_news.market_news import MarketNews


def finnhub_regex_trader_no_ticker():
    trader = RegexTrader()
    news_provider = FinnhubNewsProvider()
    # TODO


def finnhub_regex_trader_ticker(ticker="AAPL"):  # news provider obj doesnt create market_news obj which is dumb
    broker: Broker = SimulatedBroker()
    trader = RegexTrader()
    news_provider = FinnhubNewsProvider()

    news_list = news_provider.get_news_by_ticker(amount=1000, ticker=ticker, to_date="2025-08-01", from_date="2025-07-01")
    print(f"sayi: {len(news_list)}")

    for news in news_list:
        print(f"news: {news.get('headline')}")
        site_source = news_provider.get_site_source(news.get("url"))

        market_new = MarketNews(
            title=news.get("headline"),
            content=site_source,
            source=news.get("source"),
            date=news.get("datetime"),
        )

        trader.trade(broker=broker, news=market_new, current_price=150.0)


if __name__ == "__main__":
    finnhub_regex_trader_ticker()
