from utility.market_news.market_news import MarketNews
from utility.broker_apis.broker_ABS import Broker, SimulatedBroker
from utility.trader_ABS import RegexTrader


def ask_for_news() -> MarketNews:
    # Implement logic to fetch news from various sources and return a MarketNew object
    return MarketNews(title="Apple stock is bullish", content="Apple's stock is expected to grow due to strong sales.", source="News Source", date="2024-06-01")


def start():  # REGEX DEMO
    # while True:
    print("Asking for news...")
    news = ask_for_news()
    print(f"Received news: {news.title} - {news.content}")
    broker: Broker = SimulatedBroker()
    # broker.place_trade(trend_catcher)
    trader = RegexTrader()
    trader.trade(broker=broker, news=news)


if __name__ == "__main__":
    start()
