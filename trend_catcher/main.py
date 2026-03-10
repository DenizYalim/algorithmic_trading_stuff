from market_news_ABS import MarketNew
from broker_ABS import Broker, SimulatedBroker
from trader_ABS import RegexTrader

def ask_for_news() -> MarketNew:
    # Implement logic to fetch news from various sources and return a MarketNew object
    return "you asked for news and you got it (placeholder)"


def start():  # REGEX DEMO
    while True:
        news = ask_for_news()
 
        broker: Broker = SimulatedBroker()
        # broker.place_trade(trend_catcher)
        trader = RegexTrader()
        trader.trade()

if __name__ == "__main__":
    start()
