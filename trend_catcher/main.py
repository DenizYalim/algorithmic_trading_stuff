from market_news_ABS import MarketNew
from news_classifier_ABS import REGEX_classifier
from trend import Trend
from broker_ABS import Broker, simulation_broker
from trend_catcher_ABS import TrendCatcher
from trader_ABS import regex_trader

def ask_for_news() -> MarketNew:
    # Implement logic to fetch news from various sources and return a MarketNew object
    return "you asked for news and you got it (placeholder)"


def start():  # REGEX DEMO
    while True:
        news = ask_for_news()

        news_classifier = REGEX_classifier()
        # news.set_classification(news_classifier)

        # trend_catcher = TrendCatcher()
        # trend_catcher.analyze_trend(news_classification)
        
        broker: Broker = simulation_broker()
        # broker.place_trade(trend_catcher)
        trader = regex_trader()
        trader

if __name__ == "__main__":
    start()
