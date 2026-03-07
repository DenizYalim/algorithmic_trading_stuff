from market_news_ABS import MarketNew
from news_classifier_ABS import News_Classifier, REGEX_classifier
from trend import Trend
from broker_ABS import Broker

def ask_for_news() -> MarketNew:
    # Implement logic to fetch news from various sources and return a MarketNew object
    return "you asked for news and you got it (placeholder)"

def start():    # REGEX DEMO
    while True:
        news = ask_for_news()
        
        news_classifier = REGEX_classifier()
        news_classification = news.classify_news(news_classifier)

        trend_analysis = Trend_Analysis()
        trend_analysis.analyze_trend(news_classification)
        broker:Broker = None # TODO: decide which broker to use based on some criteria
        broker.place_trade(trend_analysis)
        


if __name__ == "__main__":
    start()
