from trend_catcher.trader_ABS import Trader
from utility.broker_apis.broker_ABS import Broker, SimulatedBroker
from utility.market_news.market_news_provider_ABS import NewsProviderABS
from utility.market_data.price_data_provider import PriceDataProvider
from utility.market_news.market_news import MarketNew
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO)


class Backtester:
    def __init__(self, trader: Trader, broker: Broker, news_provider: NewsProviderABS, price_provider: PriceDataProvider):
        self.trader = trader
        self.broker = broker
        self.news_provider = news_provider
        self.price_provider = price_provider

    @staticmethod
    def _is_weekend(date_str: str) -> bool:
        """
        Returns True if date_str (YYYY-MM-DD) is Saturday or Sunday.
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday

    def run_backtest(self, ticker: str, start_date: str, end_date: str):
        news_list = self.news_provider.get_news_by_ticker(ticker=ticker, from_date=start_date, to_date=end_date, amount=1000)

        news_list.sort(key=lambda x: x.get("datetime", 0))

        for news_dict in news_list:
            raw_timestamp = news_dict.get("datetime")
            if raw_timestamp is None:
                logging.warning("Skipping news item with missing datetime.")
                continue

            news_date = datetime.fromtimestamp(raw_timestamp).strftime("%Y-%m-%d")

            if self._is_weekend(news_date):
                logging.info(f"Skipping weekend news: {news_dict.get('headline')} ({news_date})")
                continue

            site_source = self.news_provider.get_site_source(news_dict.get("url"))

            market_new = MarketNew(title=news_dict.get("headline"), content=site_source, source=news_dict.get("source"), date=news_date)

            price = self.price_provider.get_price_on_date(ticker, market_new.date)
            if price is None:
                logging.info(f"No price found for {ticker} on {market_new.date}, skipping.")
                continue

            self.trader.trade(self.broker, market_new, price)

        latest_prices = self.price_provider.get_latest_prices([ticker])
        final_value = self.broker.get_portfolio_value(latest_prices)
        initial_cash = 10000
        profit = final_value - initial_cash

        print(f"Backtest completed. Final portfolio value: {final_value}, Profit: {profit}")
        return {"final_value": final_value, "profit": profit, "trade_history": self.broker.trade_history}


if __name__ == "__main__":
    from trend_catcher.trader_ABS import RegexTrader
    from utility.market_news.market_news_provider_ABS import FinnhubNewsProvider

    trader = RegexTrader(ticker="AAPL")
    broker = SimulatedBroker()
    news_provider = FinnhubNewsProvider()
    price_provider = PriceDataProvider()

    backtester = Backtester(trader, broker, news_provider, price_provider)
    results = backtester.run_backtest("AAPL", "2025-01-01", "2025-12-31")
    print(results)
