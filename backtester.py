from __future__ import annotations

from utility.trader_ABS import Trader
from utility.broker_apis.broker_ABS import Broker
from utility.market_news.market_news_provider_ABS import NewsProviderABS
from utility.market_data.price_data_provider import PriceDataProvider
from utility.market_news.market_news import MarketNews

from datetime import datetime, timedelta
from typing import Optional
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
from lead_lag_trading.investigation.lead_lag_calc import LeadLagResult


class Backtester:
    def __init__(
        self,
        trader: Trader,
        broker: Broker,
        news_provider: Optional[NewsProviderABS],
        price_provider: PriceDataProvider,
    ):
        self.trader = trader
        self.broker = broker
        self.news_provider = news_provider
        self.price_provider = price_provider

    @staticmethod
    def _is_weekend(date_str: str) -> bool:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() >= 5

    @staticmethod
    def _normalize_price_df(raw_prices, ticker: str) -> pd.DataFrame:
        """
        Normalize price-provider output into a DataFrame with one column named by ticker.
        Accepts Series or DataFrame.
        """
        if raw_prices is None:
            return pd.DataFrame(columns=[ticker])

        if isinstance(raw_prices, pd.Series):
            df = raw_prices.to_frame(name=ticker)
            return df

        if isinstance(raw_prices, pd.DataFrame):
            if raw_prices.empty:
                return pd.DataFrame(columns=[ticker])

            # If provider returns single unnamed/other-named column, rename it
            if ticker in raw_prices.columns:
                return raw_prices[[ticker]].copy()

            if raw_prices.shape[1] == 1:
                df = raw_prices.copy()
                df.columns = [ticker]
                return df

            # Try common price column names
            for col in ["Close", "Adj Close", "price", "close"]:
                if col in raw_prices.columns:
                    return raw_prices[[col]].rename(columns={col: ticker})

        raise ValueError(f"Unsupported historical price format for ticker {ticker}")

    def _get_multi_ticker_history(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Build a combined historical price DataFrame for multiple tickers by querying
        the provider one ticker at a time.
        """
        frames: list[pd.DataFrame] = []

        for ticker in sorted(set(tickers)):
            raw = self.price_provider.get_historical_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            df = self._normalize_price_df(raw, ticker)
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, axis=1)
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        return merged

    def _extract_trader_tickers(self, default_ticker: Optional[str] = None) -> list[str]:
        """
        Extract all needed tickers for market-data traders.
        Falls back to default_ticker if trader does not expose lag-related tickers.
        """
        tickers: set[str] = set()

        lag_pairs = getattr(self.trader, "lag_related_tickers", None)
        if lag_pairs:
            for pair in lag_pairs:
                leader = getattr(pair, "leader", None)
                follower = getattr(pair, "follower", None)
                if leader:
                    tickers.add(leader)
                if follower:
                    tickers.add(follower)

        trader_ticker = getattr(self.trader, "ticker", None)
        if trader_ticker:
            tickers.add(trader_ticker)

        if default_ticker:
            tickers.add(default_ticker)

        return sorted(tickers)

    def _group_news_by_date(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_news: int = 1000,
    ) -> dict[str, list[dict]]:
        if self.news_provider is None:
            return {}

        news_list = self.news_provider.get_news_by_ticker(
            ticker=ticker,
            from_date=start_date,
            to_date=end_date,
            amount=max_news,
        )

        news_list.sort(key=lambda x: x.get("datetime", 0))

        news_by_date: dict[str, list[dict]] = {}
        for news_dict in news_list:
            raw_timestamp = news_dict.get("datetime")
            if raw_timestamp is None:
                logging.warning("Skipping news item with missing datetime.")
                continue

            news_date = datetime.fromtimestamp(raw_timestamp).strftime("%Y-%m-%d")

            if self._is_weekend(news_date):
                logging.info(
                    "Skipping weekend news: %s (%s)",
                    news_dict.get("headline"),
                    news_date,
                )
                continue

            news_by_date.setdefault(news_date, []).append(news_dict)

        return news_by_date

    def _run_news_trader_day(
        self,
        date_str: str,
        ticker: str,
        day_price: float,
        daily_news: list[dict],
    ) -> None:
        """
        Run a news-driven trader for all news items on one date.
        """
        for news_dict in daily_news:
            site_source = self.news_provider.get_site_source(news_dict.get("url"))
            market_new = MarketNews(
                title=news_dict.get("headline"),
                content=site_source,
                source=news_dict.get("source"),
                date=date_str,
            )

            # For news traders, preserve legacy behavior:
            # pass the single-ticker price as marketData if that's what old traders expect.
            self.trader.trade(
                self.broker,
                news=market_new,
                marketData=day_price,
            )

    def _run_market_data_trader_day(
        self,
        history_until_date: pd.DataFrame,
    ) -> None:
        """
        Run a market-data-driven trader once for the current day using the full
        historical window up to that day.
        """
        self.trader.trade(
            self.broker,
            news=None,
            marketData=history_until_date,
        )

    def run_backtest(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_news: bool = False,
        use_market_data: bool = True,
        lookback_days: int = 30,
        market_data_tickers: Optional[list[str]] = None,
        max_news: int = 1000,
    ):
        """
        Supports:
        - news-only traders
        - market-data-only traders
        - hybrid traders

        Parameters:
        - use_news: run per-news-item processing
        - use_market_data: run once per trading day with rolling price history
        - lookback_days: minimum history length before market-data trader runs
        - market_data_tickers: override auto-detected ticker universe for market data
        """
        initial_cash = self.broker.cash

        news_by_date: dict[str, list[dict]] = {}
        if use_news and self.news_provider is not None:
            news_by_date = self._group_news_by_date(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                max_news=max_news,
            )

        if market_data_tickers is None:
            market_data_tickers = self._extract_trader_tickers(default_ticker=ticker)

        full_price_df = pd.DataFrame()
        if use_market_data:
            full_price_df = self._get_multi_ticker_history(
                tickers=market_data_tickers,
                start_date=start_date,
                end_date=end_date,
            )
            if full_price_df.empty:
                logging.warning("Market-data mode enabled, but no price history was loaded.")

        # Build the set of backtest dates from either news dates or price dates
        date_set: set[str] = set()

        if use_news:
            date_set.update(news_by_date.keys())

        if use_market_data and not full_price_df.empty:
            date_set.update(pd.to_datetime(full_price_df.index).strftime("%Y-%m-%d").tolist())

        all_dates = sorted(date_set)

        if not all_dates:
            logging.warning("No dates available for backtest.")
            return {
                "final_value": self.broker.cash,
                "profit": 0.0,
                "trade_history": getattr(self.broker, "trade_history", []),
            }

        for date_str in all_dates:
            # Skip weekends universally
            if self._is_weekend(date_str):
                continue

            # Get current market slice for this date
            day_row = None
            history_until_date = pd.DataFrame()

            if use_market_data and not full_price_df.empty:
                current_ts = pd.Timestamp(date_str)
                history_until_date = full_price_df.loc[full_price_df.index <= current_ts].copy()

                if not history_until_date.empty:
                    # only run after enough history accumulates
                    if len(history_until_date) >= lookback_days:
                        self._run_market_data_trader_day(history_until_date)

                    # last available row on/before current date
                    day_row = history_until_date.iloc[-1].dropna().to_dict()

            # News-driven path
            if use_news and date_str in news_by_date:
                try:
                    day_price = self.price_provider.get_price_on_date(ticker, date_str)
                except Exception:
                    day_price = None

                if day_price is None:
                    logging.info("No single-ticker price found for %s on %s, skipping news trades.", ticker, date_str)
                else:
                    self._run_news_trader_day(
                        date_str=date_str,
                        ticker=ticker,
                        day_price=day_price,
                        daily_news=news_by_date[date_str],
                    )

            # Portfolio valuation
            if day_row:
                current_value = self.broker.get_portfolio_value(day_row)
            else:
                # fallback to single-ticker valuation if market-data universe is unavailable
                try:
                    px = self.price_provider.get_price_on_date(ticker, date_str)
                except Exception:
                    px = None

                if px is None:
                    continue

                current_value = self.broker.get_portfolio_value({ticker: px})

            current_profit = current_value - initial_cash
            print(f"Date: {date_str}, Portfolio Value: {current_value:.2f}, Profit: {current_profit:.2f}")

        # Final valuation
        if use_market_data and not full_price_df.empty:
            latest_prices = full_price_df.iloc[-1].dropna().to_dict()
        else:
            latest_prices = self.price_provider.get_latest_prices([ticker])

        final_value = self.broker.get_portfolio_value(latest_prices)
        profit = final_value - initial_cash

        print(f"Backtest completed. Final portfolio value: {final_value:.2f}, Profit: {profit:.2f}")
        return {
            "final_value": final_value,
            "profit": profit,
            "trade_history": getattr(self.broker, "trade_history", []),
        }


if __name__ == "__main__":
    from utility.broker_apis.broker_ABS import SimulatedBroker
    from utility.market_news.market_news_provider_ABS import FinnhubNewsProvider
    from utility.market_data.price_data_provider import PriceDataProvider

    # Example 1: news trader
    from regex_news_trading.regex_trader import RegexTrader
    from lead_lag_trading.lag_trader import LagTrader

    lagged_example_pairs = [
        LeadLagResult(
            leader="AAPL",
            follower="XLE",
            best_lag=14,
            correlation=-0.1069,
            r_squared=0.0114,
            observations=738,
            direction="negative",
            stability_mean=-0.1073,
            stability_std=0.0740,
            sign_consistency=0.9057,
            stability_score=1.3130,
            score=0.9270,
            passes_filters=True,
        ),
        LeadLagResult(
            leader="UUP",
            follower="MSFT",
            best_lag=16,
            correlation=0.1400,
            r_squared=0.0196,
            observations=736,
            direction="positive",
            stability_mean=0.1166,
            stability_std=0.1157,
            sign_consistency=0.8021,
            stability_score=0.8079,
            score=0.7464,
            passes_filters=True,
        ),
    ]

    trader = LagTrader(lag_related_tickers=lagged_example_pairs)
    broker = SimulatedBroker()
    news_provider = FinnhubNewsProvider()
    price_provider = PriceDataProvider()

    backtester = Backtester(trader, broker, news_provider, price_provider)
    results = backtester.run_backtest(
        ticker="AAPL",
        start_date="2025-01-01",
        end_date="2025-12-31",
        use_news=False,
        use_market_data=True,
    )
    print(results)
