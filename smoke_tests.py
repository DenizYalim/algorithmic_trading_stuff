import unittest

import numpy as np
import pandas as pd

from backtester import Backtester
from GA_prediction import GATrader, GeneticAlgorithmConfig
from GA_prediction.Problem import MarketPredictionProblem
from momentum_trading import MomentumTrader
from utility.broker_apis.broker_ABS import SimulatedBroker
from utility.trader_ABS import TradeRequest


class SyntheticPriceProvider:
    def __init__(self):
        points = np.arange(180)
        prices = 100 + 0.04 * points + 3.0 * np.sin(points / 4.0)
        self.history = pd.DataFrame(
            {"AAPL": prices},
            index=pd.date_range("2025-01-01", periods=len(prices), freq="B"),
        )

    def get_historical_prices(self, ticker, start_date, end_date):
        return self.history.loc[start_date:end_date, [ticker]]

    def get_price_on_date(self, ticker, date):
        data = self.history.loc[:date]
        if data.empty:
            return None
        return float(data[ticker].iloc[-1])

    def get_latest_prices(self, tickers):
        return {
            ticker: float(self.history[ticker].iloc[-1])
            for ticker in tickers
        }


class GeneticAlgorithmSmokeTests(unittest.TestCase):
    def test_ga_problem_does_not_label_latest_feature_row(self):
        rng = np.random.default_rng(7)
        prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 80))
        market_data = pd.DataFrame(
            {"AAPL": prices},
            index=pd.date_range("2025-01-01", periods=len(prices), freq="B"),
        )

        problem = MarketPredictionProblem(market_data, "AAPL")

        self.assertGreater(problem.feature_frame.index[-1], problem.dataset.index[-1])

    def test_ga_trader_submits_trade_request_to_broker(self):
        points = np.arange(180)
        prices = 100 + 0.04 * points + 3.0 * np.sin(points / 4.0)
        market_data = pd.DataFrame(
            {"AAPL": prices},
            index=pd.date_range("2025-01-01", periods=len(prices), freq="B"),
        )

        trader = GATrader(
            "AAPL",
            min_training_rows=80,
            min_confidence_to_trade=0.0,
            ga_config=GeneticAlgorithmConfig(
                population_size=20,
                generations=8,
                random_seed=3,
            ),
        )
        broker = SimulatedBroker(initial_cash=10000)

        executions = trader.trade(broker, marketData=market_data)

        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0]["symbol"], "AAPL")
        self.assertIn(executions[0]["action"], {"buy", "sell"})

    def test_simulated_broker_accepts_trade_request_shape(self):
        broker = SimulatedBroker(initial_cash=1000)
        request = TradeRequest(
            symbol="AAPL",
            option="buy",
            quantity=2,
            price=100.0,
            confidence=0.9,
        )

        executions = broker.place_trade_requests([request])

        self.assertEqual(executions[0]["quantity"], 2)
        self.assertEqual(broker.positions["AAPL"], 2)
        self.assertEqual(broker.cash, 800.0)

    def test_ga_trader_runs_through_backtester_with_current_prices(self):
        trader = GATrader(
            "AAPL",
            min_training_rows=60,
            min_confidence_to_trade=0.0,
            retrain_every_rows=60,
            ga_config=GeneticAlgorithmConfig(
                population_size=12,
                generations=4,
                random_seed=3,
            ),
        )
        broker = SimulatedBroker(initial_cash=10000)
        backtester = Backtester(
            trader=trader,
            broker=broker,
            news_provider=None,
            price_provider=SyntheticPriceProvider(),
        )

        results = backtester.run_backtest(
            ticker="AAPL",
            start_date="2025-01-01",
            end_date="2025-09-30",
            use_news=False,
            use_market_data=True,
            lookback_days=60,
        )

        self.assertGreater(len(results["trade_history"]), 0)
        traded_prices = {trade["price"] for trade in results["trade_history"]}
        traded_dates = {trade["date"] for trade in results["trade_history"]}
        self.assertGreater(len(traded_prices), 1)
        self.assertNotIn(None, traded_dates)

    def test_momentum_trader_runs_through_backtester(self):
        trader = MomentumTrader(
            "AAPL",
            short_window=5,
            long_window=20,
            min_confidence_to_trade=0.0,
        )
        broker = SimulatedBroker(initial_cash=10000)
        backtester = Backtester(
            trader=trader,
            broker=broker,
            news_provider=None,
            price_provider=SyntheticPriceProvider(),
        )

        results = backtester.run_backtest(
            ticker="AAPL",
            start_date="2025-01-01",
            end_date="2025-09-30",
            use_news=False,
            use_market_data=True,
            lookback_days=20,
        )

        self.assertGreater(len(results["trade_history"]), 0)
        traded_dates = {trade["date"] for trade in results["trade_history"]}
        self.assertNotIn(None, traded_dates)


if __name__ == "__main__":
    unittest.main()
