from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from GA_prediction.GA import GeneticAlgorithm, GeneticAlgorithmConfig
from GA_prediction.Problem import MarketPredictionProblem
from GA_prediction.Solution import GASolution
from utility.broker_apis.broker_ABS import Broker
from utility.market_news.market_news import MarketNews
from utility.trader_ABS import Trader, TradeRequest

logging.basicConfig(level=logging.INFO)


class GATrader(Trader):
    """
    Market-data trader powered by a small genetic algorithm.

    It trains on the historical marketData slice supplied by the Backtester,
    evolves a directional rule, and submits buy/sell requests through the same
    Broker abstraction used by the other traders.
    """

    def __init__(
        self,
        ticker: str,
        default_quantity: int = 1,
        min_confidence_to_trade: float = 0.52,
        min_training_rows: int = 60,
        retrain_every_rows: int = 20,
        horizon_days: int = 1,
        minimum_return: float = 0.0,
        trading_cost: float = 0.0005,
        allow_position_scaling: bool = False,
        ga_config: Optional[GeneticAlgorithmConfig] = None,
    ):
        self.ticker = ticker
        self.default_quantity = default_quantity
        self.min_confidence_to_trade = min_confidence_to_trade
        self.min_training_rows = min_training_rows
        self.retrain_every_rows = retrain_every_rows
        self.horizon_days = horizon_days
        self.minimum_return = minimum_return
        self.trading_cost = trading_cost
        self.allow_position_scaling = allow_position_scaling
        self.ga_config = ga_config or GeneticAlgorithmConfig()

        self.solution: GASolution | None = None
        self.problem: MarketPredictionProblem | None = None
        self._last_trained_rows = 0

    def _should_train(self, marketData: pd.DataFrame) -> bool:
        if self.solution is None or self.problem is None:
            return True
        return len(marketData) - self._last_trained_rows >= self.retrain_every_rows

    def fit(self, marketData: pd.DataFrame) -> GASolution:
        problem = MarketPredictionProblem(
            market_data=marketData,
            ticker=self.ticker,
            horizon_days=self.horizon_days,
            minimum_return=self.minimum_return,
            trading_cost=self.trading_cost,
        )
        ga = GeneticAlgorithm(problem=problem, config=self.ga_config)
        self.solution = ga.run()
        self.problem = problem
        self._last_trained_rows = len(marketData)

        logging.info(
            "GA trained | ticker=%s fitness=%.8f metrics=%s",
            self.ticker,
            self.solution.fitness,
            self.solution.metrics,
        )
        return self.solution

    def _analyze(
        self,
        news: MarketNews = None,
        marketData: pd.DataFrame = None,
    ) -> list[TradeRequest]:
        if marketData is None:
            logging.info("No marketData provided to GATrader")
            return []
        if not isinstance(marketData, pd.DataFrame):
            raise TypeError("marketData must be a pandas DataFrame")
        if len(marketData) < self.min_training_rows:
            logging.info(
                "Skipping GA trade; need at least %d rows and got %d",
                self.min_training_rows,
                len(marketData),
            )
            return []

        if self._should_train(marketData):
            try:
                self.fit(marketData)
            except ValueError as exc:
                logging.info("Skipping GA trade; training data not ready: %s", exc)
                return []

        if self.solution is None or self.problem is None:
            return []

        latest_prediction = int(
            self.solution.representation.predict(
                self.problem.latest_features_from_market_data(marketData)
            )[0]
        )
        if latest_prediction == 0:
            logging.info("GA signal is flat for %s", self.ticker)
            return []

        validation_accuracy = float(self.solution.metrics.get("accuracy", 0.0))
        validation_coverage = float(self.solution.metrics.get("coverage", 0.0))
        confidence = validation_accuracy

        if confidence < self.min_confidence_to_trade:
            logging.info(
                "Skipping GA trade | ticker=%s confidence=%.4f coverage=%.4f threshold=%.4f",
                self.ticker,
                confidence,
                validation_coverage,
                self.min_confidence_to_trade,
            )
            return []

        option = "buy" if latest_prediction > 0 else "sell"
        price = self.problem.latest_price_from_market_data(marketData)
        date = None
        if marketData.index is not None and len(marketData.index) > 0:
            latest_index = pd.Timestamp(marketData.index[-1])
            date = latest_index.strftime("%Y-%m-%d")

        return [
            TradeRequest(
                symbol=self.ticker,
                option=option,
                quantity=self.default_quantity,
                price=price,
                confidence=confidence,
                date=date,
            )
        ]

    def trade(
        self,
        broker: Broker,
        news: MarketNews = None,
        marketData: pd.DataFrame = None,
    ):
        trade_requests = self._analyze(news=news, marketData=marketData)
        if not trade_requests:
            return []

        if not self.allow_position_scaling:
            filtered_requests = []
            positions = getattr(broker, "positions", {})
            for request in trade_requests:
                current_qty = positions.get(request.symbol, 0)
                desired_direction = 1 if request.option == "buy" else -1
                current_direction = 0
                if current_qty > 0:
                    current_direction = 1
                elif current_qty < 0:
                    current_direction = -1

                if current_direction == desired_direction:
                    logging.info(
                        "Skipping GA trade | symbol=%s already positioned %s",
                        request.symbol,
                        request.option,
                    )
                    continue
                filtered_requests.append(request)

            trade_requests = filtered_requests

        if not trade_requests:
            return []

        logging.info("Submitting %d GA trade request(s)", len(trade_requests))
        return broker.place_trade_requests(trade_requests)
