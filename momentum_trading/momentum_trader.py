from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from utility.broker_apis.broker_ABS import Broker
from utility.market_news.market_news import MarketNews
from utility.trader_ABS import Trader, TradeRequest

logging.basicConfig(level=logging.INFO)


class MomentumTrader(Trader):
    """
    Moving-average momentum trader.

    Compares a short moving average against a long moving average:
    - short MA > long MA: buy
    - short MA < long MA: sell
    - too close together: hold
    """

    def __init__(
        self,
        ticker: str,
        short_window: int = 10,
        long_window: int = 30,
        default_quantity: int = 1,
        min_confidence_to_trade: float = 0.05,
        allow_position_scaling: bool = False,
    ):
        if short_window < 1:
            raise ValueError("short_window must be at least 1")
        if long_window <= short_window:
            raise ValueError("long_window must be greater than short_window")

        self.ticker = ticker
        self.short_window = short_window
        self.long_window = long_window
        self.default_quantity = default_quantity
        self.min_confidence_to_trade = min_confidence_to_trade
        self.allow_position_scaling = allow_position_scaling

    @staticmethod
    def _validate_marketdata(marketData: pd.DataFrame) -> pd.DataFrame:
        if marketData is None:
            raise ValueError("marketData is None")
        if not isinstance(marketData, pd.DataFrame):
            raise TypeError("marketData must be a pandas DataFrame")
        if marketData.empty:
            raise ValueError("marketData is empty")

        return marketData.copy().sort_index().replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _get_price_series(marketData: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
        if ticker not in marketData.columns:
            return None

        selected = marketData[ticker]
        if isinstance(selected, pd.DataFrame):
            if selected.empty:
                return None
            selected = selected.iloc[:, 0]

        series = selected.dropna().astype(float)
        if series.empty:
            return None

        return series

    @staticmethod
    def _current_position_direction(broker: Broker, symbol: str) -> int:
        positions = getattr(broker, "positions", {})
        quantity = positions.get(symbol, 0)

        if quantity > 0:
            return 1
        if quantity < 0:
            return -1
        return 0

    def _analyze(
        self,
        news: MarketNews = None,
        marketData: pd.DataFrame = None,
    ) -> list[TradeRequest]:
        marketdata = self._validate_marketdata(marketData)
        series = self._get_price_series(marketdata, self.ticker)

        if series is None or len(series) < self.long_window:
            logging.info(
                "Skipping MomentumTrader; need %d rows for %s and got %d",
                self.long_window,
                self.ticker,
                0 if series is None else len(series),
            )
            return []

        short_average = float(series.tail(self.short_window).mean())
        long_average = float(series.tail(self.long_window).mean())

        if long_average == 0.0:
            return []

        spread = short_average / long_average - 1.0
        confidence = float(min(abs(spread) * 20.0, 1.0))

        if confidence < self.min_confidence_to_trade:
            logging.info(
                "Momentum signal too weak | symbol=%s confidence=%.4f threshold=%.4f",
                self.ticker,
                confidence,
                self.min_confidence_to_trade,
            )
            return []

        option = "buy" if spread > 0 else "sell"
        date = pd.Timestamp(series.index[-1]).strftime("%Y-%m-%d")

        return [
            TradeRequest(
                symbol=self.ticker,
                option=option,
                quantity=self.default_quantity,
                price=float(series.iloc[-1]),
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

            for request in trade_requests:
                current_direction = self._current_position_direction(broker, request.symbol)
                desired_direction = 1 if request.option == "buy" else -1

                if current_direction == desired_direction:
                    logging.info(
                        "Skipping MomentumTrader trade | symbol=%s already positioned %s",
                        request.symbol,
                        request.option,
                    )
                    continue

                filtered_requests.append(request)

            trade_requests = filtered_requests

        if not trade_requests:
            return []

        logging.info("Submitting %d MomentumTrader request(s)", len(trade_requests))
        return broker.place_trade_requests(trade_requests)
