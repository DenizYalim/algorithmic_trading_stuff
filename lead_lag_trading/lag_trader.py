from typing import Optional
import logging

import numpy as np
import pandas as pd

from lead_lag_trading.investigation.lead_lag_calc import LeadLagResult
from utility.broker_apis.broker_ABS import Broker
from utility.market_news.market_news import MarketNews
from utility.trader_ABS import Trader, TradeRequest

logging.basicConfig(level=logging.INFO)


class LagTrader(Trader):
    """
    Uses discovered lead-lag relationships to trade follower tickers.

    Core logic:
    - for each (leader, follower) relationship
    - inspect the latest signal on the leader
    - if leader is strongly long/short
    - map that into a long/short action on the follower
      using the relationship sign

    Positive relationship:
        leader long  -> follower long
        leader short -> follower short

    Negative relationship:
        leader long  -> follower short
        leader short -> follower long
    """

    def __init__(
        self,
        lag_related_tickers: list[LeadLagResult],
        default_quantity: int = 1,
        min_confidence_to_trade: float = 0.10,
        shock_threshold: float = 0.01,
        zscore_threshold: float = 1.5,
        zscore_window: int = 20,
        require_passed_filters: bool = True,
        use_zscore_if_available: bool = True,
    ):
        self.lag_related_tickers = lag_related_tickers
        self.default_quantity = default_quantity
        self.min_confidence_to_trade = min_confidence_to_trade
        self.shock_threshold = shock_threshold
        self.zscore_threshold = zscore_threshold
        self.zscore_window = zscore_window
        self.require_passed_filters = require_passed_filters
        self.use_zscore_if_available = use_zscore_if_available

    @staticmethod
    def _validate_marketdata(marketdata: pd.DataFrame) -> pd.DataFrame:
        if marketdata is None:
            raise ValueError("marketData is None")
        if not isinstance(marketdata, pd.DataFrame):
            raise TypeError("marketData must be a pandas DataFrame")
        if marketdata.empty:
            raise ValueError("marketData is empty")

        clean = marketdata.copy().sort_index()
        clean = clean.replace([np.inf, -np.inf], np.nan)
        return clean

    @staticmethod
    @staticmethod
    def _get_price_series(marketdata: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
        """
        Safely extract one ticker's price series even if marketdata[symbol]
        returns a DataFrame because of duplicate columns.
        """
        if symbol not in marketdata.columns:
            return None

        obj = marketdata[symbol]

        # If duplicate columns exist, marketdata[symbol] may be a DataFrame
        if isinstance(obj, pd.DataFrame):
            if obj.empty:
                return None

            # Keep the first non-empty column
            for col in obj.columns:
                candidate = obj[col].dropna()
                if not candidate.empty:
                    return candidate.astype(float)

            return None

        if isinstance(obj, pd.Series):
            series = obj.dropna()
            if series.empty:
                return None
            return series.astype(float)

        return None

    @staticmethod
    def _latest_price(series: pd.Series) -> Optional[float]:
        if series is None or series.empty:
            return None
        return float(series.iloc[-1])

    @staticmethod
    def _latest_return(series: pd.Series) -> Optional[float]:
        if series is None or len(series) < 2:
            return None

        returns = series.pct_change().dropna()
        if returns.empty:
            return None

        last_val = returns.iloc[-1]

        # Defensive: if something still came back as a Series, collapse it
        if isinstance(last_val, pd.Series):
            if last_val.empty:
                return None
            return float(last_val.iloc[0])

        return float(last_val)

    def _latest_zscore(self, series: pd.Series) -> Optional[float]:
        if series is None:
            return None

        returns = series.pct_change().dropna()
        if len(returns) < self.zscore_window:
            return None

        rolling_mean = returns.rolling(self.zscore_window).mean()
        rolling_std = returns.rolling(self.zscore_window).std()

        latest_ret = returns.iloc[-1]
        latest_mean = rolling_mean.iloc[-1]
        latest_std = rolling_std.iloc[-1]

        # Defensive collapse in case these are Series
        if isinstance(latest_ret, pd.Series):
            if latest_ret.empty:
                return None
            latest_ret = latest_ret.iloc[0]

        if isinstance(latest_mean, pd.Series):
            if latest_mean.empty:
                return None
            latest_mean = latest_mean.iloc[0]

        if isinstance(latest_std, pd.Series):
            if latest_std.empty:
                return None
            latest_std = latest_std.iloc[0]

        if pd.isna(latest_mean) or pd.isna(latest_std) or float(latest_std) == 0.0:
            return None

        return float((float(latest_ret) - float(latest_mean)) / float(latest_std))

    def _leader_signal_direction(
        self,
        leader_return: Optional[float],
        leader_zscore: Optional[float],
    ) -> int:
        """
        Returns:
            +1 => leader long signal
            -1 => leader short signal
             0 => no actionable signal
        """
        if self.use_zscore_if_available and leader_zscore is not None:
            if abs(leader_zscore) < self.zscore_threshold:
                return 0
            return 1 if leader_zscore > 0 else -1

        if leader_return is not None:
            if abs(leader_return) < self.shock_threshold:
                return 0
            return 1 if leader_return > 0 else -1

        return 0

    @staticmethod
    def _relationship_sign(pair: LeadLagResult) -> int:
        """
        +1 => positive relationship
        -1 => negative relationship
        """
        return 1 if pair.correlation > 0 else -1

    def _compute_confidence(
        self,
        pair: LeadLagResult,
        leader_return: Optional[float],
        leader_zscore: Optional[float],
    ) -> float:
        """
        Simple bounded confidence score.
        """
        base_corr = abs(float(pair.correlation))
        sign_consistency = float(getattr(pair, "sign_consistency", 0.5))
        stability_score = float(getattr(pair, "stability_score", 0.0))

        shock_strength = 0.0
        if leader_zscore is not None:
            shock_strength = min(abs(leader_zscore) / max(self.zscore_threshold, 1e-9), 2.0)
        elif leader_return is not None:
            shock_strength = min(abs(leader_return) / max(self.shock_threshold, 1e-9), 2.0)

        stability_component = min(max(stability_score / 2.0, 0.0), 1.0)

        confidence = 0.45 * base_corr + 0.25 * sign_consistency + 0.15 * stability_component + 0.15 * min(shock_strength / 2.0, 1.0)

        return float(min(max(confidence, 0.0), 1.0))

    def _pair_to_trade_request(
        self,
        pair: LeadLagResult,
        marketdata: pd.DataFrame,
    ) -> Optional[TradeRequest]:
        """
        Evaluate one lead/follower pair.

        If leader has a strong signal:
        - determine whether leader is long or short
        - convert that into follower long/short using relationship sign
        - return a TradeRequest
        """
        if self.require_passed_filters and hasattr(pair, "passes_filters"):
            if not pair.passes_filters:
                logging.debug("Skipping %s -> %s because passes_filters=False", pair.leader, pair.follower)
                return None

        leader_series = self._get_price_series(marketdata, pair.leader)
        follower_series = self._get_price_series(marketdata, pair.follower)

        if leader_series is None:
            logging.debug("Missing leader series: %s", pair.leader)
            return None

        if follower_series is None:
            logging.debug("Missing follower series: %s", pair.follower)
            return None

        leader_return = self._latest_return(leader_series)
        leader_zscore = self._latest_zscore(leader_series)
        follower_price = self._latest_price(follower_series)

        if follower_price is None:
            logging.debug("Missing follower price: %s", pair.follower)
            return None

        leader_direction = self._leader_signal_direction(leader_return, leader_zscore)
        if leader_direction == 0:
            logging.debug(
                "No leader signal for %s -> %s | return=%s | zscore=%s",
                pair.leader,
                pair.follower,
                leader_return,
                leader_zscore,
            )
            return None

        relationship_sign = self._relationship_sign(pair)

        # Map leader signal into follower action
        # Example:
        #   positive relation: leader long(+1) -> follower long(+1)
        #   negative relation: leader long(+1) -> follower short(-1)
        follower_direction = leader_direction * relationship_sign

        option = "buy" if follower_direction > 0 else "sell"
        confidence = self._compute_confidence(pair, leader_return, leader_zscore)

        if confidence < self.min_confidence_to_trade:
            logging.debug(
                "Skipping %s -> %s because confidence %.4f < %.4f",
                pair.leader,
                pair.follower,
                confidence,
                self.min_confidence_to_trade,
            )
            return None

        logging.info(
            "Leader signal found | leader=%s follower=%s leader_dir=%s relation=%s follower_action=%s return=%s zscore=%s price=%.4f confidence=%.4f lag=%s",
            pair.leader,
            pair.follower,
            "long" if leader_direction > 0 else "short",
            "positive" if relationship_sign > 0 else "negative",
            option,
            f"{leader_return:.6f}" if leader_return is not None else "None",
            f"{leader_zscore:.4f}" if leader_zscore is not None else "None",
            follower_price,
            confidence,
            getattr(pair, "best_lag", "N/A"),
        )

        return TradeRequest(
            symbol=pair.follower,
            option=option,
            quantity=self.default_quantity,
            price=float(follower_price),
            confidence=confidence,
        )

    @staticmethod
    def _aggregate_trade_requests(trade_requests: list[TradeRequest]) -> list[TradeRequest]:
        """
        If multiple pairs generate signals for the same follower ticker,
        combine them into one net request.
        """
        if not trade_requests:
            return []

        grouped: dict[str, list[TradeRequest]] = {}
        for req in trade_requests:
            grouped.setdefault(req.symbol, []).append(req)

        aggregated: list[TradeRequest] = []

        for symbol, reqs in grouped.items():
            net_score = 0.0
            total_qty = 0
            weighted_price_sum = 0.0
            total_conf = 0.0

            for req in reqs:
                signed_conf = req.confidence if req.option == "buy" else -req.confidence
                net_score += signed_conf
                total_qty += req.quantity
                weighted_price_sum += req.price * max(req.confidence, 1e-9)
                total_conf += req.confidence

            if abs(net_score) < 1e-12:
                logging.info("Signals canceled out for %s", symbol)
                continue

            option = "buy" if net_score > 0 else "sell"
            avg_price = weighted_price_sum / max(total_conf, 1e-9)
            avg_conf = min(abs(net_score), 1.0)

            aggregated.append(
                TradeRequest(
                    symbol=symbol,
                    option=option,
                    quantity=total_qty,
                    price=float(avg_price),
                    confidence=float(avg_conf),
                )
            )

        return aggregated

    def _analyze(
        self,
        news: MarketNews = None,
        marketData: pd.DataFrame = None,
    ) -> list[TradeRequest]:
        """
        Ignore news. Walk through all stored lead/follower relationships.
        If leader currently shows a long/short signal, trade the follower.
        """
        if marketData is None:
            logging.info("No marketData provided to LagTrader")
            return []

        marketdata = self._validate_marketdata(marketData)

        raw_requests: list[TradeRequest] = []
        for pair in self.lag_related_tickers:
            try:
                req = self._pair_to_trade_request(pair, marketdata)
                if req is not None:
                    raw_requests.append(req)
            except Exception as exc:
                logging.exception(
                    "Error evaluating pair %s -> %s: %s",
                    getattr(pair, "leader", "?"),
                    getattr(pair, "follower", "?"),
                    exc,
                )

        aggregated_requests = self._aggregate_trade_requests(raw_requests)

        logging.info(
            "LagTrader created %d raw requests and %d aggregated requests",
            len(raw_requests),
            len(aggregated_requests),
        )
        return aggregated_requests

    def trade(
        self,
        broker: Broker,
        news: MarketNews = None,
        marketData: pd.DataFrame = None,
    ):
        trade_requests = self._analyze(news=news, marketData=marketData)

        if not trade_requests:
            logging.info("No trade requests generated.")
            return []

        logging.info("Submitting %d trade requests to broker", len(trade_requests))

        try:
            result = broker.place_trade_requests(trade_requests)
            logging.info("Broker execution completed.")
            return result
        except AttributeError:
            logging.error("Broker object does not have place_trade_requests(trade_requests)")
            raise
        except Exception as exc:
            logging.exception("Broker trade execution failed: %s", exc)
            raise


if __name__ == "__main__":
    example_pairs = [
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

    mock_price_df = pd.DataFrame(
        {
            "AAPL": [250.0, 248.0, 246.5, 240.0],
            "XLE": [88.8, 89.0, 89.1, 89.3],
            "UUP": [28.0, 28.1, 28.2, 28.8],
            "MSFT": [428.0, 430.0, 431.0, 431.9],
        },
        index=pd.to_datetime(["2024-12-26", "2024-12-27", "2024-12-30", "2024-12-31"]),
    )

    trader = LagTrader(
        lag_related_tickers=example_pairs,
        default_quantity=1,
        min_confidence_to_trade=0.10,
        shock_threshold=0.003,
        zscore_threshold=0.8,
        zscore_window=3,
        require_passed_filters=True,
        use_zscore_if_available=True,
    )
