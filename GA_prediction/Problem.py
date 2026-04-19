from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from GA_prediction.Representation import GARepresentation
from GA_prediction.Solution import GASolution


@dataclass
class MarketPredictionProblem:
    """
    Converts historical prices into a supervised prediction problem.

    The GA is rewarded for producing useful long/short/flat signals, not just
    directional accuracy. Fitness is average strategy return with a small
    penalty for excessive trading.
    """

    market_data: pd.DataFrame
    ticker: str
    horizon_days: int = 1
    minimum_return: float = 0.0
    trading_cost: float = 0.0005
    validation_fraction: float = 0.25

    def __post_init__(self):
        self.prices = self.extract_price_series(self.market_data, self.ticker)
        self.feature_frame = self._build_feature_frame(self.prices)
        self.dataset = self._build_labeled_dataset(self.feature_frame, self.prices)
        self.feature_columns = list(self.feature_frame.columns)

        split_index = max(1, int(len(self.dataset) * (1.0 - self.validation_fraction)))
        self.train_data = self.dataset.iloc[:split_index].copy()
        self.validation_data = self.dataset.iloc[split_index:].copy()
        if self.validation_data.empty:
            self.validation_data = self.train_data

        self._feature_mean = self.train_data[self.feature_columns].mean()
        self._feature_std = self.train_data[self.feature_columns].std().replace(0.0, 1.0)

    @staticmethod
    def extract_price_series(market_data: pd.DataFrame, ticker: str) -> pd.Series:
        if market_data is None or not isinstance(market_data, pd.DataFrame) or market_data.empty:
            raise ValueError("market_data must be a non-empty pandas DataFrame")

        df = market_data.copy().sort_index()

        if isinstance(df.columns, pd.MultiIndex):
            for key in [(ticker, "Close"), ("Close", ticker), (ticker, "Adj Close"), ("Adj Close", ticker)]:
                if key in df.columns:
                    return df[key].dropna().astype(float)

        if ticker in df.columns:
            selected = df[ticker]
            if isinstance(selected, pd.DataFrame):
                selected = selected.iloc[:, 0]
            return selected.dropna().astype(float)

        for col in ["Close", "Adj Close", "close", "price"]:
            if col in df.columns:
                selected = df[col]
                if isinstance(selected, pd.DataFrame):
                    selected = selected.iloc[:, 0]
                return selected.dropna().astype(float)

        if df.shape[1] == 1:
            return df.iloc[:, 0].dropna().astype(float)

        raise ValueError(f"Could not find a price column for ticker {ticker}")

    def _build_feature_frame(self, prices: pd.Series) -> pd.DataFrame:
        returns = prices.pct_change()

        features = pd.DataFrame(index=prices.index)
        features["return_1"] = returns
        features["return_2"] = prices.pct_change(2)
        features["momentum_3"] = prices.pct_change(3)
        features["momentum_5"] = prices.pct_change(5)
        features["momentum_10"] = prices.pct_change(10)
        features["sma_ratio_5"] = prices / prices.rolling(5).mean() - 1.0
        features["sma_ratio_10"] = prices / prices.rolling(10).mean() - 1.0
        features["volatility_5"] = returns.rolling(5).std()
        features["volatility_10"] = returns.rolling(10).std()
        features["drawdown_10"] = prices / prices.rolling(10).max() - 1.0

        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        if len(features) < 20:
            raise ValueError("Not enough market data to build GA features; need at least 20 usable rows")

        return features

    def _build_labeled_dataset(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
    ) -> pd.DataFrame:
        dataset = features.copy()
        dataset["future_return"] = prices.pct_change(self.horizon_days).shift(-self.horizon_days)

        dataset["target"] = 0
        dataset.loc[dataset["future_return"] > self.minimum_return, "target"] = 1
        dataset.loc[dataset["future_return"] < -self.minimum_return, "target"] = -1

        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
        if len(dataset) < 20:
            raise ValueError("Not enough market data to train GA predictor; need at least 20 usable rows")

        return dataset

    @property
    def feature_count(self) -> int:
        return len(self.feature_columns)

    def _normalized_features(self, data: pd.DataFrame) -> np.ndarray:
        normalized = (data[self.feature_columns] - self._feature_mean) / self._feature_std
        return normalized.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float)

    def evaluate(
        self,
        representation: GARepresentation,
        data: Optional[pd.DataFrame] = None,
    ) -> GASolution:
        if data is None:
            data = self.train_data

        features = self._normalized_features(data)
        predictions = representation.predict(features)
        future_returns = data["future_return"].to_numpy(dtype=float)
        targets = data["target"].to_numpy(dtype=int)

        trade_changes = np.abs(np.diff(predictions, prepend=0)) > 0
        strategy_returns = predictions * future_returns
        strategy_returns = strategy_returns - (trade_changes.astype(float) * self.trading_cost)

        active_mask = predictions != 0
        accuracy = 0.0
        if active_mask.any():
            accuracy = float(np.mean(predictions[active_mask] == targets[active_mask]))

        coverage = float(np.mean(active_mask))
        mean_return = float(np.mean(strategy_returns))
        downside = strategy_returns[strategy_returns < 0.0]
        downside_penalty = float(abs(np.mean(downside))) if len(downside) else 0.0
        fitness = mean_return - 0.25 * downside_penalty

        return GASolution(
            representation=representation,
            fitness=float(fitness),
            metrics={
                "accuracy": accuracy,
                "coverage": coverage,
                "mean_strategy_return": mean_return,
                "trades": int(trade_changes.sum()),
                "rows": int(len(data)),
            },
        )

    def latest_features(self) -> np.ndarray:
        latest = self.feature_frame.iloc[[-1]]
        return self._normalized_features(latest)

    def latest_price(self) -> float:
        return float(self.prices.iloc[-1])

    def latest_features_from_market_data(self, market_data: pd.DataFrame) -> np.ndarray:
        prices = self.extract_price_series(market_data, self.ticker)
        features = self._build_feature_frame(prices)
        latest = features[self.feature_columns].iloc[[-1]]
        return self._normalized_features(latest)

    def latest_price_from_market_data(self, market_data: pd.DataFrame) -> float:
        prices = self.extract_price_series(market_data, self.ticker)
        return float(prices.iloc[-1])
