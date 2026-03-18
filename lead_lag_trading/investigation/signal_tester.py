from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from lead_lag_calc import LeadLagAnalyzer, LeadLagResult


@dataclass
class SignalTestSummary:
    leader: str
    follower: str
    lag: int
    pair_correlation: float
    pair_stability_mean: float
    pair_stability_std: float
    pair_sign_consistency: float
    pair_score: float
    pair_passes_filters: bool

    z_window: int
    threshold: float
    event_count: int
    long_event_count: int
    short_event_count: int

    avg_target_return_all: float
    avg_target_return_long: float
    avg_target_return_short: float

    median_target_return_all: float
    win_rate_all: float
    win_rate_long: float
    win_rate_short: float

    avg_strategy_return_all: float
    avg_strategy_return_long: float
    avg_strategy_return_short: float

    strategy_std_all: float
    strategy_sharpe_like: float


class SignalTester:
    """
    Tests lag-aware shock-response behavior for a leader/follower pair.

    Interpretation:
    - find a pair using LeadLagAnalyzer
    - if leader has a strong move at time t
    - ask what happens to follower at time t + lag
    """

    def __init__(self, analyzer: LeadLagAnalyzer) -> None:
        self.analyzer = analyzer

    @staticmethod
    def _validate_pair_result(pair_result: Optional[LeadLagResult]) -> LeadLagResult:
        if pair_result is None:
            raise ValueError("Pair result is None.")
        return pair_result

    def build_event_frame(
        self,
        leader: str,
        follower: str,
        lag: Optional[int] = None,
        max_lag: int = 20,
        z_window: int = 20,
        threshold: float = 1.5,
        require_passed_pair: bool = True,
        stability_window: int = 60,
    ) -> pd.DataFrame:
        """
        Builds an event-level DataFrame.

        Event logic:
        - compute z-score of leader return at time t
        - if |z| > threshold, create a signal at time t
        - evaluate follower return at time t + lag

        For positive relationship:
            z > threshold  -> long follower
            z < -threshold -> short follower

        For negative relationship:
            z > threshold  -> short follower
            z < -threshold -> long follower
        """
        if lag is None:
            pair_result = self.analyzer.analyze_pair(
                leader=leader,
                follower=follower,
                max_lag=max_lag,
                min_abs_corr=0.10,
                stability_window=stability_window,
                min_abs_stability_mean=0.05,
                max_stability_std=0.20,
                min_sign_consistency=0.60,
            )
        else:
            pair_result = self.analyzer.analyze_pair(
                leader=leader,
                follower=follower,
                max_lag=lag,
                min_abs_corr=0.0,
                stability_window=stability_window,
                min_abs_stability_mean=0.0,
                max_stability_std=np.inf,
                min_sign_consistency=0.0,
            )

        pair_result = self._validate_pair_result(pair_result)

        if require_passed_pair and not pair_result.passes_filters:
            raise ValueError(f"Pair {leader}->{follower} did not pass quality filters.")

        lag = pair_result.best_lag

        leader_return = self.analyzer.returns[leader].copy()
        follower_return = self.analyzer.returns[follower].copy()

        rolling_mean = leader_return.rolling(z_window).mean()
        rolling_std = leader_return.rolling(z_window).std()

        leader_zscore = (leader_return - rolling_mean) / rolling_std
        leader_zscore = leader_zscore.replace([np.inf, -np.inf], np.nan)

        # Future target at t + lag
        target_follower_return = follower_return.shift(-lag)

        # Relationship sign
        relation_sign = 1 if pair_result.correlation > 0 else -1

        signal = pd.Series(0.0, index=leader_return.index)

        # Positive relation:
        #   leader very positive => long follower
        #   leader very negative => short follower
        #
        # Negative relation:
        #   leader very positive => short follower
        #   leader very negative => long follower
        signal[leader_zscore > threshold] = relation_sign
        signal[leader_zscore < -threshold] = -relation_sign

        # Strategy return = signal * future follower return
        strategy_return = signal * target_follower_return

        out = pd.DataFrame(
            {
                "leader_return": leader_return,
                "leader_zscore": leader_zscore,
                "signal": signal,
                "follower_return_now": follower_return,
                "target_follower_return": target_follower_return,
                "strategy_return": strategy_return,
                "lag": float(lag),
                "pair_correlation": float(pair_result.correlation),
                "pair_stability_mean": float(pair_result.stability_mean),
                "pair_stability_std": float(pair_result.stability_std),
                "pair_sign_consistency": float(pair_result.sign_consistency),
                "pair_score": float(pair_result.score),
                "pair_passes_filters": bool(pair_result.passes_filters),
            }
        )

        out["event"] = out["signal"] != 0
        out["event_type"] = np.where(
            out["signal"] > 0,
            "long",
            np.where(out["signal"] < 0, "short", "none"),
        )

        return out

    @staticmethod
    def _safe_mean(series: pd.Series) -> float:
        if series.empty:
            return np.nan
        return float(series.mean())

    @staticmethod
    def _safe_median(series: pd.Series) -> float:
        if series.empty:
            return np.nan
        return float(series.median())

    @staticmethod
    def _safe_std(series: pd.Series) -> float:
        if series.empty:
            return np.nan
        return float(series.std())

    @staticmethod
    def _win_rate_target(target_returns: pd.Series, signals: pd.Series) -> float:
        """
        Win if the realized target return matches the signal direction.
        """
        if target_returns.empty:
            return np.nan

        wins = ((signals * target_returns) > 0).mean()
        return float(wins)

    def summarize_events(
        self,
        event_frame: pd.DataFrame,
        leader: str,
        follower: str,
    ) -> SignalTestSummary:
        events = event_frame[event_frame["event"]].dropna(subset=["target_follower_return", "strategy_return"])

        long_events = events[events["signal"] > 0]
        short_events = events[events["signal"] < 0]

        strategy_std_all = self._safe_std(events["strategy_return"])
        avg_strategy_return_all = self._safe_mean(events["strategy_return"])

        if pd.isna(strategy_std_all) or strategy_std_all == 0:
            sharpe_like = np.nan
        else:
            sharpe_like = float(avg_strategy_return_all / strategy_std_all)

        first = event_frame.iloc[0]

        return SignalTestSummary(
            leader=leader,
            follower=follower,
            lag=int(first["lag"]),
            pair_correlation=float(first["pair_correlation"]),
            pair_stability_mean=float(first["pair_stability_mean"]),
            pair_stability_std=float(first["pair_stability_std"]),
            pair_sign_consistency=float(first["pair_sign_consistency"]),
            pair_score=float(first["pair_score"]),
            pair_passes_filters=bool(first["pair_passes_filters"]),
            z_window=np.nan,  # filled by caller if needed
            threshold=np.nan,  # filled by caller if needed
            event_count=int(len(events)),
            long_event_count=int(len(long_events)),
            short_event_count=int(len(short_events)),
            avg_target_return_all=self._safe_mean(events["target_follower_return"]),
            avg_target_return_long=self._safe_mean(long_events["target_follower_return"]),
            avg_target_return_short=self._safe_mean(short_events["target_follower_return"]),
            median_target_return_all=self._safe_median(events["target_follower_return"]),
            win_rate_all=self._win_rate_target(events["target_follower_return"], events["signal"]),
            win_rate_long=self._win_rate_target(long_events["target_follower_return"], long_events["signal"]),
            win_rate_short=self._win_rate_target(short_events["target_follower_return"], short_events["signal"]),
            avg_strategy_return_all=avg_strategy_return_all,
            avg_strategy_return_long=self._safe_mean(long_events["strategy_return"]),
            avg_strategy_return_short=self._safe_mean(short_events["strategy_return"]),
            strategy_std_all=strategy_std_all,
            strategy_sharpe_like=sharpe_like,
        )

    def test_pair(
        self,
        leader: str,
        follower: str,
        lag: Optional[int] = None,
        max_lag: int = 20,
        z_window: int = 20,
        threshold: float = 1.5,
        require_passed_pair: bool = True,
        stability_window: int = 60,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
        - event-level DataFrame
        - one-row summary DataFrame
        """
        event_frame = self.build_event_frame(
            leader=leader,
            follower=follower,
            lag=lag,
            max_lag=max_lag,
            z_window=z_window,
            threshold=threshold,
            require_passed_pair=require_passed_pair,
            stability_window=stability_window,
        )

        summary = self.summarize_events(
            event_frame=event_frame,
            leader=leader,
            follower=follower,
        )

        summary.z_window = z_window
        summary.threshold = threshold

        summary_df = pd.DataFrame([asdict(summary)])
        return event_frame, summary_df

    def test_top_pairs(
        self,
        pairs_df: pd.DataFrame,
        z_window: int = 20,
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Run lag-aware signal tests on multiple discovered pairs.
        Expects columns:
            leader, follower, best_lag
        """
        summaries: list[pd.DataFrame] = []

        for _, row in pairs_df.iterrows():
            leader = row["leader"]
            follower = row["follower"]
            lag = int(row["best_lag"])

            try:
                _, summary_df = self.test_pair(
                    leader=leader,
                    follower=follower,
                    lag=lag,
                    z_window=z_window,
                    threshold=threshold,
                    require_passed_pair=False,
                )
                summaries.append(summary_df)
            except Exception as exc:
                print(f"Skipping {leader}->{follower}: {exc}")

        if not summaries:
            return pd.DataFrame()

        out = pd.concat(summaries, ignore_index=True)
        out = out.sort_values(
            by=["strategy_sharpe_like", "avg_strategy_return_all", "win_rate_all"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        return out
