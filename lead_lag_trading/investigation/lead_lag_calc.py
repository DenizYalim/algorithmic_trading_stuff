from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import itertools

import numpy as np
import pandas as pd


"""
Correlation intuition:
0.00–0.05  -> noise
0.05–0.10  -> weak
0.10–0.20  -> borderline
0.20+      -> interesting
0.30+      -> strong (rare)

Important:
- A pair should not be accepted only because of one decent full-sample correlation.
- Rolling stability should be checked.
- A stable relationship should not spend most of its time near 0 or constantly flip sign.
"""


@dataclass
class LeadLagResult:
    leader: str
    follower: str
    best_lag: int
    correlation: float
    r_squared: float
    observations: int
    direction: str  # "positive" or "negative"

    stability_mean: float
    stability_std: float
    sign_consistency: float
    stability_score: float

    score: float
    passes_filters: bool


class LeadLagAnalyzer:
    """
    Research tool for discovering lead-lag relationships across assets.

    Expected input:
        - rows: datetime index
        - columns: asset tickers
        - values: prices

    Method:
        - converts prices to returns
        - tests whether returns of asset A at time t
          relate to returns of asset B at time t + lag
        - scans over a lag window
        - evaluates rolling stability for the best lag
        - ranks results using both strength and stability

    Notes:
        - This is a relationship discovery tool.
        - It does NOT prove tradability.
        - It does NOT include transaction costs, slippage, or execution logic.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        return_method: str = "log",
        min_obs: int = 120,
    ) -> None:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")

        if prices.shape[1] < 2:
            raise ValueError("prices must contain at least 2 asset columns")

        if return_method not in {"log", "pct"}:
            raise ValueError("return_method must be either 'log' or 'pct'")

        self.prices = prices.sort_index().copy()
        self.return_method = return_method
        self.min_obs = min_obs
        self.returns = self._compute_returns(self.prices, self.return_method)

    @staticmethod
    def _compute_returns(prices: pd.DataFrame, method: str) -> pd.DataFrame:
        clean = prices.copy().astype(float)
        clean = clean.replace([np.inf, -np.inf], np.nan)

        if method == "log":
            returns = np.log(clean / clean.shift(1))
        else:
            returns = clean.pct_change()

        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna(how="all")

        return returns

    @staticmethod
    def _aligned_pair(
        leader_series: pd.Series,
        follower_series: pd.Series,
        lag: int,
    ) -> pd.DataFrame:
        """
        If lag = 3:
            leader[t] is compared to follower[t + 3]
        Meaning leader may lead follower by 3 periods.
        """
        if lag < 1:
            raise ValueError("lag must be >= 1")

        df = pd.concat(
            [
                leader_series.rename("leader"),
                follower_series.shift(-lag).rename("follower"),
            ],
            axis=1,
        ).dropna()

        return df

    @staticmethod
    def _pair_stats(df: pd.DataFrame) -> tuple[float, float, int]:
        """
        Returns:
            correlation, r_squared, n_obs
        """
        n_obs = len(df)
        if n_obs < 2:
            return np.nan, np.nan, n_obs

        corr = df["leader"].corr(df["follower"])
        if pd.isna(corr):
            return np.nan, np.nan, n_obs

        r2 = float(corr**2)
        return float(corr), r2, n_obs

    @staticmethod
    def _safe_std(series: pd.Series) -> float:
        val = float(series.std())
        if np.isnan(val):
            return np.nan
        return val

    def rolling_pair_score(
        self,
        leader: str,
        follower: str,
        lag: int,
        window: int = 60,
    ) -> pd.Series:
        """
        Rolling correlation time series for a chosen leader/follower/lag.
        Used to measure relationship stability through time.
        """
        aligned = self._aligned_pair(
            self.returns[leader],
            self.returns[follower],
            lag=lag,
        )

        rolling_corr = aligned["leader"].rolling(window).corr(aligned["follower"])
        return rolling_corr.rename(f"{leader}_leads_{follower}_lag{lag}")

    def _rolling_stability_metrics(
        self,
        leader: str,
        follower: str,
        lag: int,
        base_corr: float,
        stability_window: int = 60,
    ) -> tuple[float, float, float, float]:
        """
        Returns:
            stability_mean
            stability_std
            sign_consistency
            stability_score

        sign_consistency:
            fraction of rolling windows whose sign matches the full-sample correlation sign

        stability_score:
            rewards larger stable rolling correlation and penalizes noisy flipping behavior
        """
        rolling = self.rolling_pair_score(
            leader=leader,
            follower=follower,
            lag=lag,
            window=stability_window,
        ).dropna()

        if rolling.empty:
            return np.nan, np.nan, np.nan, -np.inf

        stability_mean = float(rolling.mean())
        stability_std = self._safe_std(rolling)

        expected_sign = np.sign(base_corr)
        if expected_sign == 0:
            sign_consistency = 0.0
        else:
            sign_consistency = float((np.sign(rolling) == expected_sign).mean())

        # Penalize mean close to 0, unstable std, and low sign consistency.
        if np.isnan(stability_std):
            stability_score = -np.inf
        else:
            stability_score = abs(stability_mean) * sign_consistency / (stability_std + 1e-6)

        return stability_mean, stability_std, sign_consistency, float(stability_score)

    @staticmethod
    def _composite_score(
        corr: float,
        n_obs: int,
        stability_mean: float,
        stability_std: float,
        sign_consistency: float,
    ) -> float:
        """
        Composite ranking score:
        - wants meaningful correlation
        - wants larger sample size
        - wants stable rolling behavior
        - wants sign consistency

        Higher is better.
        """
        if any(pd.isna(x) for x in [corr, stability_mean, stability_std, sign_consistency]):
            return -np.inf

        obs_bonus = np.log(max(n_obs, 2))
        stability_penalty = stability_std + 1e-6

        score = abs(corr) * obs_bonus * max(abs(stability_mean), 1e-6) * max(sign_consistency, 1e-6) / stability_penalty
        return float(score)

    @staticmethod
    def _passes_thresholds(
        corr: float,
        stability_mean: float,
        stability_std: float,
        sign_consistency: float,
        min_abs_corr: float,
        min_abs_stability_mean: float,
        max_stability_std: float,
        min_sign_consistency: float,
    ) -> bool:
        if pd.isna(corr) or pd.isna(stability_mean) or pd.isna(stability_std) or pd.isna(sign_consistency):
            return False

        if abs(corr) < min_abs_corr:
            return False

        if abs(stability_mean) < min_abs_stability_mean:
            return False

        if stability_std > max_stability_std:
            return False

        if sign_consistency < min_sign_consistency:
            return False

        return True

    def analyze_pair(
        self,
        leader: str,
        follower: str,
        max_lag: int = 20,
        min_abs_corr: float = 0.10,
        stability_window: int = 60,
        min_abs_stability_mean: float = 0.05,
        max_stability_std: float = 0.20,
        min_sign_consistency: float = 0.60,
    ) -> Optional[LeadLagResult]:
        """
        Finds the best lag for one ordered pair:
            leader -> follower

        Selection is based on composite score, not just raw correlation.
        """
        if leader == follower:
            return None

        if leader not in self.returns.columns:
            raise KeyError(f"Unknown leader asset: {leader}")

        if follower not in self.returns.columns:
            raise KeyError(f"Unknown follower asset: {follower}")

        leader_series = self.returns[leader]
        follower_series = self.returns[follower]

        best_result: Optional[LeadLagResult] = None
        best_score = -np.inf

        for lag in range(1, max_lag + 1):
            aligned = self._aligned_pair(leader_series, follower_series, lag)
            corr, r2, n_obs = self._pair_stats(aligned)

            if n_obs < self.min_obs or pd.isna(corr):
                continue

            stability_mean, stability_std, sign_consistency, stability_score = self._rolling_stability_metrics(
                leader=leader,
                follower=follower,
                lag=lag,
                base_corr=corr,
                stability_window=stability_window,
            )

            score = self._composite_score(
                corr=corr,
                n_obs=n_obs,
                stability_mean=stability_mean,
                stability_std=stability_std,
                sign_consistency=sign_consistency,
            )

            passes_filters = self._passes_thresholds(
                corr=corr,
                stability_mean=stability_mean,
                stability_std=stability_std,
                sign_consistency=sign_consistency,
                min_abs_corr=min_abs_corr,
                min_abs_stability_mean=min_abs_stability_mean,
                max_stability_std=max_stability_std,
                min_sign_consistency=min_sign_consistency,
            )

            # Keep the best candidate even if it fails, but mark it as failing.
            # This is useful for inspection/debugging.
            if score > best_score:
                best_score = score
                best_result = LeadLagResult(
                    leader=leader,
                    follower=follower,
                    best_lag=lag,
                    correlation=float(corr),
                    r_squared=float(r2),
                    observations=int(n_obs),
                    direction="positive" if corr > 0 else "negative",
                    stability_mean=float(stability_mean),
                    stability_std=float(stability_std),
                    sign_consistency=float(sign_consistency),
                    stability_score=float(stability_score),
                    score=float(score),
                    passes_filters=bool(passes_filters),
                )

        return best_result

    def analyze_universe(
        self,
        max_lag: int = 20,
        min_abs_corr: float = 0.10,
        stability_window: int = 60,
        min_abs_stability_mean: float = 0.05,
        max_stability_std: float = 0.20,
        min_sign_consistency: float = 0.60,
        only_passed: bool = True,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Tests all ordered asset pairs:
            A -> B and B -> A are treated separately

        Returns a DataFrame ranked by composite score.
        """
        results: list[LeadLagResult] = []

        for leader, follower in itertools.permutations(self.returns.columns, 2):
            result = self.analyze_pair(
                leader=leader,
                follower=follower,
                max_lag=max_lag,
                min_abs_corr=min_abs_corr,
                stability_window=stability_window,
                min_abs_stability_mean=min_abs_stability_mean,
                max_stability_std=max_stability_std,
                min_sign_consistency=min_sign_consistency,
            )
            if result is not None:
                results.append(result)

        if not results:
            return pd.DataFrame(
                columns=[
                    "leader",
                    "follower",
                    "best_lag",
                    "correlation",
                    "r_squared",
                    "observations",
                    "direction",
                    "stability_mean",
                    "stability_std",
                    "sign_consistency",
                    "stability_score",
                    "score",
                    "passes_filters",
                ]
            )

        df = pd.DataFrame([asdict(r) for r in results])

        if only_passed:
            df = df[df["passes_filters"]].copy()

        if df.empty:
            return df.reset_index(drop=True)

        df = df.sort_values(
            by=["score", "correlation", "stability_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        if top_n is not None:
            df = df.head(top_n).reset_index(drop=True)

        return df

    def pair_diagnostics(
        self,
        leader: str,
        follower: str,
        lag: Optional[int] = None,
        max_lag: int = 20,
        stability_window: int = 60,
    ) -> dict:
        """
        Convenience method for inspecting one pair in more detail.
        """
        if lag is None:
            pair_result = self.analyze_pair(
                leader=leader,
                follower=follower,
                max_lag=max_lag,
                min_abs_corr=0.0,
                stability_window=stability_window,
                min_abs_stability_mean=0.0,
                max_stability_std=np.inf,
                min_sign_consistency=0.0,
            )
            if pair_result is None:
                raise ValueError("Pair could not be analyzed")
            lag = pair_result.best_lag
        else:
            pair_result = self.analyze_pair(
                leader=leader,
                follower=follower,
                max_lag=lag,
                min_abs_corr=0.0,
                stability_window=stability_window,
                min_abs_stability_mean=0.0,
                max_stability_std=np.inf,
                min_sign_consistency=0.0,
            )

        rolling = self.rolling_pair_score(
            leader=leader,
            follower=follower,
            lag=lag,
            window=stability_window,
        )

        return {
            "result": pair_result,
            "rolling_corr": rolling,
        }

    def simple_signal_from_pair(
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
        Very simple research signal:
        - chooses a lagged relationship
        - z-scores the leader return
        - uses the detected direction
        - outputs a research-only long/short indicator for follower

        This is NOT production trading logic.

        Important:
        - This should only be used after relationship filtering.
        """
        if lag is None:
            pair_result = self.analyze_pair(
                leader=leader,
                follower=follower,
                max_lag=max_lag,
                min_abs_corr=0.10,
                stability_window=stability_window,
                min_abs_stability_mean=0.05,
                max_stability_std=0.20,
                min_sign_consistency=0.60,
            )
            if pair_result is None:
                raise ValueError("Pair could not be analyzed")
            lag = pair_result.best_lag
        else:
            pair_result = self.analyze_pair(
                leader=leader,
                follower=follower,
                max_lag=lag,
                min_abs_corr=0.0,
                stability_window=stability_window,
                min_abs_stability_mean=0.0,
                max_stability_std=np.inf,
                min_sign_consistency=0.0,
            )

        if pair_result is None:
            raise ValueError("No valid pair result found")

        if require_passed_pair and not pair_result.passes_filters:
            raise ValueError(f"Pair {leader}->{follower} did not pass quality filters. " "Refusing to generate a research signal.")

        sign = 1 if pair_result.correlation > 0 else -1

        leader_return = self.returns[leader]
        follower_return = self.returns[follower]

        # The signal is formed from leader information available at time t.
        rolling_mean = leader_return.rolling(z_window).mean()
        rolling_std = leader_return.rolling(z_window).std()

        z = (leader_return - rolling_mean) / rolling_std
        z = z.replace([np.inf, -np.inf], np.nan)

        signal = pd.Series(0.0, index=z.index)

        # For positive relationship:
        # strong positive leader shock -> long follower
        # strong negative leader shock -> short follower
        #
        # For negative relationship, sign flips.
        signal[z > threshold] = sign
        signal[z < -threshold] = -sign

        out = pd.DataFrame(
            {
                "leader_return": leader_return,
                "follower_return": follower_return,
                "leader_zscore": z,
                "signal_for_follower": signal,
                "detected_lag": float(lag),
                "pair_correlation": float(pair_result.correlation),
                "pair_stability_mean": float(pair_result.stability_mean),
                "pair_stability_std": float(pair_result.stability_std),
                "pair_sign_consistency": float(pair_result.sign_consistency),
                "pair_score": float(pair_result.score),
                "pair_passes_filters": bool(pair_result.passes_filters),
            }
        )

        return out
