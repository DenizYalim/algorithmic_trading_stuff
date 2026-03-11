from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Sequence, TypeVar
import math

C = TypeVar("C")  # chromosome / candidate type


class Problem(ABC, Generic[C]):
    """
    Generic optimization problem.
    Your GA should use:
      - random_candidate(): create a candidate
      - (optional) repair()/is_valid(): keep candidates feasible
    """

    @abstractmethod
    def random_candidate(self) -> C:
        raise NotImplementedError

    def repair(self, candidate: C) -> C:
        return candidate

    def is_valid(self, candidate: C) -> bool:
        return True


class ObjectiveFunction(ABC, Generic[C]):
    def __init__(self, problem: Problem[C]) -> None:
        self.problem = problem

    @abstractmethod
    def evaluate(self, candidate: C) -> float:
        """Return fitness (maximize) or cost (minimize) depending on your GA convention."""
        raise NotImplementedError

    def __call__(self, candidate: C) -> float:
        return self.evaluate(candidate)


# --- Concrete example: market forecasting as parameter fitting ---

@dataclass(frozen=True, slots=True)
class MarketForecastProblem(Problem[List[float]]):
    """
    Fit a simple model to recent prices and forecast T steps ahead.
    Candidate/chromosome: [a, b] for a linear model: price(t) = a*t + b
      - We evaluate candidate by how well it fits history (lower error is better).
    """

    prices: Sequence[float]   # length n
    T: int                    # forecast horizon (steps ahead)

    def __post_init__(self) -> None:
        if len(self.prices) < 2:
            raise ValueError("prices must have length >= 2")
        if self.T <= 0:
            raise ValueError("T must be > 0")

    def random_candidate(self) -> List[float]:
        # NOTE: problem itself should not own RNG if you want deterministic runs.
        # Keep this simple; GA can also generate candidates.
        # Default: random a,b in a reasonable range based on data scale.
        p0, p1 = float(self.prices[0]), float(self.prices[-1])
        scale = max(1.0, abs(p1 - p0))
        # candidate = [a, b]
        import random
        a = random.uniform(-scale, scale)
        b = random.uniform(min(self.prices) - scale, max(self.prices) + scale)
        return [a, b]

    def is_valid(self, candidate: List[float]) -> bool:
        return (
            isinstance(candidate, list)
            and len(candidate) == 2
            and all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in candidate)
        )

    def forecast(self, candidate: List[float]) -> float:
        """Forecast price at time index (n-1 + T) using the candidate model."""
        a, b = float(candidate[0]), float(candidate[1])
        n = len(self.prices)
        t_future = (n - 1) + self.T
        return a * t_future + b


class MarketForecastObjective(ObjectiveFunction[List[float]]):
    """
    Minimization objective: mean squared error (MSE) on history.
    If your GA MAXIMIZES fitness, use fitness = -MSE.
    """

    def evaluate(self, candidate: List[float]) -> float:
        prob = self.problem
        if not prob.is_valid(candidate):
            return float("inf")  # or a big penalty

        a, b = float(candidate[0]), float(candidate[1])
        prices = prob.prices

        # fit quality on history
        err2_sum = 0.0
        for t, y in enumerate(prices):
            y_hat = a * t + b
            d = float(y) - y_hat
            err2_sum += d * d

        mse = err2_sum / len(prices)
        return mse
