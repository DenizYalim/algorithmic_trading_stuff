# backtest_main.py
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from GeneticAlgorithmForecast import GeneticAlgorithm, Individual  # adjust if your filenames differ


# -----------------------------
# Data
# -----------------------------

def fetch_close_series(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.Series:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} (period={period}, interval={interval})")
    if "Close" not in df.columns:
        raise RuntimeError(f"Expected 'Close' column, got: {list(df.columns)}")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]

    close = close.dropna().astype(float)
    if len(close) < 2:
        raise RuntimeError("Not enough price points after dropping NaNs.")
    close.name = "Close"
    return close


# -----------------------------
# Problem + Objective (linear model a*t + b)
# -----------------------------

@dataclass(frozen=True, slots=True)
class MarketForecastProblem:
    prices: Sequence[float]
    T: int

    def __post_init__(self) -> None:
        if len(self.prices) < 2:
            raise ValueError("prices must have length >= 2")
        if self.T <= 0:
            raise ValueError("T must be > 0")

    def forecast(self, genome: List[float]) -> float:
        a, b = float(genome[0]), float(genome[1])
        n = len(self.prices)
        t_future = (n - 1) + self.T
        return a * t_future + b


class MarketForecastObjective:
    def __init__(self, problem: MarketForecastProblem) -> None:
        self.problem = problem

    def mse(self, genome: List[float]) -> float:
        if not isinstance(genome, list) or len(genome) != 2:
            return float("inf")
        a, b = float(genome[0]), float(genome[1])
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("inf")

        err2 = 0.0
        for t, y in enumerate(self.problem.prices):
            y_hat = a * t + b
            d = float(y) - y_hat
            err2 += d * d
        return err2 / len(self.problem.prices)


# -----------------------------
# GA training helper
# -----------------------------

def train_linear_ga(
    train_prices: Sequence[float],
    T: int,
    *,
    pop_size: int,
    seconds: float,
    seed: Optional[int],
) -> Tuple[List[float], float]:
    problem = MarketForecastProblem(prices=train_prices, T=T)
    obj = MarketForecastObjective(problem)

    # GA maximizes => fitness = -MSE
    def fitness_fn(genome: List[float]) -> float:
        mse = obj.mse(genome)
        return -mse if math.isfinite(mse) else float("-inf")

    scale = max(1.0, float(max(train_prices)) - float(min(train_prices)))

    def create_genome(rng) -> List[float]:
        a = rng.uniform(-scale, scale)
        b = rng.uniform(min(train_prices) - scale, max(train_prices) + scale)
        return [a, b]

    def crossover_fn(g1: List[float], g2: List[float], rng) -> Tuple[List[float], List[float]]:
        alpha = rng.random()
        c1 = [alpha * g1[0] + (1 - alpha) * g2[0], alpha * g1[1] + (1 - alpha) * g2[1]]
        c2 = [(1 - alpha) * g1[0] + alpha * g2[0], (1 - alpha) * g1[1] + alpha * g2[1]]
        return c1, c2

    def mutate_fn(g: List[float], rng) -> List[float]:
        sigma = 0.05 * scale
        return [g[0] + rng.gauss(0.0, sigma), g[1] + rng.gauss(0.0, sigma)]

    ga = GeneticAlgorithm(
        pop_size=pop_size,
        create_genome=create_genome,
        fitness_fn=fitness_fn,
        crossover_fn=crossover_fn,
        mutate_fn=mutate_fn,
        mutation_rate=0.2,
        crossover_rate=0.9,
        elitism=2,
        tournament_k=3,
        maximize=True,
        seed=seed,
    )

    # requires your GA class to have run_for_seconds(seconds)
    best: Individual[List[float]] = ga.run_for_seconds(seconds)
    best_mse = obj.mse(best.genome)
    return best.genome, best_mse


# -----------------------------
# Backtest (walk-forward)
# -----------------------------

def backtest_walk_forward(
    close: pd.Series,
    *,
    T: int,
    train_window: int,
    stride: int,
    pop_size: int,
    seconds_per_fit: float,
    seed: Optional[int],
) -> pd.DataFrame:
    prices = close.to_numpy(dtype=float)
    dates = close.index

    rows = []
    # i = index of "today" (last point in training window)
    start_i = train_window - 1
    end_i = len(prices) - T - 1
    if end_i <= start_i:
        raise ValueError("Not enough data for the given train_window and T.")

    for i in range(start_i, end_i + 1, stride):
        train_slice = prices[i - train_window + 1 : i + 1]
        genome, fit_mse = train_linear_ga(
            train_slice,
            T,
            pop_size=pop_size,
            seconds=seconds_per_fit,
            seed=seed,
        )

        # forecast and compare
        prob = MarketForecastProblem(train_slice, T)
        y_pred = prob.forecast(genome)
        y_now = float(prices[i])
        y_true = float(prices[i + T])

        pred_return = y_pred - y_now
        true_return = y_true - y_now
        dir_ok = (pred_return >= 0 and true_return >= 0) or (pred_return < 0 and true_return < 0)

        err = y_pred - y_true

        rows.append(
            {
                "date": dates[i],
                "target_date": dates[i + T],
                "y_now": y_now,
                "y_true": y_true,
                "y_pred": y_pred,
                "error": err,
                "abs_error": abs(err),
                "squared_error": err * err,
                "direction_correct": int(dir_ok),
                "fit_mse_on_train": fit_mse,
                "a": genome[0],
                "b": genome[1],
            }
        )

    df = pd.DataFrame(rows).set_index("target_date").sort_index()
    return df


def print_metrics(df: pd.DataFrame) -> None:
    mae = float(df["abs_error"].mean())
    rmse = math.sqrt(float(df["squared_error"].mean()))
    mape = float((df["abs_error"] / df["y_true"].abs().replace(0.0, float("nan"))).mean() * 100.0)
    dir_acc = float(df["direction_correct"].mean() * 100.0)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Directional accuracy: {dir_acc:.2f}%")


# -----------------------------
# Plotting (matplotlib only; no custom colors)
# -----------------------------

def plot_predictions(df: pd.DataFrame, ticker: str, T: int) -> None:
    plt.figure()
    plt.plot(df.index, df["y_true"], label="Actual")
    plt.plot(df.index, df["y_pred"], label="Predicted")
    plt.title(f"{ticker} - Actual vs Predicted (T={T})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_abs_error(df: pd.DataFrame, ticker: str, T: int) -> None:
    plt.figure()
    plt.plot(df.index, df["abs_error"], label="Absolute error")
    plt.title(f"{ticker} - Absolute Error Over Time (T={T})")
    plt.xlabel("Date")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_hist(df: pd.DataFrame, ticker: str, T: int) -> None:
    plt.figure()
    plt.hist(df["error"].to_numpy(), bins=30)
    plt.title(f"{ticker} - Error Distribution (T={T})")
    plt.xlabel("Prediction error (pred - true)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="6mo")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--train_window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)

    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--seconds_per_fit", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    close = fetch_close_series(args.ticker, period=args.period, interval=args.interval)

    df = backtest_walk_forward(
        close,
        T=args.T,
        train_window=args.train_window,
        stride=args.stride,
        pop_size=args.pop,
        seconds_per_fit=args.seconds_per_fit,
        seed=args.seed,
    )

    print(f"ticker={args.ticker} points={len(close)} period={args.period} interval={args.interval}")
    print(f"backtest_points={len(df)} T={args.T} train_window={args.train_window} stride={args.stride}")
    print_metrics(df)

    if args.plot:
        plot_predictions(df, args.ticker, args.T)
        plot_abs_error(df, args.ticker, args.T)
        plot_error_hist(df, args.ticker, args.T)


if __name__ == "__main__":
    main()
