# main.py
from __future__ import annotations

import argparse
import math
from typing import List, Tuple
import yfinance as yf
import pandas as pd

from Problem import MarketForecastProblem, MarketForecastObjective  # adjust import
from GeneticAlgorithmForecast import GeneticAlgorithm  # adjust import



def fetch_close_prices(ticker: str, period: str = "6mo", interval: str = "1d") -> List[float]:
    df = yf.download(
        ticker,                 # use positional, not tickers=
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",      # makes columns more predictable
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} (period={period}, interval={interval})")

    if "Close" not in df.columns:
        raise RuntimeError(f"Expected 'Close' column, got: {list(df.columns)}")

    close = df["Close"]

    # If it's a DataFrame (multi-index / multiple columns), pick the ticker column
    if isinstance(close, pd.DataFrame):
        # common shape: columns are tickers; try direct ticker match first
        if ticker in close.columns:
            close = close[ticker]
        else:
            # fallback: first column
            close = close.iloc[:, 0]

    prices = close.dropna().astype(float).to_list()
    if len(prices) < 2:
        raise RuntimeError("Not enough price points after dropping NaNs.")
    return prices


def make_linear_ga(problem: MarketForecastProblem, *, pop_size: int, generations: int, seed: int | None):
    obj = MarketForecastObjective(problem)

    # GA is configured to MAXIMIZE fitness, so use fitness = -MSE
    def fitness_fn(genome: List[float]) -> float:
        mse = obj.evaluate(genome)
        if not math.isfinite(mse):
            return float("-inf")
        return -mse

    # genome = [a, b] for price(t) = a*t + b
    scale = max(1.0, float(max(problem.prices)) - float(min(problem.prices)))

    def create_genome(rng) -> List[float]:
        a = rng.uniform(-scale, scale)
        b = rng.uniform(min(problem.prices) - scale, max(problem.prices) + scale)
        return [a, b]

    def crossover_fn(g1: List[float], g2: List[float], rng) -> Tuple[List[float], List[float]]:
        alpha = rng.random()
        c1 = [alpha * g1[0] + (1 - alpha) * g2[0], alpha * g1[1] + (1 - alpha) * g2[1]]
        c2 = [(1 - alpha) * g1[0] + alpha * g2[0], (1 - alpha) * g1[1] + alpha * g2[1]]
        return c1, c2

    def mutate_fn(g: List[float], rng) -> List[float]:
        # gaussian noise
        sigma_a = 0.05 * scale
        sigma_b = 0.05 * scale
        return [g[0] + rng.gauss(0.0, sigma_a), g[1] + rng.gauss(0.0, sigma_b)]

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
    print("starting GA")
    best = ga.run_for_seconds(60.0)
    return best, obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="6mo")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--T", type=int, default=5, help="forecast horizon in steps")
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--gens", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prices = fetch_close_prices(args.ticker, period=args.period, interval=args.interval)

    problem = MarketForecastProblem(prices=prices, T=args.T)
    best_ind, obj = make_linear_ga(problem, pop_size=args.pop, generations=args.gens, seed=args.seed)

    best_genome = best_ind.genome
    best_mse = obj.evaluate(best_genome)
    forecast_price = problem.forecast(best_genome)

    print(f"ticker={args.ticker} points={len(prices)} period={args.period} interval={args.interval}")
    print(f"best genome [a,b] = {best_genome}")
    print(f"best MSE (lower is better) = {best_mse:.6f}")
    print(f"last observed price = {prices[-1]:.4f}")
    print(f"forecast price at T={args.T} = {forecast_price:.4f}")


if __name__ == "__main__":
    main()
