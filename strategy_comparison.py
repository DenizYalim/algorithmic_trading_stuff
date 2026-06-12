from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from contextlib import redirect_stdout
from io import StringIO
from typing import Callable, Optional

import pandas as pd

from backtester import Backtester
from utility.broker_apis.broker_ABS import SimulatedBroker, Broker
from utility.market_data.price_data_provider import PriceDataProvider
from utility.market_news.market_news_provider_ABS import NewsProviderABS
from utility.trader_ABS import Trader


@dataclass
class StrategySpec:
    """
    Configuration for one strategy comparison run.

    `trader_factory` is used so each strategy gets a fresh trader instance.
    """

    name: str
    trader_factory: Callable[[], Trader]
    use_news: bool = False
    use_market_data: bool = True
    lookback_days: int = 30
    market_data_tickers: Optional[list[str]] = None
    max_news: int = 1000
    metadata: dict = field(default_factory=dict)


class StrategyComparator:
    def __init__(
        self,
        price_provider: PriceDataProvider,
        news_provider: Optional[NewsProviderABS] = None,
        broker_factory: Optional[Callable[[], Broker]] = None,
        suppress_backtester_output: bool = True,
    ):
        self.price_provider = price_provider
        self.news_provider = news_provider
        self.broker_factory = broker_factory or (lambda: SimulatedBroker())
        self.suppress_backtester_output = suppress_backtester_output

    def run_strategy(
        self,
        strategy: StrategySpec,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        trader = strategy.trader_factory()
        broker = self.broker_factory()
        backtester = Backtester(
            trader=trader,
            broker=broker,
            news_provider=self.news_provider,
            price_provider=self.price_provider,
        )

        if self.suppress_backtester_output:
            with redirect_stdout(StringIO()):
                result = backtester.run_backtest(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    use_news=strategy.use_news,
                    use_market_data=strategy.use_market_data,
                    lookback_days=strategy.lookback_days,
                    market_data_tickers=strategy.market_data_tickers,
                    max_news=strategy.max_news,
                )
        else:
            result = backtester.run_backtest(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                use_news=strategy.use_news,
                use_market_data=strategy.use_market_data,
                lookback_days=strategy.lookback_days,
                market_data_tickers=strategy.market_data_tickers,
                max_news=strategy.max_news,
            )

        initial_cash = float(getattr(broker, "initial_cash", getattr(broker, "cash", 0.0)))
        if initial_cash == 0.0 and "final_value" in result and "profit" in result:
            initial_cash = float(result["final_value"] - result["profit"])

        profit = float(result["profit"])
        final_value = float(result["final_value"])
        trade_history = list(result.get("trade_history", []))
        return_pct = profit / initial_cash * 100.0 if initial_cash else 0.0

        payload = {
            "strategy": strategy.name,
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "final_value": final_value,
            "profit": profit,
            "return_pct": return_pct,
            "trade_count": len(trade_history),
            "trade_history": trade_history,
        }
        payload.update(strategy.metadata)
        return payload

    def compare(
        self,
        strategies: list[StrategySpec],
        ticker: str,
        start_date: str,
        end_date: str,
        sort_by: str = "final_value",
        ascending: bool = False,
    ) -> pd.DataFrame:
        rows = [
            self.run_strategy(
                strategy=strategy,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            for strategy in strategies
        ]

        comparison = pd.DataFrame(rows)
        if comparison.empty:
            return comparison

        return comparison.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)


def compare_strategies(
    strategies: list[StrategySpec],
    ticker: str,
    start_date: str,
    end_date: str,
    price_provider: PriceDataProvider,
    news_provider: Optional[NewsProviderABS] = None,
    broker_factory: Optional[Callable[[], Broker]] = None,
    sort_by: str = "final_value",
    ascending: bool = False,
    suppress_backtester_output: bool = True,
) -> pd.DataFrame:
    comparator = StrategyComparator(
        price_provider=price_provider,
        news_provider=news_provider,
        broker_factory=broker_factory,
        suppress_backtester_output=suppress_backtester_output,
    )
    return comparator.compare(
        strategies=strategies,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_by,
        ascending=ascending,
    )


def build_standard_strategy_specs(
    ticker: str,
    strategy_names: list[str],
    ga_population_size: int = 24,
    ga_generations: int = 10,
    ga_random_seed: int = 11,
    ga_min_training_rows: int = 90,
    ga_retrain_every_rows: int = 45,
    ga_min_confidence_to_trade: float = 0.50,
    momentum_short_window: int = 10,
    momentum_long_window: int = 30,
    momentum_min_confidence_to_trade: float = 0.05,
    regex_confidence_needed: float = 0.05,
    regex_max_news: int = 200,
) -> list[StrategySpec]:
    normalized = [name.strip().lower() for name in strategy_names]
    specs: list[StrategySpec] = []

    for name in normalized:
        if name == "ga":
            from GA_prediction import GATrader, GeneticAlgorithmConfig

            specs.append(
                StrategySpec(
                    name="ga",
                    trader_factory=lambda t=ticker: GATrader(
                        ticker=t,
                        min_training_rows=ga_min_training_rows,
                        min_confidence_to_trade=ga_min_confidence_to_trade,
                        retrain_every_rows=ga_retrain_every_rows,
                        ga_config=GeneticAlgorithmConfig(
                            population_size=ga_population_size,
                            generations=ga_generations,
                            random_seed=ga_random_seed,
                        ),
                    ),
                    use_news=False,
                    use_market_data=True,
                    lookback_days=ga_min_training_rows,
                    metadata={"category": "market_data"},
                )
            )
        elif name == "momentum":
            from momentum_trading import MomentumTrader

            specs.append(
                StrategySpec(
                    name=f"momentum_{momentum_short_window}_{momentum_long_window}",
                    trader_factory=lambda t=ticker: MomentumTrader(
                        ticker=t,
                        short_window=momentum_short_window,
                        long_window=momentum_long_window,
                        min_confidence_to_trade=momentum_min_confidence_to_trade,
                    ),
                    use_news=False,
                    use_market_data=True,
                    lookback_days=momentum_long_window,
                    metadata={"category": "market_data"},
                )
            )
        elif name == "regex":
            from regex_news_trading.regex_trader import RegexTrader

            specs.append(
                StrategySpec(
                    name="regex_news",
                    trader_factory=lambda t=ticker: RegexTrader(
                        ticker=t,
                        confidence_needed=regex_confidence_needed,
                    ),
                    use_news=True,
                    use_market_data=False,
                    lookback_days=1,
                    max_news=regex_max_news,
                    metadata={"category": "news"},
                )
            )
        else:
            raise ValueError(
                f"Unknown strategy name: {name}. "
                "Supported values are: ga, momentum, regex"
            )

    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare built-in traders over the same ticker/date range."
    )
    parser.add_argument("--ticker", default="AMZN", help="Ticker symbol to backtest.")
    parser.add_argument("--start-date", default="2025-01-01", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2025-12-31", help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument(
        "--strategies",
        default="ga,momentum,regex",
        help="Comma-separated list from: ga,momentum,regex",
    )
    parser.add_argument("--sort-by", default="final_value", help="Column to sort the comparison by.")
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    parser.add_argument(
        "--show-backtester-output",
        action="store_true",
        help="Show per-day backtester prints instead of suppressing them.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for trader/backtester output.",
    )
    parser.add_argument("--output-csv", default=None, help="Optional path to save the comparison table as CSV.")

    parser.add_argument("--ga-population-size", type=int, default=24)
    parser.add_argument("--ga-generations", type=int, default=10)
    parser.add_argument("--ga-random-seed", type=int, default=11)
    parser.add_argument("--ga-min-training-rows", type=int, default=90)
    parser.add_argument("--ga-retrain-every-rows", type=int, default=45)
    parser.add_argument("--ga-min-confidence", type=float, default=0.50)

    parser.add_argument("--momentum-short-window", type=int, default=10)
    parser.add_argument("--momentum-long-window", type=int, default=30)
    parser.add_argument("--momentum-min-confidence", type=float, default=0.05)

    parser.add_argument("--regex-confidence", type=float, default=0.05)
    parser.add_argument("--regex-max-news", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    strategy_names = [name.strip() for name in args.strategies.split(",") if name.strip()]
    specs = build_standard_strategy_specs(
        ticker=args.ticker,
        strategy_names=strategy_names,
        ga_population_size=args.ga_population_size,
        ga_generations=args.ga_generations,
        ga_random_seed=args.ga_random_seed,
        ga_min_training_rows=args.ga_min_training_rows,
        ga_retrain_every_rows=args.ga_retrain_every_rows,
        ga_min_confidence_to_trade=args.ga_min_confidence,
        momentum_short_window=args.momentum_short_window,
        momentum_long_window=args.momentum_long_window,
        momentum_min_confidence_to_trade=args.momentum_min_confidence,
        regex_confidence_needed=args.regex_confidence,
        regex_max_news=args.regex_max_news,
    )

    price_provider = PriceDataProvider()
    news_provider: Optional[NewsProviderABS] = None
    if any(spec.use_news for spec in specs):
        from utility.market_news.market_news_provider_ABS import FinnhubNewsProvider

        news_provider = FinnhubNewsProvider()

    comparison = compare_strategies(
        strategies=specs,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        price_provider=price_provider,
        news_provider=news_provider,
        sort_by=args.sort_by,
        ascending=args.ascending,
        suppress_backtester_output=not args.show_backtester_output,
    )

    display_columns = [
        column
        for column in [
            "strategy",
            "ticker",
            "start_date",
            "end_date",
            "final_value",
            "profit",
            "return_pct",
            "trade_count",
            "category",
        ]
        if column in comparison.columns
    ]

    print(comparison[display_columns].round(4).to_string(index=False))

    if args.output_csv:
        comparison.to_csv(args.output_csv, index=False)
        print(f"\nSaved comparison to {args.output_csv}")


if __name__ == "__main__":
    main()
