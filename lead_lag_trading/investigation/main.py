import pandas as pd
import yfinance as yf

from lead_lag_calc import LeadLagAnalyzer
from signal_tester import SignalTester


def download_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    return prices.dropna(how="all")


def main() -> None:
    tickers = [
        "SPY",
        "QQQ",
        "IWM",
        "XLE",
        "XLK",
        "XLF",
        "TLT",
        "GLD",
        "DBC",
        "UUP",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "CVX",
    ]

    prices = download_close_prices(
        tickers=tickers,
        start="2022-01-01",
        end="2025-01-01",
    )

    analyzer = LeadLagAnalyzer(
        prices=prices,
        return_method="log",
        min_obs=120,
    )

    results = analyzer.analyze_universe(
        max_lag=20,
        min_abs_corr=0.10,
        stability_window=60,
        min_abs_stability_mean=0.05,
        max_stability_std=0.20,
        min_sign_consistency=0.60,
        only_passed=True,
        top_n=10,
    )

    print("\nTop discovered pairs:")
    print(results[["leader", "follower", "best_lag", "correlation", "stability_mean", "stability_std", "score"]].round(4).to_string(index=False))

    tester = SignalTester(analyzer)

    # Test a single pair
    event_df, summary_df = tester.test_pair(
        leader="AAPL",
        follower="XLE",
        lag=14,
        z_window=20,
        threshold=1.5,
        require_passed_pair=False,
    )

    print("\nSingle-pair summary:")
    print(summary_df.round(4).to_string(index=False))

    print("\nRecent event rows:")
    print(event_df[event_df["event"]].tail(10).round(4).to_string())

    # Test all top pairs
    all_summaries = tester.test_top_pairs(
        pairs_df=results,
        z_window=20,
        threshold=1.5,
    )

    print("\nTop pair signal-test summaries:")
    print(all_summaries.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
