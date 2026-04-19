import pandas as pd
from typing import Dict
from pathlib import Path


class PriceDataProvider:
    def __init__(self, yfinance_cache_dir: str | None = None):
        self.yfinance_cache_dir = yfinance_cache_dir or str(
            Path(__file__).resolve().parents[2] / ".yfinance_cache"
        )

    def _load_yfinance(self):
        import yfinance as yf

        cache_dir = Path(self.yfinance_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
        return yf

    def get_historical_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        yf = self._load_yfinance()

        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        return df

    def get_price_on_date(self, ticker: str, date: str) -> float | None:
        yf = self._load_yfinance()

        start = pd.to_datetime(date)
        end = start + pd.Timedelta(days=1)

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

        if df.empty:
            return None

        try:
            close_data = df["Close"]

            # If yfinance returned a DataFrame (e.g. multi-index style), reduce it
            if isinstance(close_data, pd.DataFrame):
                if close_data.empty:
                    return None
                return float(close_data.iloc[0, 0])

            # Normal Series case
            return float(close_data.iloc[0])

        except Exception:
            return None

    def get_latest_prices(self, tickers: list) -> Dict[str, float]:
        yf = self._load_yfinance()

        prices = {}
        for ticker in tickers:
            data = yf.Ticker(ticker).history(period="1d", auto_adjust=False)
            if not data.empty:
                prices[ticker] = float(data["Close"].iloc[-1])
        return prices
