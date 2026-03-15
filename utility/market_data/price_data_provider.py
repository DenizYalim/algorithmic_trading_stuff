import yfinance as yf
import pandas as pd
from typing import Dict


class PriceDataProvider:
    def __init__(self):
        pass

    def get_historical_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        return df

    def get_price_on_date(self, ticker: str, date: str) -> float | None:
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
        prices = {}
        for ticker in tickers:
            data = yf.Ticker(ticker).history(period="1d", auto_adjust=False)
            if not data.empty:
                prices[ticker] = float(data["Close"].iloc[-1])
        return prices
