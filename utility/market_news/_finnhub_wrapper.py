import os
import requests
from dotenv import load_dotenv

load_dotenv()
finnhub_api_key = os.getenv("FINNHUB_API_KEY")

if not finnhub_api_key:
    raise RuntimeError("FINNHUB_API_KEY not set")


def get_news() -> list[dict]:
    print("Getting finnhub news...")
    url = "https://finnhub.io/api/v1/news?category=general&token=" + str(finnhub_api_key)
    news_list = requests.get(url).json()
    return news_list


def ticker_news(ticker: str, from_date: str, to_date: str) -> list[dict]:  # this will require a painfull refactor later
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={finnhub_api_key}"
    return requests.get(url=url).json()


if __name__ == "__main__":
    get_news()
