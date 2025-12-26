"""Simple Yahoo Finance data fetcher without multitasking dependency."""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_historical_data(symbol: str, period: str = "6mo", interval: str = "1h",
                          start_date: str = None, end_date: str = None):
    """Fetch historical data from Yahoo Finance using direct API.

    Args:
        symbol: Stock ticker symbol
        period: Time period (1mo, 3mo, 6mo, 1y, 2y) - used if no dates specified
        interval: Bar interval (1h, 1d, etc.)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    """

    # Use date range if provided, otherwise use period
    if start_date and end_date:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    else:
        end_ts = int(datetime.now().timestamp())
        period_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730
        }
        days = period_map.get(period, 180)
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp())

    # Yahoo Finance interval mapping
    interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
        "1wk": "1wk",
        "1mo": "1mo"
    }
    yf_interval = interval_map.get(interval, "1h")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": yf_interval,
        "events": "history"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        result = data.get("chart", {}).get("result", [])
        if not result:
            print(f"[YF] No data for {symbol}")
            return None

        chart_data = result[0]
        timestamps = chart_data.get("timestamp", [])
        quotes = chart_data.get("indicators", {}).get("quote", [{}])[0]

        if not timestamps:
            return None

        df = pd.DataFrame({
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", [])
        }, index=pd.to_datetime(timestamps, unit='s'))

        # Remove rows with NaN
        df = df.dropna()

        return df

    except Exception as e:
        print(f"[YF] Error fetching {symbol}: {e}")
        return None


class Ticker:
    """Simple Ticker class to mimic yfinance interface."""

    def __init__(self, symbol: str):
        self.symbol = symbol

    def history(self, period: str = "6mo", interval: str = "1h"):
        return fetch_historical_data(self.symbol, period, interval)
