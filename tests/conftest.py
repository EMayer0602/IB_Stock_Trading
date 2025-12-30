"""
Pytest configuration and fixtures for IB Stock Trading tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture
def sample_ohlcv_df():
    """
    Sample OHLCV data with realistic price movements for testing indicators.
    Creates 100 bars of hourly data with a clear uptrend followed by downtrend.
    """
    np.random.seed(42)  # Reproducible results
    n_bars = 100

    # Create a price series with trend
    base_price = 100.0
    trend = np.concatenate([
        np.linspace(0, 20, 50),   # Uptrend
        np.linspace(20, 5, 50)    # Downtrend
    ])
    noise = np.random.normal(0, 1, n_bars)
    close_prices = base_price + trend + noise

    # Generate OHLC from close
    high_prices = close_prices + np.abs(np.random.normal(1, 0.5, n_bars))
    low_prices = close_prices - np.abs(np.random.normal(1, 0.5, n_bars))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')

    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(10000, 100000, n_bars)
    }, index=dates)


@pytest.fixture
def simple_ohlcv_df():
    """
    Simple OHLCV data with known values for exact calculation verification.
    """
    return pd.DataFrame({
        'open':  [100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0, 104.0],
        'high':  [102.0, 103.0, 104.0, 103.5, 105.0, 104.0, 106.0, 105.5, 107.0, 106.0],
        'low':   [99.0,  100.0, 101.0, 100.5, 102.0, 101.0, 103.0, 102.5, 104.0, 103.0],
        'close': [101.0, 102.0, 103.0, 102.5, 104.0, 103.0, 105.0, 104.5, 106.0, 105.0],
        'volume': [1000] * 10
    })


@pytest.fixture
def trending_up_df():
    """OHLCV data with clear uptrend for signal testing."""
    n_bars = 50
    close_prices = np.linspace(100, 150, n_bars) + np.random.normal(0, 0.5, n_bars)

    return pd.DataFrame({
        'open': np.roll(close_prices, 1),
        'high': close_prices + 1,
        'low': close_prices - 1,
        'close': close_prices,
        'volume': [10000] * n_bars
    })


@pytest.fixture
def trending_down_df():
    """OHLCV data with clear downtrend for signal testing."""
    n_bars = 50
    close_prices = np.linspace(150, 100, n_bars) + np.random.normal(0, 0.5, n_bars)

    return pd.DataFrame({
        'open': np.roll(close_prices, 1),
        'high': close_prices + 1,
        'low': close_prices - 1,
        'close': close_prices,
        'volume': [10000] * n_bars
    })


@pytest.fixture
def mock_trading_state():
    """Fresh trading state for paper trader tests."""
    return {
        'total_capital': 14000.0,
        'positions': [],
        'closed_trades': [],
        'last_update': datetime.now(NY_TZ).isoformat()
    }


@pytest.fixture
def mock_position():
    """Sample open position for testing."""
    return {
        'key': 'AAPL_long_supertrend',
        'symbol': 'AAPL',
        'direction': 'long',
        'indicator': 'supertrend',
        'htf': '1d',
        'entry_time': datetime.now(NY_TZ).isoformat(),
        'entry_price': 150.0,
        'stake': 1000.0,
        'quantity': 6,
        'param_a': 10,
        'param_b': 3.0,
        'atr_mult': 1.5,
        'min_hold_bars': 12,
        'last_price': 150.0,
        'bars_held': 0,
        'unrealized_pnl': 0.0
    }


@pytest.fixture
def mock_short_position():
    """Sample short position for testing."""
    return {
        'key': 'AAPL_short_supertrend',
        'symbol': 'AAPL',
        'direction': 'short',
        'indicator': 'supertrend',
        'htf': '1d',
        'entry_time': datetime.now(NY_TZ).isoformat(),
        'entry_price': 150.0,
        'stake': 1000.0,
        'quantity': 6,
        'param_a': 10,
        'param_b': 3.0,
        'atr_mult': 1.5,
        'min_hold_bars': 12,
        'last_price': 150.0,
        'bars_held': 0,
        'unrealized_pnl': 0.0
    }
