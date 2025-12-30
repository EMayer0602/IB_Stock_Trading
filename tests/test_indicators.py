"""
Tests for technical indicator calculations in supertrend_strategy.py

These tests verify the correctness of critical financial calculations:
- ATR (Average True Range)
- Supertrend
- KAMA (Kaufman Adaptive Moving Average)
- JMA (Jurik Moving Average approximation)
- PSAR (Parabolic SAR)
- RSI (Relative Strength Index)
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from supertrend_strategy import (
    calculate_atr,
    calculate_supertrend,
    calculate_kama,
    calculate_jma,
    calculate_psar,
    calculate_rsi,
    generate_indicator_signals,
    timeframe_to_minutes,
)


class TestTimeframeConversion:
    """Tests for timeframe string parsing."""

    def test_timeframe_minutes(self):
        """Test minute timeframe parsing."""
        assert timeframe_to_minutes("1m") == 1
        assert timeframe_to_minutes("5m") == 5
        assert timeframe_to_minutes("15m") == 15
        assert timeframe_to_minutes("30m") == 30

    def test_timeframe_hours(self):
        """Test hour timeframe parsing."""
        assert timeframe_to_minutes("1h") == 60
        assert timeframe_to_minutes("4h") == 240

    def test_timeframe_days(self):
        """Test day timeframe parsing."""
        assert timeframe_to_minutes("1d") == 1440

    def test_timeframe_with_min_suffix(self):
        """Test 'min' suffix parsing."""
        assert timeframe_to_minutes("15min") == 15

    def test_timeframe_invalid_raises(self):
        """Test that invalid timeframe raises ValueError."""
        with pytest.raises(ValueError):
            timeframe_to_minutes("1w")


class TestCalculateATR:
    """Tests for Average True Range calculation."""

    def test_atr_returns_series(self, sample_ohlcv_df):
        """ATR should return a pandas Series."""
        result = calculate_atr(sample_ohlcv_df, window=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_atr_positive_values(self, sample_ohlcv_df):
        """ATR should always be positive (after warmup period)."""
        result = calculate_atr(sample_ohlcv_df, window=14)
        # Skip NaN values from warmup
        valid_values = result.dropna()
        assert (valid_values >= 0).all()

    def test_atr_increases_with_volatility(self):
        """ATR should be higher for more volatile data."""
        # Low volatility data
        low_vol = pd.DataFrame({
            'open':  [100.0] * 20,
            'high':  [101.0] * 20,
            'low':   [99.0] * 20,
            'close': [100.0] * 20,
        })

        # High volatility data
        high_vol = pd.DataFrame({
            'open':  [100.0] * 20,
            'high':  [110.0] * 20,
            'low':   [90.0] * 20,
            'close': [100.0] * 20,
        })

        atr_low = calculate_atr(low_vol, window=5).iloc[-1]
        atr_high = calculate_atr(high_vol, window=5).iloc[-1]

        assert atr_high > atr_low

    def test_atr_window_parameter(self, sample_ohlcv_df):
        """Different window sizes should produce different results."""
        atr_5 = calculate_atr(sample_ohlcv_df, window=5)
        atr_14 = calculate_atr(sample_ohlcv_df, window=14)

        # They should have different values (at least in some positions)
        assert not atr_5.equals(atr_14)

    def test_atr_handles_gaps(self):
        """ATR should handle price gaps correctly (true range includes gaps)."""
        df = pd.DataFrame({
            'open':  [100.0, 110.0, 105.0],  # Gap up on bar 2
            'high':  [102.0, 112.0, 107.0],
            'low':   [98.0,  108.0, 103.0],
            'close': [101.0, 111.0, 106.0],
        })

        atr = calculate_atr(df, window=2)
        # The ATR should capture the gap
        assert atr.iloc[-1] > 4  # True range of gap bar is high


class TestCalculateSupertrend:
    """Tests for Supertrend indicator calculation."""

    def test_supertrend_returns_tuple(self, sample_ohlcv_df):
        """Supertrend should return a tuple of (supertrend, direction)."""
        sample_ohlcv_df['atr'] = calculate_atr(sample_ohlcv_df, 14)
        result = calculate_supertrend(sample_ohlcv_df, length=10, factor=3.0)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_supertrend_series_lengths(self, sample_ohlcv_df):
        """Both returned series should match input length."""
        sample_ohlcv_df['atr'] = calculate_atr(sample_ohlcv_df, 14)
        supertrend, direction = calculate_supertrend(sample_ohlcv_df, length=10, factor=3.0)

        assert len(supertrend) == len(sample_ohlcv_df)
        assert len(direction) == len(sample_ohlcv_df)

    def test_supertrend_direction_values(self, sample_ohlcv_df):
        """Direction should only contain 1 (uptrend) or -1 (downtrend)."""
        sample_ohlcv_df['atr'] = calculate_atr(sample_ohlcv_df, 14)
        _, direction = calculate_supertrend(sample_ohlcv_df, length=10, factor=3.0)

        unique_values = set(direction.dropna().unique())
        assert unique_values.issubset({1, -1})

    def test_supertrend_uptrend_detection(self, trending_up_df):
        """Supertrend should detect uptrend in rising prices."""
        trending_up_df['atr'] = calculate_atr(trending_up_df, 14)
        _, direction = calculate_supertrend(trending_up_df, length=10, factor=3.0)

        # Most of the direction values should be 1 (uptrend)
        uptrend_pct = (direction == 1).sum() / len(direction)
        assert uptrend_pct > 0.5  # At least 50% uptrend (indicator lags)

    def test_supertrend_downtrend_detection(self, trending_down_df):
        """Supertrend should detect downtrend in falling prices."""
        trending_down_df['atr'] = calculate_atr(trending_down_df, 14)
        _, direction = calculate_supertrend(trending_down_df, length=10, factor=3.0)

        # Most of the direction values should be -1 (downtrend)
        downtrend_pct = (direction == -1).sum() / len(direction)
        assert downtrend_pct > 0.6  # At least 60% downtrend

    def test_supertrend_factor_sensitivity(self, sample_ohlcv_df):
        """Higher factor should result in fewer direction changes."""
        sample_ohlcv_df['atr'] = calculate_atr(sample_ohlcv_df, 14)

        _, dir_low = calculate_supertrend(sample_ohlcv_df, length=10, factor=1.5)
        _, dir_high = calculate_supertrend(sample_ohlcv_df, length=10, factor=4.0)

        # Count direction changes
        changes_low = (dir_low.diff() != 0).sum()
        changes_high = (dir_high.diff() != 0).sum()

        # Higher factor = fewer changes (more smoothing)
        assert changes_high <= changes_low


class TestCalculateKAMA:
    """Tests for Kaufman Adaptive Moving Average calculation."""

    def test_kama_returns_series(self, sample_ohlcv_df):
        """KAMA should return a pandas Series."""
        result = calculate_kama(sample_ohlcv_df['close'], fast_length=14, slow_length=30)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_kama_follows_trend(self, trending_up_df):
        """KAMA should generally follow price direction."""
        close = trending_up_df['close']
        kama = calculate_kama(close, fast_length=10, slow_length=30)

        # In uptrend, KAMA should be below close most of the time
        below_close = (kama < close).sum()
        assert below_close > len(close) * 0.5

    def test_kama_smoother_than_price(self, sample_ohlcv_df):
        """KAMA should be smoother than raw price."""
        close = sample_ohlcv_df['close']
        kama = calculate_kama(close, fast_length=14, slow_length=30)

        # Calculate variance of changes
        price_var = close.diff().var()
        kama_var = kama.diff().var()

        # KAMA changes should have lower variance
        assert kama_var < price_var

    def test_kama_fast_slow_relationship(self, sample_ohlcv_df):
        """Faster KAMA should be more responsive than slower KAMA."""
        close = sample_ohlcv_df['close']

        kama_fast = calculate_kama(close, fast_length=5, slow_length=15)
        kama_slow = calculate_kama(close, fast_length=20, slow_length=50)

        # Fast KAMA should have more variance
        fast_var = kama_fast.diff().var()
        slow_var = kama_slow.diff().var()

        assert fast_var > slow_var


class TestCalculateJMA:
    """Tests for Jurik Moving Average (approximation) calculation."""

    def test_jma_returns_series(self, sample_ohlcv_df):
        """JMA should return a pandas Series."""
        result = calculate_jma(sample_ohlcv_df['close'], length=20, phase=0)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_jma_smoother_than_price(self, sample_ohlcv_df):
        """JMA should smooth out price noise."""
        close = sample_ohlcv_df['close']
        jma = calculate_jma(close, length=20, phase=0)

        price_var = close.diff().var()
        jma_var = jma.diff().var()

        assert jma_var < price_var

    def test_jma_length_effect(self, sample_ohlcv_df):
        """Longer JMA length should produce smoother result."""
        close = sample_ohlcv_df['close']

        jma_short = calculate_jma(close, length=10, phase=0)
        jma_long = calculate_jma(close, length=50, phase=0)

        short_var = jma_short.diff().var()
        long_var = jma_long.diff().var()

        assert long_var < short_var

    def test_jma_phase_effect(self, sample_ohlcv_df):
        """Different phase values should produce different results."""
        close = sample_ohlcv_df['close']

        jma_neg = calculate_jma(close, length=20, phase=-50)
        jma_zero = calculate_jma(close, length=20, phase=0)
        jma_pos = calculate_jma(close, length=20, phase=50)

        # All should be different
        assert not jma_neg.equals(jma_zero)
        assert not jma_zero.equals(jma_pos)


class TestCalculatePSAR:
    """Tests for Parabolic SAR calculation."""

    def test_psar_returns_tuple(self, sample_ohlcv_df):
        """PSAR should return a tuple of (psar, direction)."""
        result = calculate_psar(sample_ohlcv_df, step=0.02, max_step=0.2)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_psar_series_lengths(self, sample_ohlcv_df):
        """Both returned series should match input length."""
        psar, direction = calculate_psar(sample_ohlcv_df, step=0.02, max_step=0.2)

        assert len(psar) == len(sample_ohlcv_df)
        assert len(direction) == len(sample_ohlcv_df)

    def test_psar_direction_values(self, sample_ohlcv_df):
        """Direction should only contain 1 (up) or -1 (down)."""
        _, direction = calculate_psar(sample_ohlcv_df, step=0.02, max_step=0.2)

        unique_values = set(direction.dropna().unique())
        assert unique_values.issubset({1, -1})

    def test_psar_below_price_in_uptrend(self, trending_up_df):
        """PSAR should be below price in uptrend."""
        psar, direction = calculate_psar(trending_up_df, step=0.02, max_step=0.2)
        close = trending_up_df['close']

        # When direction is up (1), PSAR should be below close
        uptrend_mask = direction == 1
        if uptrend_mask.any():
            psar_below = (psar[uptrend_mask] < close[uptrend_mask]).mean()
            assert psar_below > 0.9  # 90% of time PSAR below in uptrend

    def test_psar_above_price_in_downtrend(self, trending_down_df):
        """PSAR should be above price in downtrend."""
        psar, direction = calculate_psar(trending_down_df, step=0.02, max_step=0.2)
        close = trending_down_df['close']

        # When direction is down (-1), PSAR should be above close
        downtrend_mask = direction == -1
        if downtrend_mask.any():
            psar_above = (psar[downtrend_mask] > close[downtrend_mask]).mean()
            assert psar_above > 0.9  # 90% of time PSAR above in downtrend

    def test_psar_step_sensitivity(self, sample_ohlcv_df):
        """Higher step should result in faster PSAR movement."""
        psar_slow, _ = calculate_psar(sample_ohlcv_df, step=0.01, max_step=0.1)
        psar_fast, _ = calculate_psar(sample_ohlcv_df, step=0.05, max_step=0.5)

        # Fast PSAR should have higher variance
        slow_var = psar_slow.diff().abs().mean()
        fast_var = psar_fast.diff().abs().mean()

        assert fast_var > slow_var


class TestCalculateRSI:
    """Tests for Relative Strength Index calculation."""

    def test_rsi_returns_series(self, sample_ohlcv_df):
        """RSI should return a pandas Series."""
        result = calculate_rsi(sample_ohlcv_df['close'], window=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_rsi_bounded_0_100(self, sample_ohlcv_df):
        """RSI should be bounded between 0 and 100."""
        rsi = calculate_rsi(sample_ohlcv_df['close'], window=14)
        valid_rsi = rsi.dropna()

        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_high_in_uptrend(self, trending_up_df):
        """RSI should be high (>50) in strong uptrend."""
        rsi = calculate_rsi(trending_up_df['close'], window=14)
        valid_rsi = rsi.dropna()

        avg_rsi = valid_rsi.mean()
        assert avg_rsi > 50

    def test_rsi_low_in_downtrend(self, trending_down_df):
        """RSI should be low (<50) in strong downtrend."""
        rsi = calculate_rsi(trending_down_df['close'], window=14)
        valid_rsi = rsi.dropna()

        avg_rsi = valid_rsi.mean()
        assert avg_rsi < 50

    def test_rsi_overbought_oversold(self):
        """RSI should reach high values in uptrend, low values in downtrend."""
        np.random.seed(42)

        # Mixed up move (mostly up with some down days for valid RSI)
        prices = [100.0]
        for i in range(60):
            # 80% chance of up day, 20% chance of down day
            if np.random.random() < 0.8:
                prices.append(prices[-1] + np.random.uniform(0.5, 2.0))
            else:
                prices.append(prices[-1] - np.random.uniform(0.1, 0.5))

        up_prices = pd.Series(prices)
        rsi_up = calculate_rsi(up_prices, window=14)
        valid_rsi_up = rsi_up.dropna()

        # Mixed down move (mostly down with some up days)
        prices = [200.0]
        for i in range(60):
            if np.random.random() < 0.8:
                prices.append(prices[-1] - np.random.uniform(0.5, 2.0))
            else:
                prices.append(prices[-1] + np.random.uniform(0.1, 0.5))

        down_prices = pd.Series(prices)
        rsi_down = calculate_rsi(down_prices, window=14)
        valid_rsi_down = rsi_down.dropna()

        # Should show bias toward overbought/oversold
        assert len(valid_rsi_up) > 0, "RSI up should have valid values"
        assert valid_rsi_up.mean() > 55, f"RSI mean should be above neutral, got {valid_rsi_up.mean()}"
        assert len(valid_rsi_down) > 0, "RSI down should have valid values"
        assert valid_rsi_down.mean() < 45, f"RSI mean should be below neutral, got {valid_rsi_down.mean()}"


class TestGenerateIndicatorSignals:
    """Tests for signal generation from indicators."""

    def test_signals_supertrend(self, sample_ohlcv_df):
        """Test supertrend signal generation."""
        sample_ohlcv_df['atr'] = calculate_atr(sample_ohlcv_df, 14)
        signals = generate_indicator_signals(sample_ohlcv_df, 'supertrend', 10, 3.0)

        assert isinstance(signals, pd.Series)
        assert set(signals.dropna().unique()).issubset({1, -1})

    def test_signals_kama(self, sample_ohlcv_df):
        """Test KAMA signal generation."""
        signals = generate_indicator_signals(sample_ohlcv_df, 'kama', 14, 30)

        assert isinstance(signals, pd.Series)
        assert set(signals.dropna().unique()).issubset({0, 1, -1})

    def test_signals_jma(self, sample_ohlcv_df):
        """Test JMA signal generation."""
        signals = generate_indicator_signals(sample_ohlcv_df, 'jma', 20, 0)

        assert isinstance(signals, pd.Series)
        assert set(signals.dropna().unique()).issubset({0, 1, -1})

    def test_signals_psar(self, sample_ohlcv_df):
        """Test PSAR signal generation."""
        signals = generate_indicator_signals(sample_ohlcv_df, 'psar', 0.02, 0.2)

        assert isinstance(signals, pd.Series)
        assert set(signals.dropna().unique()).issubset({1, -1})

    def test_signals_unknown_indicator_raises(self, sample_ohlcv_df):
        """Unknown indicator should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown indicator"):
            generate_indicator_signals(sample_ohlcv_df, 'unknown', 10, 3.0)

    def test_signals_uptrend_produces_long(self, trending_up_df):
        """Uptrend should produce mostly long (1) signals."""
        trending_up_df['atr'] = calculate_atr(trending_up_df, 14)
        signals = generate_indicator_signals(trending_up_df, 'supertrend', 10, 3.0)

        long_pct = (signals == 1).sum() / len(signals)
        assert long_pct > 0.5

    def test_signals_downtrend_produces_short(self, trending_down_df):
        """Downtrend should produce mostly short (-1) signals."""
        trending_down_df['atr'] = calculate_atr(trending_down_df, 14)
        signals = generate_indicator_signals(trending_down_df, 'supertrend', 10, 3.0)

        short_pct = (signals == -1).sum() / len(signals)
        assert short_pct > 0.5
