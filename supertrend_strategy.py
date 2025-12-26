"""
Supertrend Strategy for IB Stock Trading.
Adapted from Crypto2 project for Interactive Brokers.
"""

import os
import math
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from ta.volatility import AverageTrueRange
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("[Warning] ta library not installed. Run: pip install ta")

from tickers_config import TICKERS, SYMBOLS, get_ticker_config, get_capital_for_symbol


# Load environment variables
def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_absolute():
        env_path = Path(__file__).resolve().parent / env_path
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()
    except OSError:
        pass


_load_env_file()


# Timezones
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def timeframe_to_minutes(tf_str: str) -> int:
    """Convert timeframe string to minutes."""
    tf_str = tf_str.lower().strip()
    if tf_str.endswith("m") or tf_str.endswith("min"):
        return int(tf_str.replace("min", "").replace("m", ""))
    elif tf_str.endswith("h"):
        return int(tf_str.replace("h", "")) * 60
    elif tf_str.endswith("d"):
        return int(tf_str.replace("d", "")) * 1440
    raise ValueError(f"Unsupported timeframe: {tf_str}")


# ============================================================================
# Configuration
# ============================================================================

EXCHANGE_ID = "IB"  # Interactive Brokers
TIMEFRAME = "1h"    # 1 hour bars
LOOKBACK = 500      # Number of bars for analysis
HIGHER_TIMEFRAME = "1d"  # Daily for HTF filter
HTF_LOOKBACK = 100

# Trading settings
RUN_PARAMETER_SWEEP = False
RUN_SAVED_PARAMS = False
RUN_OVERALL_BEST = True
ENABLE_LONGS = True
ENABLE_SHORTS = True

# Hold filter
USE_MIN_HOLD_FILTER = True
DEFAULT_MIN_HOLD_DAYS = 0
MIN_HOLD_DAY_VALUES = [0, 1, 2]

# Higher timeframe filter
USE_HIGHER_TIMEFRAME_FILTER = True
HTF_LENGTH = 20
HTF_FACTOR = 3.0
HTF_PSAR_STEP = 0.02
HTF_PSAR_MAX_STEP = 0.2
HTF_JMA_LENGTH = 30
HTF_JMA_PHASE = 0
HTF_KAMA_LENGTH = 20
HTF_KAMA_SLOW_LENGTH = 40
HTF_MAMA_FAST_LIMIT = 0.5
HTF_MAMA_SLOW_LIMIT = 0.05

# Momentum filter
USE_MOMENTUM_FILTER = False
MOMENTUM_TYPE = "RSI"
MOMENTUM_WINDOW = 14
RSI_LONG_THRESHOLD = 55
RSI_SHORT_THRESHOLD = 45

# Breakout filter
USE_BREAKOUT_FILTER = False
BREAKOUT_ATR_MULT = 1.5
BREAKOUT_REQUIRE_DIRECTION = True

# Capital and risk
START_EQUITY = sum(t["initial_capital_long"] + t["initial_capital_short"] for t in TICKERS.values())
RISK_FRACTION = 1
STAKE_DIVISOR = 14
FEE_RATE = 0.001  # IB fees are typically lower
ATR_WINDOW = 14
ATR_STOP_MULTS = [None, 1.0, 1.5, 2.0]

# Output
BASE_OUT_DIR = "report_html"
BARS_PER_DAY = max(1, int(1440 / timeframe_to_minutes(TIMEFRAME)))
CLEAR_BASE_OUTPUT_ON_SWEEP = True

OVERALL_SUMMARY_HTML = os.path.join(BASE_OUT_DIR, "overall_best_results.html")
OVERALL_PARAMS_CSV = os.path.join(BASE_OUT_DIR, "best_params_overall.csv")
OVERALL_DETAILED_HTML = os.path.join(BASE_OUT_DIR, "overall_best_detailed.html")
OVERALL_FLAT_CSV = os.path.join(BASE_OUT_DIR, "overall_best_flat_trades.csv")
OVERALL_FLAT_JSON = os.path.join(BASE_OUT_DIR, "overall_best_flat_trades.json")
GLOBAL_BEST_RESULTS = {}


# ============================================================================
# Config override system
# ============================================================================

def _load_config_overrides():
    """Load configuration overrides from config_local.py if it exists."""
    config_file = os.getenv("CONFIG_FILE", "config_local.py")
    config_path = Path(__file__).resolve().parent / config_file
    if not config_path.exists():
        return {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_local", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        overrides = {k: v for k, v in vars(config_module).items() if not k.startswith("_")}
        if overrides:
            print(f"[Config] Loaded {len(overrides)} overrides from {config_file}")
        return overrides
    except Exception as e:
        print(f"[Config] Error loading {config_file}: {e}")
        return {}


_config_overrides = _load_config_overrides()
for _k, _v in _config_overrides.items():
    if _k in globals():
        globals()[_k] = _v

# Recalculate derived values
BARS_PER_DAY = max(1, int(1440 / timeframe_to_minutes(TIMEFRAME)))


# ============================================================================
# Indicator Presets
# ============================================================================

INDICATOR_PRESETS = {
    "supertrend": {
        "display_name": "Supertrend",
        "slug": "supertrend",
        "param_a_label": "Length",
        "param_b_label": "Factor",
        "param_a_values": [7, 10, 14],
        "param_b_values": [2.0, 3.0, 4.0],
        "default_a": 10,
        "default_b": 3.0,
    },
    "psar": {
        "display_name": "Parabolic SAR",
        "slug": "psar",
        "param_a_label": "Step",
        "param_b_label": "MaxStep",
        "param_a_values": [0.01, 0.02, 0.03],
        "param_b_values": [0.1, 0.2, 0.3],
        "default_a": 0.02,
        "default_b": 0.2,
    },
    "jma": {
        "display_name": "JMA Trend",
        "slug": "jma",
        "param_a_label": "Length",
        "param_b_label": "Phase",
        "param_a_values": [10, 20, 30, 50],
        "param_b_values": [-50, 0, 50],
        "default_a": 20,
        "default_b": 0,
    },
    "kama": {
        "display_name": "KAMA Trend",
        "slug": "kama",
        "param_a_label": "FastLen",
        "param_b_label": "SlowLen",
        "param_a_values": [10, 14, 20],
        "param_b_values": [20, 30, 40],
        "default_a": 14,
        "default_b": 30,
    },
}


# ============================================================================
# Technical Indicators
# ============================================================================

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    if not TA_AVAILABLE:
        # Fallback calculation
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=window
    )
    return atr.average_true_range()


def calculate_supertrend(
    df: pd.DataFrame,
    length: int = 10,
    factor: float = 3.0,
    atr_col: str = "atr"
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Supertrend indicator."""
    hl2 = (df["high"] + df["low"]) / 2
    atr = df[atr_col] if atr_col in df.columns else calculate_atr(df, length)

    upper = hl2 + factor * atr
    lower = hl2 - factor * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upper.iloc[i]
            direction.iloc[i] = -1
            continue

        prev_close = df["close"].iloc[i - 1]
        prev_supertrend = supertrend.iloc[i - 1]
        prev_direction = direction.iloc[i - 1]
        curr_close = df["close"].iloc[i]

        if prev_direction == 1:
            # Uptrend
            lower_band = max(lower.iloc[i], prev_supertrend)
            if curr_close < lower_band:
                supertrend.iloc[i] = upper.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band
                direction.iloc[i] = 1
        else:
            # Downtrend
            upper_band = min(upper.iloc[i], prev_supertrend)
            if curr_close > upper_band:
                supertrend.iloc[i] = lower.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band
                direction.iloc[i] = -1

    return supertrend, direction


def calculate_jma(series: pd.Series, length: int = 20, phase: int = 0) -> pd.Series:
    """Calculate Jurik Moving Average (approximation)."""
    # Simplified JMA using EMA with phase adjustment
    alpha = 2 / (length + 1)
    phase_ratio = (phase / 100 + 1) / 2 if phase != 0 else 0.5

    # First pass - standard EMA
    ema1 = series.ewm(span=length, adjust=False).mean()

    # Second pass - smoothing with phase
    ema2 = ema1.ewm(span=int(length * phase_ratio) + 1, adjust=False).mean()

    return ema2


def calculate_kama(
    series: pd.Series,
    fast_length: int = 14,
    slow_length: int = 30
) -> pd.Series:
    """Calculate Kaufman Adaptive Moving Average."""
    change = (series - series.shift(fast_length)).abs()
    volatility = series.diff().abs().rolling(window=fast_length).sum()

    # Efficiency ratio
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)

    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (slow_length + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # KAMA calculation
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[0] = series.iloc[0]

    for i in range(1, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])

    return kama


def calculate_psar(
    df: pd.DataFrame,
    step: float = 0.02,
    max_step: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Parabolic SAR."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    psar = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    af = step
    ep = low.iloc[0]
    hp = high.iloc[0]
    lp = low.iloc[0]
    trend = 1  # 1 = up, -1 = down

    psar.iloc[0] = low.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if trend == 1:
            psar.iloc[i] = psar.iloc[i - 1] + af * (hp - psar.iloc[i - 1])
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i - 1], low.iloc[i - 2] if i > 1 else low.iloc[i - 1])

            if low.iloc[i] < psar.iloc[i]:
                trend = -1
                psar.iloc[i] = hp
                lp = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            psar.iloc[i] = psar.iloc[i - 1] + af * (lp - psar.iloc[i - 1])
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i - 1], high.iloc[i - 2] if i > 1 else high.iloc[i - 1])

            if high.iloc[i] > psar.iloc[i]:
                trend = 1
                psar.iloc[i] = lp
                hp = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + step, max_step)

        direction.iloc[i] = trend

    return psar, direction


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================================
# Signal Generation
# ============================================================================

def generate_indicator_signals(
    df: pd.DataFrame,
    indicator: str,
    param_a: float,
    param_b: float
) -> pd.Series:
    """Generate trading signals based on indicator."""
    close = df["close"]

    if indicator == "supertrend":
        _, direction = calculate_supertrend(df, int(param_a), param_b)
        return direction  # 1 = long, -1 = short

    elif indicator == "psar":
        _, direction = calculate_psar(df, param_a, param_b)
        return direction

    elif indicator == "jma":
        jma = calculate_jma(close, int(param_a), int(param_b))
        direction = pd.Series(0, index=df.index)
        direction[close > jma] = 1
        direction[close < jma] = -1
        return direction

    elif indicator == "kama":
        kama = calculate_kama(close, int(param_a), int(param_b))
        direction = pd.Series(0, index=df.index)
        direction[close > kama] = 1
        direction[close < kama] = -1
        return direction

    else:
        raise ValueError(f"Unknown indicator: {indicator}")


def check_entry_signal(
    df: pd.DataFrame,
    indicator: str,
    param_a: float,
    param_b: float,
    direction: str = "long"
) -> Tuple[bool, str]:
    """
    Check if there's an entry signal on the latest bar.

    Returns:
        Tuple of (signal_detected, reason)
    """
    if len(df) < 2:
        return False, "Insufficient data"

    signals = generate_indicator_signals(df, indicator, param_a, param_b)

    current_signal = signals.iloc[-1]
    prev_signal = signals.iloc[-2]

    if direction == "long":
        if current_signal == 1 and prev_signal != 1:
            return True, f"{indicator} flipped to LONG"
    else:  # short
        if current_signal == -1 and prev_signal != -1:
            return True, f"{indicator} flipped to SHORT"

    return False, "No signal"


def check_exit_signal(
    df: pd.DataFrame,
    indicator: str,
    param_a: float,
    param_b: float,
    position_direction: str = "long",
    entry_price: float = 0,
    atr_mult: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Check if there's an exit signal on the latest bar.

    Returns:
        Tuple of (exit_signal, reason)
    """
    if len(df) < 2:
        return False, ""

    signals = generate_indicator_signals(df, indicator, param_a, param_b)
    current_signal = signals.iloc[-1]
    current_price = df["close"].iloc[-1]

    # Check for trend reversal
    if position_direction == "long" and current_signal == -1:
        return True, "Trend flip"
    if position_direction == "short" and current_signal == 1:
        return True, "Trend flip"

    # Check ATR-based stop
    if atr_mult is not None and entry_price > 0:
        atr = df["atr"].iloc[-1] if "atr" in df.columns else calculate_atr(df).iloc[-1]

        if position_direction == "long":
            stop_price = entry_price - atr_mult * atr
            if current_price < stop_price:
                return True, f"ATR stop x{atr_mult:.2f}"
        else:
            stop_price = entry_price + atr_mult * atr
            if current_price > stop_price:
                return True, f"ATR stop x{atr_mult:.2f}"

    return False, ""


# ============================================================================
# Data Fetching (with IB integration)
# ============================================================================

_data_cache: Dict[str, pd.DataFrame] = {}


def fetch_data(
    symbol: str,
    timeframe: str = TIMEFRAME,
    limit: int = LOOKBACK
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a symbol.
    Uses IB connector when available, otherwise returns cached data.
    """
    cache_key = f"{symbol}_{timeframe}_{limit}"

    if cache_key in _data_cache:
        return _data_cache[cache_key]

    try:
        from ib_connector import get_connector

        connector = get_connector()
        if not connector.is_connected():
            connector.connect()

        # Convert timeframe to IB format
        bar_size_map = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
        }
        bar_size = bar_size_map.get(timeframe, "1 hour")

        # Calculate duration
        minutes = timeframe_to_minutes(timeframe)
        days_needed = max(1, int(limit * minutes / 1440) + 5)
        duration = f"{days_needed} D"

        df = connector.fetch_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
            use_rth=True
        )

        if df is not None and len(df) > 0:
            # Add ATR
            df["atr"] = calculate_atr(df, ATR_WINDOW)
            _data_cache[cache_key] = df
            return df

    except ImportError:
        print("[Data] IB connector not available, using sample data")
    except Exception as e:
        print(f"[Data] Error fetching {symbol}: {e}")

    return None


def clear_cache():
    """Clear the data cache."""
    global _data_cache
    _data_cache = {}


# ============================================================================
# Exchange Configuration
# ============================================================================

_exchange = None
_use_paper = True


def configure_exchange(use_paper: bool = True):
    """Configure the IB exchange connection."""
    global _exchange, _use_paper
    _use_paper = use_paper

    try:
        from ib_connector import configure_connection, IBConfig

        port = 7497 if use_paper else 7496
        _exchange = configure_connection(port=port, use_paper=use_paper)
        print(f"[IB] Configured for {'paper' if use_paper else 'live'} trading on port {port}")
    except ImportError:
        print("[IB] ib_connector not available")


def get_exchange():
    """Get the current exchange instance."""
    return _exchange


# ============================================================================
# Main entry point for testing
# ============================================================================

if __name__ == "__main__":
    print("IB Stock Trading - Supertrend Strategy")
    print(f"Symbols: {SYMBOLS}")
    print(f"Total capital: ${START_EQUITY:,.2f}")
    print(f"Timeframe: {TIMEFRAME}, HTF: {HIGHER_TIMEFRAME}")
