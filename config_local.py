# Local Configuration Overrides for IB Stock Trading
# This file overrides settings from supertrend_strategy.py
# Optimized based on parameter sweep results (Dec 2024)

# ============================================================================
# Timeframe Settings
# ============================================================================
TIMEFRAME = "1h"           # Main timeframe (sweep confirmed 1h is best)
HIGHER_TIMEFRAME = "4h"    # HTF filter (sweep tested 3h-12h, 4h works well)
LOOKBACK = 500             # Number of bars for analysis
HTF_LOOKBACK = 100         # Higher timeframe lookback

# ============================================================================
# Trading Mode
# ============================================================================
ENABLE_LONGS = True        # Enable long positions
ENABLE_SHORTS = True       # Enable short positions (per-symbol in tickers_config)

# ============================================================================
# Hold Filter (prevents overtrading)
# ============================================================================
USE_MIN_HOLD_FILTER = True
DEFAULT_MIN_HOLD_BARS = 6  # Minimum bars to hold (sweep result)

# ============================================================================
# Higher Timeframe Filter (trend confirmation)
# ============================================================================
USE_HIGHER_TIMEFRAME_FILTER = False  # Disabled - per-symbol optimization is better
HTF_LENGTH = 14
HTF_FACTOR = 2.5

# ============================================================================
# Momentum Filter (RSI)
# ============================================================================
USE_MOMENTUM_FILTER = False    # Disabled for now

# ============================================================================
# Breakout Filter
# ============================================================================
USE_BREAKOUT_FILTER = False

# ============================================================================
# Risk Management
# ============================================================================
RISK_FRACTION = 1              # Fraction of capital to risk per trade
STAKE_DIVISOR = 14             # Divides capital for position sizing
FEE_RATE = 0.001               # Trading fee rate (0.1%)
ATR_WINDOW = 14                # ATR period for volatility calculation
ATR_STOP_MULT = 1.5            # ATR multiplier for stop loss

# ============================================================================
# Optimized Indicator Parameters (defaults, per-symbol in tickers_config.py)
# ============================================================================
# Best performers from sweep:
# - KAMA: param_a=10-30, param_b=20-50 (most symbols)
# - JMA: param_a=10-20, param_b=0 (QBTS)
# - Supertrend: param_a=7-14, param_b=2.5 (fallback)

DEFAULT_INDICATOR = "kama"     # Best overall indicator
DEFAULT_PARAM_A = 14           # Default length/fast period
DEFAULT_PARAM_B = 30           # Default phase/slow period

# ============================================================================
# Output Directories
# ============================================================================
BASE_OUT_DIR = "report_html"
CLEAR_BASE_OUTPUT_ON_SWEEP = True
