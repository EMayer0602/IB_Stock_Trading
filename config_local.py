# Local Configuration Overrides for IB Stock Trading
# This file overrides settings from supertrend_strategy.py
# Uncomment and modify the settings you want to change

# ============================================================================
# Timeframe Settings
# ============================================================================
TIMEFRAME = "1h"           # Main timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d
HIGHER_TIMEFRAME = "1d"    # Higher timeframe for trend filter
LOOKBACK = 500             # Number of bars for analysis
HTF_LOOKBACK = 100         # Higher timeframe lookback

# ============================================================================
# Trading Mode
# ============================================================================
ENABLE_LONGS = True        # Enable long positions
ENABLE_SHORTS = True       # Enable short positions

# ============================================================================
# Hold Filter (prevents overtrading)
# ============================================================================
USE_MIN_HOLD_FILTER = True
DEFAULT_MIN_HOLD_DAYS = 0  # Minimum days to hold a position

# ============================================================================
# Higher Timeframe Filter (trend confirmation)
# ============================================================================
USE_HIGHER_TIMEFRAME_FILTER = True
HTF_LENGTH = 20
HTF_FACTOR = 3.0
HTF_PSAR_STEP = 0.02
HTF_PSAR_MAX_STEP = 0.2
HTF_JMA_LENGTH = 30
HTF_JMA_PHASE = 0
HTF_KAMA_LENGTH = 20
HTF_KAMA_SLOW_LENGTH = 40

# ============================================================================
# Momentum Filter (RSI)
# ============================================================================
USE_MOMENTUM_FILTER = False    # Set to True to enable RSI filter
MOMENTUM_TYPE = "RSI"
MOMENTUM_WINDOW = 14
RSI_LONG_THRESHOLD = 55        # RSI above this for long entries
RSI_SHORT_THRESHOLD = 45       # RSI below this for short entries

# ============================================================================
# Breakout Filter
# ============================================================================
USE_BREAKOUT_FILTER = False
BREAKOUT_ATR_MULT = 1.5
BREAKOUT_REQUIRE_DIRECTION = True

# ============================================================================
# Risk Management
# ============================================================================
RISK_FRACTION = 1              # Fraction of capital to risk per trade
STAKE_DIVISOR = 14             # Divides capital for position sizing
FEE_RATE = 0.001               # Trading fee rate (0.1%)
ATR_WINDOW = 14                # ATR period for volatility calculation
# ATR_STOP_MULTS = [None, 1.0, 1.5, 2.0]  # ATR multipliers for stop loss

# ============================================================================
# Parameter Sweep (Optimization)
# ============================================================================
RUN_PARAMETER_SWEEP = False
RUN_SAVED_PARAMS = False
RUN_OVERALL_BEST = True

# ============================================================================
# Output Directories
# ============================================================================
BASE_OUT_DIR = "report_html"
CLEAR_BASE_OUTPUT_ON_SWEEP = True
