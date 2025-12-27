# IB Stock Tickers Configuration
# Each ticker has: symbol, conID, long/short flags, capital allocation, rounding, trade timing
# Best indicator/direction based on 1-year backtest (2024)

TICKERS = {
    # === PROFITABLE SYMBOLS (sorted by P&L) ===
    "QBTS": {
        "symbol": "QBTS",
        "conID": 532663595,
        "enabled": True,
        "indicator": "jma",          # Sweep: +$2194, 219%
        "param_a": 10,
        "param_b": 0,
        "long": True,
        "short": False,
        "initial_capital_long": 1000,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "INTC": {
        "symbol": "INTC",
        "conID": 1977552,
        "enabled": True,
        "indicator": "kama",         # Sweep: +$913, 91%
        "param_a": 30,
        "param_b": 30,
        "long": True,
        "short": False,
        "initial_capital_long": 1000,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "QUBT": {
        "symbol": "QUBT",
        "conID": 380357230,
        "enabled": True,
        "indicator": "kama",         # Sweep: +$2470, 124%
        "param_a": 10,
        "param_b": 20,
        "long": True,
        "short": False,
        "initial_capital_long": 2000,
        "initial_capital_short": 0,
        "order_round_factor": 10,
        "trade_on": "Open"
    },
    "NVDA": {
        "symbol": "NVDA",
        "conID": 4815747,
        "enabled": True,
        "indicator": "kama",         # Sweep: +$947, 53%
        "param_a": 20,
        "param_b": 20,
        "long": True,
        "short": False,
        "initial_capital_long": 1800,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "GOOGL": {
        "symbol": "GOOGL",
        "conID": 208813720,
        "enabled": True,
        "indicator": "kama",         # Sweep: +$571, 48%
        "param_a": 14,
        "param_b": 50,
        "long": True,
        "short": False,
        "initial_capital_long": 1200,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "MRNA": {
        "symbol": "MRNA",
        "conID": 41450682,
        "enabled": True,
        "indicator": "supertrend",  # Best: +$182, 30.8% win rate (SHORT!)
        "long": False,
        "short": True,              # Only short for MRNA
        "initial_capital_long": 0,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "NFLX": {
        "symbol": "NFLX",
        "conID": 213276,
        "enabled": True,
        "indicator": "supertrend",  # Best: +$148, 30.4% win rate
        "long": True,
        "short": False,
        "initial_capital_long": 1500,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Open"
    },
    "AAPL": {
        "symbol": "AAPL",
        "conID": 265598,
        "enabled": True,
        "indicator": "supertrend",  # Best: +$16, 38.1% win rate (marginal)
        "long": True,
        "short": False,
        "initial_capital_long": 1000,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Open"
    },
    # === UNPROFITABLE SYMBOLS (disabled) ===
    "MSFT": {
        "symbol": "MSFT",
        "conID": 272093,
        "enabled": False,           # Best was -$44
        "indicator": "jma",
        "long": True,
        "short": False,
        "initial_capital_long": 1100,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "META": {
        "symbol": "META",
        "conID": 11263881,
        "enabled": False,           # Best was -$49
        "indicator": "supertrend",
        "long": False,
        "short": True,
        "initial_capital_long": 0,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "AMZN": {
        "symbol": "AMZN",
        "conID": 3691937,
        "enabled": False,           # Best was -$135
        "indicator": "jma",
        "long": False,
        "short": True,
        "initial_capital_long": 0,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "TSLA": {
        "symbol": "TSLA",
        "conID": 76792991,
        "enabled": False,           # Best was -$192
        "indicator": "jma",
        "long": True,
        "short": False,
        "initial_capital_long": 1000,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "AMD": {
        "symbol": "AMD",
        "conID": 4391,
        "enabled": False,           # Best was -$194
        "indicator": "kama",
        "long": True,
        "short": False,
        "initial_capital_long": 1000,
        "initial_capital_short": 0,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "BRRR": {
        "symbol": "BRRR",
        "conID": 582852809,
        "enabled": False,           # Best was -$248
        "indicator": "kama",
        "long": False,
        "short": True,
        "initial_capital_long": 0,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Open"
    },
}

# Get list of all symbols
SYMBOLS = list(TICKERS.keys())

# Get list of enabled symbols only
ENABLED_SYMBOLS = [s for s, t in TICKERS.items() if t.get("enabled", True)]

# Calculate total capital (only enabled symbols)
TOTAL_CAPITAL_LONG = sum(t["initial_capital_long"] for t in TICKERS.values() if t.get("enabled", True))
TOTAL_CAPITAL_SHORT = sum(t["initial_capital_short"] for t in TICKERS.values() if t.get("enabled", True))
TOTAL_CAPITAL = TOTAL_CAPITAL_LONG + TOTAL_CAPITAL_SHORT

def get_ticker_config(symbol: str) -> dict:
    """Get configuration for a specific ticker."""
    return TICKERS.get(symbol, {})

def get_enabled_symbols(direction: str = None) -> list:
    """Get list of enabled symbols, optionally filtered by direction."""
    enabled = [s for s, t in TICKERS.items() if t.get("enabled", True)]
    if direction == "long":
        return [s for s in enabled if TICKERS[s].get("long", False)]
    elif direction == "short":
        return [s for s in enabled if TICKERS[s].get("short", False)]
    return enabled

def get_indicator_for_symbol(symbol: str) -> str:
    """Get the best indicator for a symbol."""
    config = TICKERS.get(symbol, {})
    return config.get("indicator", "supertrend")

def get_direction_for_symbol(symbol: str) -> str:
    """Get the trading direction for a symbol."""
    config = TICKERS.get(symbol, {})
    if config.get("long", False):
        return "long"
    elif config.get("short", False):
        return "short"
    return "long"

def get_capital_for_symbol(symbol: str, direction: str = "long") -> float:
    """Get allocated capital for a symbol and direction."""
    config = TICKERS.get(symbol, {})
    if direction == "long":
        return config.get("initial_capital_long", 1000)
    return config.get("initial_capital_short", 1000)

def get_best_strategies() -> list:
    """Get list of best strategy configurations for enabled symbols."""
    strategies = []
    for symbol, config in TICKERS.items():
        if config.get("enabled", True):
            strategies.append({
                "symbol": symbol,
                "indicator": config.get("indicator", "supertrend"),
                "param_a": config.get("param_a"),
                "param_b": config.get("param_b"),
                "direction": "long" if config.get("long") else "short",
                "capital": config.get("initial_capital_long", 0) or config.get("initial_capital_short", 1000)
            })
    return strategies
