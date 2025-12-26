# IB Stock Tickers Configuration
# Each ticker has: symbol, conID, long/short flags, capital allocation, rounding, trade timing

TICKERS = {
    "AAPL": {
        "symbol": "AAPL",
        "conID": 265598,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Open"
    },
    "GOOGL": {
        "symbol": "GOOGL",
        "conID": 208813720,
        "long": True,
        "short": True,
        "initial_capital_long": 1200,
        "initial_capital_short": 1200,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "NVDA": {
        "symbol": "NVDA",
        "conID": 4815747,
        "long": True,
        "short": True,
        "initial_capital_long": 1800,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "MSFT": {
        "symbol": "MSFT",
        "conID": 272093,
        "long": True,
        "short": True,
        "initial_capital_long": 1100,
        "initial_capital_short": 1100,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "META": {
        "symbol": "META",
        "conID": 11263881,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "AMD": {
        "symbol": "AMD",
        "conID": 4391,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "QBTS": {
        "symbol": "QBTS",
        "conID": 532663595,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "TSLA": {
        "symbol": "TSLA",
        "conID": 76792991,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "MRNA": {
        "symbol": "MRNA",
        "conID": 41450682,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "NFLX": {
        "symbol": "NFLX",
        "conID": 213276,
        "long": True,
        "short": True,
        "initial_capital_long": 1500,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Open"
    },
    "AMZN": {
        "symbol": "AMZN",
        "conID": 3691937,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "INTC": {
        "symbol": "INTC",
        "conID": 1977552,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Close"
    },
    "BRRR": {
        "symbol": "BRRR",
        "conID": 582852809,
        "long": True,
        "short": True,
        "initial_capital_long": 1000,
        "initial_capital_short": 1000,
        "order_round_factor": 1,
        "trade_on": "Open"
    },
    "QUBT": {
        "symbol": "QUBT",
        "conID": 380357230,
        "long": True,
        "short": True,
        "initial_capital_long": 2000,
        "initial_capital_short": 1000,
        "order_round_factor": 10,
        "trade_on": "Open"
    },
}

# Get list of symbols
SYMBOLS = list(TICKERS.keys())

# Calculate total capital
TOTAL_CAPITAL_LONG = sum(t["initial_capital_long"] for t in TICKERS.values())
TOTAL_CAPITAL_SHORT = sum(t["initial_capital_short"] for t in TICKERS.values())
TOTAL_CAPITAL = TOTAL_CAPITAL_LONG + TOTAL_CAPITAL_SHORT

def get_ticker_config(symbol: str) -> dict:
    """Get configuration for a specific ticker."""
    return TICKERS.get(symbol, {})

def get_enabled_symbols(direction: str = "long") -> list:
    """Get list of symbols enabled for a specific direction."""
    if direction == "long":
        return [s for s, t in TICKERS.items() if t.get("long", False)]
    elif direction == "short":
        return [s for s, t in TICKERS.items() if t.get("short", False)]
    return SYMBOLS

def get_capital_for_symbol(symbol: str, direction: str = "long") -> float:
    """Get allocated capital for a symbol and direction."""
    config = TICKERS.get(symbol, {})
    if direction == "long":
        return config.get("initial_capital_long", 1000)
    return config.get("initial_capital_short", 1000)
