#!/usr/bin/env python3
"""
Paper Trader for IB Stock Trading.
Simulates trading with the Supertrend strategy on US stocks via Interactive Brokers.
"""

import os
import sys
import json
import argparse
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

import supertrend_strategy as st
from tickers_config import (
    TICKERS, SYMBOLS, get_ticker_config,
    get_enabled_symbols, get_capital_for_symbol
)

# Timezones
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# File paths
STATE_FILE = "paper_trading_state.json"
TRADE_LOG_CSV = "paper_trading_log.csv"
SIMULATION_LOG_CSV = "paper_trading_simulation_log.csv"
SIMULATION_LOG_JSON = "paper_trading_simulation_log.json"

# Trading settings
STAKE_DIVISOR = 14
FEE_RATE = 0.001  # IB commission approximation
MAX_OPEN_POSITIONS = 10
DEFAULT_SIGNAL_INTERVAL_MIN = 60  # Check every hour for stocks

# Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0


@dataclass
class Position:
    """Represents an open position."""
    key: str
    symbol: str
    direction: str  # "long" or "short"
    indicator: str
    htf: str
    entry_time: str
    entry_price: float
    stake: float
    quantity: int
    param_a: float
    param_b: float
    atr_mult: Optional[float]
    min_hold_bars: int
    last_price: float = 0.0
    bars_held: int = 0
    unrealized_pnl: float = 0.0


@dataclass
class ClosedTrade:
    """Represents a completed trade."""
    symbol: str
    direction: str
    indicator: str
    htf: str
    param_desc: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    stake: float
    quantity: int
    fees: float
    pnl: float
    equity_after: float
    reason: str


def is_market_open() -> bool:
    """Check if US stock market is currently open."""
    now = datetime.now(NY_TZ)

    # Check weekday
    if now.weekday() >= 5:  # Saturday or Sunday
        return False

    market_open = now.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )
    market_close = now.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0
    )

    return market_open <= now <= market_close


def time_until_market_open() -> timedelta:
    """Get time until next market open."""
    now = datetime.now(NY_TZ)

    next_open = now.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )

    if now >= next_open:
        next_open += timedelta(days=1)

    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    return next_open - now


def load_state() -> Dict:
    """Load trading state from file."""
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[State] Error loading: {e}")

    # Default state
    total_capital = sum(
        t["initial_capital_long"] + t["initial_capital_short"]
        for t in TICKERS.values()
    )

    return {
        "total_capital": total_capital,
        "positions": [],
        "closed_trades": [],
        "last_update": datetime.now(NY_TZ).isoformat()
    }


def save_state(state: Dict):
    """Save trading state to file."""
    state["last_update"] = datetime.now(NY_TZ).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def find_position(state: Dict, key: str) -> Optional[Dict]:
    """Find a position by key."""
    for pos in state.get("positions", []):
        if pos.get("key") == key:
            return pos
    return None


def calculate_stake(symbol: str, direction: str, state: Dict) -> float:
    """Calculate stake for a new position."""
    config = get_ticker_config(symbol)

    if direction == "long":
        allocated = config.get("initial_capital_long", 1000)
    else:
        allocated = config.get("initial_capital_short", 1000)

    # Use allocated capital or portion of total
    total = state.get("total_capital", 10000)
    stake = min(allocated, total / STAKE_DIVISOR)

    return max(stake, 0)


def calculate_quantity(stake: float, price: float, round_factor: int = 1) -> int:
    """Calculate number of shares to buy/sell."""
    if price <= 0:
        return 0

    raw_qty = stake / price
    # Round to nearest round_factor
    qty = round(raw_qty / round_factor) * round_factor

    return max(int(qty), 1)


def open_position(
    state: Dict,
    symbol: str,
    direction: str,
    indicator: str,
    param_a: float,
    param_b: float,
    htf: str,
    price: float,
    atr_mult: Optional[float] = None,
    min_hold_bars: int = 0
) -> bool:
    """Open a new position."""
    key = f"{symbol}_{direction}_{indicator}"

    if find_position(state, key):
        print(f"[Position] Already open: {key}")
        return False

    # Check max positions
    if len(state.get("positions", [])) >= MAX_OPEN_POSITIONS:
        print(f"[Position] Max positions reached ({MAX_OPEN_POSITIONS})")
        return False

    config = get_ticker_config(symbol)
    stake = calculate_stake(symbol, direction, state)
    quantity = calculate_quantity(stake, price, config.get("order_round_factor", 1))

    if quantity <= 0:
        print(f"[Position] Invalid quantity for {symbol}")
        return False

    position = {
        "key": key,
        "symbol": symbol,
        "direction": direction,
        "indicator": indicator,
        "htf": htf,
        "entry_time": datetime.now(NY_TZ).isoformat(),
        "entry_price": price,
        "stake": stake,
        "quantity": quantity,
        "param_a": param_a,
        "param_b": param_b,
        "atr_mult": atr_mult,
        "min_hold_bars": min_hold_bars,
        "last_price": price,
        "bars_held": 0,
        "unrealized_pnl": 0.0
    }

    state.setdefault("positions", []).append(position)
    save_state(state)

    print(f"[ENTRY] {direction.upper()} {symbol} @ ${price:.2f} | {quantity} shares | ${stake:.2f}")
    return True


def close_position(
    state: Dict,
    key: str,
    exit_price: float,
    reason: str = "Signal"
) -> Optional[ClosedTrade]:
    """Close an existing position."""
    position = find_position(state, key)
    if not position:
        return None

    entry_price = position["entry_price"]
    stake = position["stake"]
    quantity = position["quantity"]
    direction = position["direction"]

    # Calculate P&L
    if direction == "long":
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price

    gross_pnl = price_diff * quantity
    fees = stake * FEE_RATE * 2  # Entry + exit
    net_pnl = gross_pnl - fees

    # Update capital
    state["total_capital"] = state.get("total_capital", 0) + net_pnl

    # Create closed trade record
    trade = ClosedTrade(
        symbol=position["symbol"],
        direction=direction,
        indicator=position["indicator"],
        htf=position["htf"],
        param_desc=f"ParamA={position['param_a']}, ParamB={position['param_b']}",
        entry_time=position["entry_time"],
        entry_price=entry_price,
        exit_time=datetime.now(NY_TZ).isoformat(),
        exit_price=exit_price,
        stake=stake,
        quantity=quantity,
        fees=fees,
        pnl=net_pnl,
        equity_after=state["total_capital"],
        reason=reason
    )

    # Remove from positions, add to closed trades
    state["positions"] = [p for p in state["positions"] if p["key"] != key]
    state.setdefault("closed_trades", []).append(asdict(trade))
    save_state(state)

    pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
    print(f"[EXIT] {direction.upper()} {position['symbol']} @ ${exit_price:.2f} | {reason} | {pnl_str}")

    return trade


def update_position_prices(state: Dict):
    """Update unrealized P&L for all positions."""
    for position in state.get("positions", []):
        symbol = position["symbol"]

        # Try to get current price
        try:
            from ib_connector import get_connector
            connector = get_connector()
            if connector.is_connected():
                price = connector.get_current_price(symbol)
                if price:
                    position["last_price"] = price

                    entry = position["entry_price"]
                    qty = position["quantity"]

                    if position["direction"] == "long":
                        position["unrealized_pnl"] = (price - entry) * qty
                    else:
                        position["unrealized_pnl"] = (entry - price) * qty
        except:
            pass

    save_state(state)


def run_signal_check(state: Dict, symbols: Optional[List[str]] = None):
    """Check for entry/exit signals across all symbols."""
    if symbols is None:
        symbols = SYMBOLS

    print(f"\n[{datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')}] Running signal check...")

    # Check exits first
    for position in list(state.get("positions", [])):
        symbol = position["symbol"]
        df = st.fetch_data(symbol)

        if df is None or len(df) < 10:
            continue

        # Increment bars held
        position["bars_held"] = position.get("bars_held", 0) + 1

        # Check min hold
        if position["bars_held"] < position.get("min_hold_bars", 0):
            continue

        exit_signal, reason = st.check_exit_signal(
            df,
            position["indicator"],
            position["param_a"],
            position["param_b"],
            position["direction"],
            position["entry_price"],
            position.get("atr_mult")
        )

        if exit_signal:
            exit_price = float(df["close"].iloc[-1])
            close_position(state, position["key"], exit_price, reason)

    # Check entries
    for symbol in symbols:
        config = get_ticker_config(symbol)
        if not config:
            continue

        df = st.fetch_data(symbol)
        if df is None or len(df) < 10:
            continue

        current_price = float(df["close"].iloc[-1])

        # Check each indicator
        for indicator in ["supertrend", "jma", "kama"]:
            preset = st.INDICATOR_PRESETS.get(indicator, {})
            param_a = preset.get("default_a", 10)
            param_b = preset.get("default_b", 0)

            # Check long
            if config.get("long", False) and st.ENABLE_LONGS:
                key = f"{symbol}_long_{indicator}"
                if not find_position(state, key):
                    entry, reason = st.check_entry_signal(df, indicator, param_a, param_b, "long")
                    if entry:
                        open_position(
                            state, symbol, "long", indicator,
                            param_a, param_b, st.HIGHER_TIMEFRAME,
                            current_price, atr_mult=1.5, min_hold_bars=12
                        )

            # Check short
            if config.get("short", False) and st.ENABLE_SHORTS:
                key = f"{symbol}_short_{indicator}"
                if not find_position(state, key):
                    entry, reason = st.check_entry_signal(df, indicator, param_a, param_b, "short")
                    if entry:
                        open_position(
                            state, symbol, "short", indicator,
                            param_a, param_b, st.HIGHER_TIMEFRAME,
                            current_price, atr_mult=1.5, min_hold_bars=12
                        )


def print_status(state: Dict):
    """Print current trading status."""
    print("\n" + "=" * 60)
    print("IB STOCK TRADING - STATUS")
    print("=" * 60)

    capital = state.get("total_capital", 0)
    positions = state.get("positions", [])
    trades = state.get("closed_trades", [])

    print(f"Capital: ${capital:,.2f}")
    print(f"Open Positions: {len(positions)}")
    print(f"Closed Trades: {len(trades)}")

    if positions:
        print("\nOpen Positions:")
        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            print(f"  {pos['symbol']} {pos['direction'].upper()} @ ${pos['entry_price']:.2f} | {pnl_str}")

    if trades:
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        print(f"\nTotal P&L: ${total_pnl:+,.2f} | Win Rate: {wins}/{len(trades)}")

    print("=" * 60)


def run_monitor(state: Dict, interval_minutes: int = 60):
    """Run continuous monitoring loop."""
    print(f"\n[Monitor] Starting with {interval_minutes}min interval...")
    print("[Monitor] Press Ctrl+C to stop")

    while True:
        try:
            if is_market_open():
                run_signal_check(state)
                update_position_prices(state)
                print_status(state)
            else:
                wait_time = time_until_market_open()
                print(f"[Monitor] Market closed. Next open in {wait_time}")

            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n[Monitor] Stopped by user")
            break
        except Exception as e:
            print(f"[Monitor] Error: {e}")
            time.sleep(60)


def save_logs(state: Dict):
    """Save trade logs to CSV and JSON."""
    trades = state.get("closed_trades", [])
    if not trades:
        return

    df = pd.DataFrame(trades)
    df.to_csv(SIMULATION_LOG_CSV, index=False)

    with open(SIMULATION_LOG_JSON, "w") as f:
        json.dump(trades, f, indent=2)

    print(f"[Logs] Saved {len(trades)} trades to {SIMULATION_LOG_CSV}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IB Stock Paper Trader")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--monitor", action="store_true", help="Run monitoring loop")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in minutes")
    parser.add_argument("--check", action="store_true", help="Run one signal check")
    parser.add_argument("--reset", action="store_true", help="Reset state to initial capital")

    args = parser.parse_args()

    if args.reset:
        total = sum(t["initial_capital_long"] + t["initial_capital_short"] for t in TICKERS.values())
        state = {
            "total_capital": total,
            "positions": [],
            "closed_trades": []
        }
        save_state(state)
        print(f"[Reset] State reset to ${total:,.2f}")
        return

    state = load_state()

    if args.status:
        print_status(state)
        return

    if args.check:
        run_signal_check(state)
        print_status(state)
        save_logs(state)
        return

    if args.monitor:
        run_monitor(state, args.interval)
        save_logs(state)
        return

    # Default: show status
    print_status(state)


if __name__ == "__main__":
    main()
