#!/usr/bin/env python3
"""
Backtest Script for IB Stock Trading Strategy.
Simulates trading over historical data using yfinance.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    try:
        import simple_yf as yf
        YF_AVAILABLE = True
        print("[Info] Using simple_yf fallback for Yahoo Finance data")
    except ImportError:
        YF_AVAILABLE = False
        print("[Warning] yfinance not installed. Run: pip install yfinance")

from tickers_config import TICKERS, SYMBOLS, get_ticker_config
import supertrend_strategy as st

NY_TZ = ZoneInfo("America/New_York")
BACKTEST_OUTPUT_DIR = "backtest_results"
FEE_RATE = 0.001


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    indicator: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    quantity: int
    stake: float
    gross_pnl: float
    fees: float
    net_pnl: float
    bars_held: int
    reason: str


@dataclass 
class BacktestResult:
    symbol: str
    indicator: str
    direction: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    start_capital: float
    end_capital: float
    return_pct: float


def fetch_historical_data(symbol: str, period: str = "6mo", interval: str = "1h") -> Optional[pd.DataFrame]:
    if not YF_AVAILABLE:
        print(f"[Backtest] yfinance required")
        return None
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        df.columns = df.columns.str.lower()
        df["atr"] = st.calculate_atr(df, st.ATR_WINDOW)
        return df
    except Exception as e:
        print(f"[Backtest] Error fetching {symbol}: {e}")
        return None


def run_backtest_for_symbol(symbol, df, indicator="supertrend", direction="long", 
                            initial_capital=1000, param_a=None, param_b=None,
                            atr_mult=1.5, min_hold_bars=6):
    preset = st.INDICATOR_PRESETS.get(indicator, {})
    if param_a is None:
        param_a = preset.get("default_a", 10)
    if param_b is None:
        param_b = preset.get("default_b", 0)

    signals = st.generate_indicator_signals(df, indicator, param_a, param_b)
    trades = []
    equity = initial_capital
    position = None
    equity_curve = [equity]

    for i in range(1, len(df)):
        current_time = df.index[i]
        current_price = df["close"].iloc[i]
        current_signal = signals.iloc[i]
        prev_signal = signals.iloc[i - 1]
        current_atr = df["atr"].iloc[i] if "atr" in df.columns else 0

        if position is not None:
            position["bars_held"] += 1
            should_exit = False
            exit_reason = ""

            if position["bars_held"] >= min_hold_bars:
                if direction == "long" and current_signal == -1:
                    should_exit, exit_reason = True, "Trend flip"
                elif direction == "short" and current_signal == 1:
                    should_exit, exit_reason = True, "Trend flip"

                if not should_exit and atr_mult and current_atr > 0:
                    if direction == "long":
                        if current_price < position["entry_price"] - atr_mult * current_atr:
                            should_exit, exit_reason = True, f"ATR stop"
                    else:
                        if current_price > position["entry_price"] + atr_mult * current_atr:
                            should_exit, exit_reason = True, f"ATR stop"

            if should_exit:
                entry_price = position["entry_price"]
                quantity = position["quantity"]
                stake = position["stake"]
                if direction == "long":
                    gross_pnl = (current_price - entry_price) * quantity
                else:
                    gross_pnl = (entry_price - current_price) * quantity
                fees = stake * FEE_RATE * 2
                net_pnl = gross_pnl - fees
                equity += net_pnl
                trades.append(BacktestTrade(symbol, direction, indicator,
                    str(position["entry_time"]), entry_price, str(current_time),
                    current_price, quantity, stake, gross_pnl, fees, net_pnl,
                    position["bars_held"], exit_reason))
                position = None

        if position is None:
            should_enter = False
            if direction == "long" and current_signal == 1 and prev_signal != 1:
                should_enter = True
            elif direction == "short" and current_signal == -1 and prev_signal != -1:
                should_enter = True

            if should_enter and equity > 100:
                stake = min(equity * 0.95, initial_capital)
                quantity = int(stake / current_price)
                if quantity > 0:
                    position = {"entry_time": current_time, "entry_price": current_price,
                               "quantity": quantity, "stake": stake, "bars_held": 0}

        equity_curve.append(equity)

    total_trades = len(trades)
    winning = sum(1 for t in trades if t.net_pnl > 0)
    losing = total_trades - winning
    win_rate = winning / total_trades if total_trades > 0 else 0
    total_pnl = sum(t.net_pnl for t in trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

    result = BacktestResult(symbol, indicator, direction, total_trades, winning, losing,
        win_rate, total_pnl, avg_pnl, max_dd, initial_capital, equity,
        (equity - initial_capital) / initial_capital * 100)

    return trades, result


def run_full_backtest(symbols=None, period="6mo", interval="1h", indicators=None, directions=None):
    if symbols is None:
        symbols = SYMBOLS
    if indicators is None:
        indicators = ["supertrend", "jma", "kama"]
    if directions is None:
        directions = ["long"]

    print(f"\n{'='*60}")
    print(f"IB STOCK TRADING BACKTEST - {period}")
    print(f"{'='*60}\n")

    all_trades, all_results = [], []

    for symbol in symbols:
        config = get_ticker_config(symbol)
        print(f"[{symbol}] Fetching data...")
        df = fetch_historical_data(symbol, period, interval)
        if df is None or len(df) < 50:
            print(f"[{symbol}] Skipped")
            continue
        print(f"[{symbol}] {len(df)} bars")

        for indicator in indicators:
            for direction in directions:
                capital = config.get(f"initial_capital_{direction}", 1000)
                trades, result = run_backtest_for_symbol(symbol, df, indicator, direction, capital)
                all_trades.extend(trades)
                all_results.append(result)
                pnl = f"+${result.total_pnl:.2f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):.2f}"
                print(f"  {indicator:12} {direction:5} | {result.total_trades:3} trades | {result.win_rate*100:5.1f}% | {pnl:>10}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_pnl = sum(r.total_pnl for r in all_results)
    total_capital = sum(r.start_capital for r in all_results)
    print(f"Total Trades: {len(all_trades)}")
    print(f"Total P&L: ${total_pnl:+,.2f}")
    print(f"Return: {total_pnl/total_capital*100:+.2f}%")

    Path(BACKTEST_OUTPUT_DIR).mkdir(exist_ok=True)
    pd.DataFrame([asdict(t) for t in all_trades]).to_csv(f"{BACKTEST_OUTPUT_DIR}/backtest_trades.csv", index=False)
    pd.DataFrame([asdict(r) for r in all_results]).to_csv(f"{BACKTEST_OUTPUT_DIR}/backtest_results.csv", index=False)
    print(f"\nResults saved to {BACKTEST_OUTPUT_DIR}/")

    return all_trades, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IB Stock Backtest")
    parser.add_argument("--period", default="6mo", help="Period: 1mo, 3mo, 6mo, 1y")
    parser.add_argument("--interval", default="1h", help="Interval: 1h, 1d")
    parser.add_argument("--symbols", nargs="+", help="Symbols to test")
    parser.add_argument("--indicator", help="Single indicator")
    parser.add_argument("--long-only", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    indicators = [args.indicator] if args.indicator else None
    directions = ["long"] if args.long_only else ["long", "short"]
    run_full_backtest(symbols, args.period, args.interval, indicators, directions)
