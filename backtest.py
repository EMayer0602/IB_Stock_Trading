#!/usr/bin/env python3
"""
Backtest Script for IB Stock Trading Strategy.
Simulates trading over historical data using yfinance.
Supports HTF (Higher Timeframe) filter and crossover modes.
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


def fetch_historical_data(symbol: str, period: str = "6mo", interval: str = "1h",
                          start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """Fetch historical data from Yahoo Finance."""
    if not YF_AVAILABLE:
        print(f"[Backtest] yfinance required")
        return None
    try:
        # Use simple_yf with date range support
        if start_date and end_date:
            from simple_yf import fetch_historical_data as yf_fetch
            df = yf_fetch(symbol, period, interval, start_date, end_date)
        else:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df.columns = df.columns.str.lower()
        df["atr"] = st.calculate_atr(df, st.ATR_WINDOW)
        return df
    except Exception as e:
        print(f"[Backtest] Error fetching {symbol}: {e}")
        return None


def get_htf_signal_for_time(htf_signals, current_time):
    """Get the HTF signal for a given time (use last available HTF bar)."""
    if htf_signals is None or htf_signals.empty:
        return 0  # No filter if no HTF data
    # Find the latest HTF signal before or at current_time
    valid_signals = htf_signals[htf_signals.index <= current_time]
    if valid_signals.empty:
        return 0
    return valid_signals.iloc[-1]


def get_htf_indicator_value_for_time(htf_values, current_time):
    """Get the HTF indicator value for a given time (use last available HTF bar)."""
    if htf_values is None or htf_values.empty:
        return None
    # Find the latest HTF value before or at current_time
    valid_values = htf_values[htf_values.index <= current_time]
    if valid_values.empty:
        return None
    return valid_values.iloc[-1]


def calculate_htf_indicator_values(htf_df, indicator, param_a, param_b):
    """Calculate the actual HTF indicator values (not signals)."""
    close = htf_df["close"]

    if indicator == "supertrend":
        supertrend, _ = st.calculate_supertrend(htf_df, int(param_a), param_b)
        return supertrend
    elif indicator == "psar":
        psar, _ = st.calculate_psar(htf_df, param_a, param_b)
        return psar
    elif indicator == "jma":
        return st.calculate_jma(close, int(param_a), int(param_b))
    elif indicator == "kama":
        return st.calculate_kama(close, int(param_a), int(param_b))
    else:
        # Fallback to simple EMA
        return close.ewm(span=int(param_a), adjust=False).mean()


def run_backtest_for_symbol(symbol, df, indicator="supertrend", direction="long",
                            initial_capital=1000, param_a=None, param_b=None,
                            atr_mult=1.5, min_hold_bars=6, htf_signals=None, use_htf_filter=True,
                            htf_values=None, use_htf_crossover=False):
    """Run backtest for a single symbol with optional HTF filter or crossover."""
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

        # Get HTF signal for current time (for filter mode)
        htf_signal = get_htf_signal_for_time(htf_signals, current_time) if use_htf_filter else 0

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

            # HTF Crossover Mode: entry when close crosses the HTF indicator value
            if use_htf_crossover and htf_values is not None:
                current_close = df["close"].iloc[i]
                prev_close = df["close"].iloc[i - 1]
                htf_value = get_htf_indicator_value_for_time(htf_values, current_time)
                prev_htf_value = get_htf_indicator_value_for_time(htf_values, df.index[i - 1])

                if htf_value is not None and prev_htf_value is not None:
                    if direction == "long":
                        # Long entry: close crosses ABOVE HTF indicator
                        if prev_close <= prev_htf_value and current_close > htf_value:
                            should_enter = True
                    elif direction == "short":
                        # Short entry: close crosses BELOW HTF indicator
                        if prev_close >= prev_htf_value and current_close < htf_value:
                            should_enter = True
            else:
                # Original mode: use LTF indicator signal with optional HTF filter
                if direction == "long" and current_signal == 1 and prev_signal != 1:
                    # HTF Filter: only enter long if HTF is also bullish (1) or no filter
                    if not use_htf_filter or htf_signals is None or htf_signal == 1:
                        should_enter = True
                elif direction == "short" and current_signal == -1 and prev_signal != -1:
                    # HTF Filter: only enter short if HTF is also bearish (-1) or no filter
                    if not use_htf_filter or htf_signals is None or htf_signal == -1:
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


def run_full_backtest(symbols=None, period="6mo", interval="1h", indicators=None, directions=None,
                      start_date=None, end_date=None, use_htf_filter=False, htf_indicator="kama",
                      use_htf_crossover=False):
    """Run backtest across multiple symbols with optional HTF filter or crossover."""
    if symbols is None:
        symbols = SYMBOLS
    if indicators is None:
        indicators = ["supertrend", "jma", "kama"]
    if directions is None:
        directions = ["long"]

    # Determine date range description
    if start_date and end_date:
        date_desc = f"{start_date} to {end_date}"
    else:
        date_desc = period

    if use_htf_crossover:
        htf_desc = f" + HTF Crossover ({htf_indicator.upper()})"
    elif use_htf_filter:
        htf_desc = f" + HTF Filter ({htf_indicator.upper()})"
    else:
        htf_desc = ""

    print(f"\n{'='*60}")
    print(f"IB STOCK TRADING BACKTEST - {date_desc}{htf_desc}")
    print(f"{'='*60}\n")

    all_trades, all_results = [], []

    for symbol in symbols:
        config = get_ticker_config(symbol)
        print(f"[{symbol}] Fetching data...")
        df = fetch_historical_data(symbol, period, interval, start_date, end_date)
        if df is None or len(df) < 50:
            print(f"[{symbol}] Skipped")
            continue

        # Fetch HTF (daily) data if HTF filter or crossover is enabled
        htf_signals = None
        htf_values = None
        if use_htf_filter or use_htf_crossover:
            htf_df = fetch_historical_data(symbol, period, "1d", start_date, end_date)
            if htf_df is not None and len(htf_df) > 10:
                htf_preset = st.INDICATOR_PRESETS.get(htf_indicator, {})
                htf_param_a = htf_preset.get("default_a", 14)
                htf_param_b = htf_preset.get("default_b", 30)

                if use_htf_crossover:
                    # Calculate actual indicator values for crossover detection
                    htf_values = calculate_htf_indicator_values(htf_df, htf_indicator, htf_param_a, htf_param_b)
                    print(f"[{symbol}] {len(df)} bars + HTF Crossover ({len(htf_df)} daily bars, {htf_indicator.upper()})")
                else:
                    htf_signals = st.generate_indicator_signals(htf_df, htf_indicator, htf_param_a, htf_param_b)
                    print(f"[{symbol}] {len(df)} bars + HTF Filter ({len(htf_df)} daily bars)")
            else:
                print(f"[{symbol}] {len(df)} bars (no HTF data)")
        else:
            print(f"[{symbol}] {len(df)} bars")

        for indicator in indicators:
            for direction in directions:
                capital = config.get(f"initial_capital_{direction}", 1000)
                trades, result = run_backtest_for_symbol(symbol, df, indicator, direction, capital,
                                                         htf_signals=htf_signals, use_htf_filter=use_htf_filter,
                                                         htf_values=htf_values, use_htf_crossover=use_htf_crossover)
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
    parser.add_argument("--start", help="Start date YYYY-MM-DD (use with --end)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (use with --start)")
    parser.add_argument("--htf", action="store_true", help="Enable HTF (daily) trend filter")
    parser.add_argument("--htf-crossover", action="store_true", help="Enable HTF crossover mode (entry on close crossing HTF indicator)")
    parser.add_argument("--htf-indicator", default="kama", help="HTF indicator: kama, supertrend, jma, psar")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    indicators = [args.indicator] if args.indicator else None
    directions = ["long"] if args.long_only else ["long", "short"]
    run_full_backtest(symbols, args.period, args.interval, indicators, directions,
                      args.start, args.end, args.htf, args.htf_indicator, args.htf_crossover)
