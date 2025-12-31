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


def fetch_historical_data(symbol: str, period: str = "6mo", interval: str = "1h",
                          start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
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


def run_backtest_for_symbol(symbol, df, indicator="supertrend", direction="long",
                            initial_capital=1000, param_a=None, param_b=None,
                            atr_mult=1.5, min_hold_bars=6, max_hold_bars=0,
                            htf_signals=None, use_htf_filter=True, htf_indicator=None):
    """Run backtest for a single symbol/indicator/direction combination.

    Args:
        symbol: Stock symbol
        df: OHLCV DataFrame
        indicator: Entry indicator (supertrend, kama, jma, psar)
        direction: Trade direction (long, short)
        initial_capital: Starting capital
        param_a: Indicator parameter A (e.g., length for supertrend)
        param_b: Indicator parameter B (e.g., factor for supertrend)
        atr_mult: ATR multiplier for stop loss
        min_hold_bars: Minimum bars to hold before exit
        max_hold_bars: Maximum bars to hold (0 = no limit, forces exit after N bars)
        htf_signals: Pre-calculated HTF signals for filtering
        use_htf_filter: Whether to use HTF trend filter
        htf_indicator: Name of HTF indicator used (for logging)
    """
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

        # Get HTF signal for current time
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

            # Force exit after max_hold_bars (time-based exit)
            if not should_exit and max_hold_bars > 0 and position["bars_held"] >= max_hold_bars:
                should_exit, exit_reason = True, "Max hold"

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


@dataclass
class DirectionConfig:
    """Configuration for a single direction (long or short)."""
    enabled: bool = True
    indicators: List[str] = None
    param_a: float = None
    param_b: float = None
    atr_mult: float = 1.5
    min_hold_bars: int = 6
    max_hold_bars: int = 0  # 0 = no limit, >0 = force exit after N bars
    htf_filter: bool = False
    htf_indicator: str = "kama"

    def __post_init__(self):
        if self.indicators is None:
            self.indicators = ["supertrend"]


def run_full_backtest(symbols=None, period="6mo", interval="1h", indicators=None, directions=None,
                      start_date=None, end_date=None, use_htf_filter=False, htf_indicator="kama",
                      long_config=None, short_config=None):
    """Run full backtest with optional separate long/short configurations.

    Args:
        symbols: List of symbols to test
        period: Data period (1mo, 3mo, 6mo, 1y)
        interval: Data interval (1h, 1d)
        indicators: List of indicators (used if no separate config)
        directions: List of directions (used if no separate config)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        use_htf_filter: Enable HTF filter (used if no separate config)
        htf_indicator: HTF indicator (used if no separate config)
        long_config: DirectionConfig for long trades
        short_config: DirectionConfig for short trades
    """
    if symbols is None:
        symbols = SYMBOLS

    # If no separate configs provided, create from legacy parameters
    if long_config is None and short_config is None:
        if indicators is None:
            indicators = ["supertrend", "jma", "kama"]
        if directions is None:
            directions = ["long"]

        long_config = DirectionConfig(
            enabled="long" in directions,
            indicators=indicators,
            htf_filter=use_htf_filter,
            htf_indicator=htf_indicator
        )
        short_config = DirectionConfig(
            enabled="short" in directions,
            indicators=indicators,
            htf_filter=use_htf_filter,
            htf_indicator=htf_indicator
        )

    # Determine date range description
    if start_date and end_date:
        date_desc = f"{start_date} to {end_date}"
    else:
        date_desc = period

    print(f"\n{'='*70}")
    print(f"IB STOCK TRADING BACKTEST - {date_desc}")
    print(f"{'='*70}")

    # Print configuration summary
    if long_config.enabled:
        htf_desc = f" + HTF({long_config.htf_indicator})" if long_config.htf_filter else ""
        max_desc = f" MaxHold={long_config.max_hold_bars}" if long_config.max_hold_bars > 0 else ""
        print(f"LONG:  {','.join(long_config.indicators)} | ATR×{long_config.atr_mult} | MinHold={long_config.min_hold_bars}{max_desc}{htf_desc}")
    if short_config.enabled:
        htf_desc = f" + HTF({short_config.htf_indicator})" if short_config.htf_filter else ""
        max_desc = f" MaxHold={short_config.max_hold_bars}" if short_config.max_hold_bars > 0 else ""
        print(f"SHORT: {','.join(short_config.indicators)} | ATR×{short_config.atr_mult} | MinHold={short_config.min_hold_bars}{max_desc}{htf_desc}")
    print(f"{'='*70}\n")

    all_trades, all_results = [], []

    for symbol in symbols:
        config = get_ticker_config(symbol)
        print(f"[{symbol}] Fetching data...")
        df = fetch_historical_data(symbol, period, interval, start_date, end_date)
        if df is None or len(df) < 50:
            print(f"[{symbol}] Skipped")
            continue

        print(f"[{symbol}] {len(df)} bars")

        # Fetch HTF data if needed for either direction
        htf_signals_long = None
        htf_signals_short = None

        if long_config.enabled and long_config.htf_filter:
            htf_df = fetch_historical_data(symbol, period, "1d", start_date, end_date)
            if htf_df is not None and len(htf_df) > 10:
                htf_preset = st.INDICATOR_PRESETS.get(long_config.htf_indicator, {})
                htf_param_a = htf_preset.get("default_a", 14)
                htf_param_b = htf_preset.get("default_b", 30)
                htf_signals_long = st.generate_indicator_signals(htf_df, long_config.htf_indicator, htf_param_a, htf_param_b)
                print(f"  HTF Long: {long_config.htf_indicator.upper()} ({len(htf_df)} daily bars)")

        if short_config.enabled and short_config.htf_filter:
            htf_df = fetch_historical_data(symbol, period, "1d", start_date, end_date)
            if htf_df is not None and len(htf_df) > 10:
                htf_preset = st.INDICATOR_PRESETS.get(short_config.htf_indicator, {})
                htf_param_a = htf_preset.get("default_a", 14)
                htf_param_b = htf_preset.get("default_b", 30)
                htf_signals_short = st.generate_indicator_signals(htf_df, short_config.htf_indicator, htf_param_a, htf_param_b)
                print(f"  HTF Short: {short_config.htf_indicator.upper()} ({len(htf_df)} daily bars)")

        # Run long trades
        if long_config.enabled:
            for indicator in long_config.indicators:
                capital = config.get("initial_capital_long", 1000)
                trades, result = run_backtest_for_symbol(
                    symbol, df, indicator, "long", capital,
                    param_a=long_config.param_a,
                    param_b=long_config.param_b,
                    atr_mult=long_config.atr_mult,
                    min_hold_bars=long_config.min_hold_bars,
                    max_hold_bars=long_config.max_hold_bars,
                    htf_signals=htf_signals_long,
                    use_htf_filter=long_config.htf_filter,
                    htf_indicator=long_config.htf_indicator
                )
                all_trades.extend(trades)
                all_results.append(result)
                pnl = f"+${result.total_pnl:.2f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):.2f}"
                print(f"  {indicator:12} long  | {result.total_trades:3} trades | {result.win_rate*100:5.1f}% | {pnl:>10}")

        # Run short trades
        if short_config.enabled:
            for indicator in short_config.indicators:
                capital = config.get("initial_capital_short", 1000)
                trades, result = run_backtest_for_symbol(
                    symbol, df, indicator, "short", capital,
                    param_a=short_config.param_a,
                    param_b=short_config.param_b,
                    atr_mult=short_config.atr_mult,
                    min_hold_bars=short_config.min_hold_bars,
                    max_hold_bars=short_config.max_hold_bars,
                    htf_signals=htf_signals_short,
                    use_htf_filter=short_config.htf_filter,
                    htf_indicator=short_config.htf_indicator
                )
                all_trades.extend(trades)
                all_results.append(result)
                pnl = f"+${result.total_pnl:.2f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):.2f}"
                print(f"  {indicator:12} short | {result.total_trades:3} trades | {result.win_rate*100:5.1f}% | {pnl:>10}")

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
    parser = argparse.ArgumentParser(
        description="IB Stock Backtest with separate Long/Short configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (long only)
  python backtest.py

  # Run both long and short with different indicators
  python backtest.py --long-indicator supertrend --short-indicator kama

  # Different ATR stops for long vs short
  python backtest.py --long-atr-mult 1.5 --short-atr-mult 2.5

  # Long with HTF filter, short without
  python backtest.py --long-htf --long-htf-indicator kama

  # Full example with different settings
  python backtest.py --long-indicator supertrend --long-atr-mult 1.5 --long-min-hold 6 --long-htf \\
                     --short-indicator kama --short-atr-mult 2.0 --short-min-hold 12
"""
    )

    # General settings
    parser.add_argument("--period", default="6mo", help="Period: 1mo, 3mo, 6mo, 1y")
    parser.add_argument("--interval", default="1h", help="Interval: 1h, 1d")
    parser.add_argument("--symbols", nargs="+", help="Symbols to test")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")

    # Legacy single-config options (for backward compatibility)
    parser.add_argument("--indicator", help="Single indicator for both directions (legacy)")
    parser.add_argument("--long-only", action="store_true", help="Only run long trades")
    parser.add_argument("--short-only", action="store_true", help="Only run short trades")
    parser.add_argument("--htf", action="store_true", help="Enable HTF filter for both (legacy)")
    parser.add_argument("--htf-indicator", default="kama", help="HTF indicator for both (legacy)")

    # Long-specific settings
    long_group = parser.add_argument_group("Long Trade Settings")
    long_group.add_argument("--long-indicator", nargs="+", help="Indicator(s) for long trades")
    long_group.add_argument("--long-param-a", type=float, help="Param A for long indicator")
    long_group.add_argument("--long-param-b", type=float, help="Param B for long indicator")
    long_group.add_argument("--long-atr-mult", type=float, default=1.5, help="ATR multiplier for long stops (default: 1.5)")
    long_group.add_argument("--long-min-hold", type=int, default=6, help="Min hold bars for long (default: 6)")
    long_group.add_argument("--long-max-hold", type=int, default=0, help="Max hold bars for long (0=no limit)")
    long_group.add_argument("--long-htf", action="store_true", help="Enable HTF filter for long")
    long_group.add_argument("--long-htf-indicator", default="kama", help="HTF indicator for long")

    # Short-specific settings
    short_group = parser.add_argument_group("Short Trade Settings")
    short_group.add_argument("--short-indicator", nargs="+", help="Indicator(s) for short trades")
    short_group.add_argument("--short-param-a", type=float, help="Param A for short indicator")
    short_group.add_argument("--short-param-b", type=float, help="Param B for short indicator")
    short_group.add_argument("--short-atr-mult", type=float, default=2.0, help="ATR multiplier for short stops (default: 2.0)")
    short_group.add_argument("--short-min-hold", type=int, default=8, help="Min hold bars for short (default: 8)")
    short_group.add_argument("--short-max-hold", type=int, default=0, help="Max hold bars for short (0=no limit)")
    short_group.add_argument("--short-htf", action="store_true", help="Enable HTF filter for short")
    short_group.add_argument("--short-htf-indicator", default="kama", help="HTF indicator for short")

    args = parser.parse_args()
    symbols = args.symbols or SYMBOLS

    # Determine if using separate configs or legacy mode
    using_separate_config = any([
        args.long_indicator, args.short_indicator,
        args.long_param_a, args.short_param_a,
        args.long_htf, args.short_htf
    ])

    if using_separate_config or args.long_only or args.short_only:
        # Build separate DirectionConfigs
        default_indicators = ["supertrend"]
        if args.indicator:
            default_indicators = [args.indicator]

        long_config = DirectionConfig(
            enabled=not args.short_only,
            indicators=args.long_indicator or default_indicators,
            param_a=args.long_param_a,
            param_b=args.long_param_b,
            atr_mult=args.long_atr_mult,
            min_hold_bars=args.long_min_hold,
            max_hold_bars=args.long_max_hold,
            htf_filter=args.long_htf or args.htf,
            htf_indicator=args.long_htf_indicator if args.long_htf else args.htf_indicator
        )

        short_config = DirectionConfig(
            enabled=not args.long_only,
            indicators=args.short_indicator or default_indicators,
            param_a=args.short_param_a,
            param_b=args.short_param_b,
            atr_mult=args.short_atr_mult,
            min_hold_bars=args.short_min_hold,
            max_hold_bars=args.short_max_hold,
            htf_filter=args.short_htf or args.htf,
            htf_indicator=args.short_htf_indicator if args.short_htf else args.htf_indicator
        )

        run_full_backtest(symbols, args.period, args.interval,
                          start_date=args.start, end_date=args.end,
                          long_config=long_config, short_config=short_config)
    else:
        # Legacy mode - same settings for both directions
        indicators = [args.indicator] if args.indicator else None
        directions = ["long"] if args.long_only else ["long", "short"]
        run_full_backtest(symbols, args.period, args.interval, indicators, directions,
                          args.start, args.end, args.htf, args.htf_indicator)
