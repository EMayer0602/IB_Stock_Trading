#!/usr/bin/env python3
"""
Enhanced Parameter Sweep for IB Stock Trading.
Optimizes: indicator params, direction, timeframes, HTF filter, hold times.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo
import itertools
import json

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    try:
        import simple_yf as yf
        YF_AVAILABLE = True
        print("[Info] Using simple_yf fallback")
    except ImportError:
        YF_AVAILABLE = False

from tickers_config import TICKERS, SYMBOLS, get_ticker_config
import supertrend_strategy as st

NY_TZ = ZoneInfo("America/New_York")
SWEEP_OUTPUT_DIR = "sweep_results"
FEE_RATE = 0.001


# =============================================================================
# Parameter Grids
# =============================================================================

INDICATOR_PARAMS = {
    "supertrend": {
        "param_a": [7, 10, 14, 20],           # Length
        "param_b": [1.5, 2.0, 2.5, 3.0, 4.0]  # Factor
    },
    "jma": {
        "param_a": [10, 20, 30, 50],          # Length
        "param_b": [-50, 0, 50]               # Phase
    },
    "kama": {
        "param_a": [10, 14, 20, 30],          # Fast length
        "param_b": [20, 30, 40, 50]           # Slow length
    }
}

# Timeframes to test
TIMEFRAMES = ["1h", "2h", "4h"]

# Directions
DIRECTIONS = ["long", "short"]

# ATR stop multipliers
ATR_MULTS = [None, 1.0, 1.5, 2.0, 2.5]

# Minimum hold bars
MIN_HOLD_BARS = [0, 6, 12, 24, 48]

# HTF Filter settings
HTF_CONFIGS = [
    {"enabled": False, "htf": None, "length": None, "factor": None},
    {"enabled": True, "htf": "3h", "length": 14, "factor": 2.5},
    {"enabled": True, "htf": "4h", "length": 14, "factor": 2.0},
    {"enabled": True, "htf": "4h", "length": 20, "factor": 3.0},
    {"enabled": True, "htf": "5h", "length": 14, "factor": 2.5},
    {"enabled": True, "htf": "6h", "length": 14, "factor": 2.0},
    {"enabled": True, "htf": "6h", "length": 20, "factor": 3.0},
    {"enabled": True, "htf": "8h", "length": 14, "factor": 2.5},
    {"enabled": True, "htf": "8h", "length": 20, "factor": 2.0},
    {"enabled": True, "htf": "12h", "length": 10, "factor": 3.0},
    {"enabled": True, "htf": "12h", "length": 20, "factor": 2.5},
    {"enabled": True, "htf": "1d", "length": 10, "factor": 3.0},
    {"enabled": True, "htf": "1d", "length": 20, "factor": 2.5},
]


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_data(symbol: str, period: str = "1y", interval: str = "1h") -> Optional[pd.DataFrame]:
    """Fetch historical data with ATR calculation."""
    if not YF_AVAILABLE:
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
        print(f"[Sweep] Error fetching {symbol} {interval}: {e}")
        return None


def resample_to_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample 1h data to higher timeframe."""
    if target_tf == "1h":
        return df

    tf_map = {
        "2h": "2h", "3h": "3h", "4h": "4h", "5h": "5h",
        "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D"
    }
    rule = tf_map.get(target_tf, target_tf.lower())

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    resampled["atr"] = st.calculate_atr(resampled, st.ATR_WINDOW)
    return resampled


# =============================================================================
# Backtest Engine
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    indicator: str,
    direction: str,
    param_a: float,
    param_b: float,
    atr_mult: Optional[float],
    min_hold_bars: int,
    htf_df: Optional[pd.DataFrame] = None,
    htf_config: Optional[Dict] = None,
    initial_capital: float = 1000
) -> Dict:
    """Run a single backtest with specific parameters."""

    # Generate signals
    signals = st.generate_indicator_signals(df, indicator, param_a, param_b)

    # HTF filter signals (if enabled)
    htf_signals = None
    if htf_config and htf_config.get("enabled") and htf_df is not None:
        htf_signals = st.generate_indicator_signals(
            htf_df, indicator,
            htf_config.get("length", param_a),
            htf_config.get("factor", param_b)
        )
        # Reindex to match main timeframe
        htf_signals = htf_signals.reindex(df.index, method='ffill')

    equity = initial_capital
    position = None
    trades = []
    max_equity = equity
    max_drawdown = 0
    equity_curve = [equity]

    for i in range(1, len(df)):
        current_price = df["close"].iloc[i]
        current_signal = signals.iloc[i]
        prev_signal = signals.iloc[i - 1]
        current_atr = df["atr"].iloc[i] if "atr" in df.columns else 0

        # HTF filter check
        htf_aligned = True
        if htf_signals is not None:
            htf_signal = htf_signals.iloc[i] if i < len(htf_signals) else 0
            if direction == "long" and htf_signal != 1:
                htf_aligned = False
            elif direction == "short" and htf_signal != -1:
                htf_aligned = False

        # Track drawdown
        if equity > max_equity:
            max_equity = equity
        dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        # Exit logic
        if position is not None:
            position["bars_held"] += 1
            should_exit = False

            if position["bars_held"] >= min_hold_bars:
                # Trend flip exit
                if direction == "long" and current_signal == -1:
                    should_exit = True
                elif direction == "short" and current_signal == 1:
                    should_exit = True

                # ATR stop
                if not should_exit and atr_mult and current_atr > 0:
                    if direction == "long":
                        stop_price = position["entry_price"] - atr_mult * current_atr
                        if current_price < stop_price:
                            should_exit = True
                    else:
                        stop_price = position["entry_price"] + atr_mult * current_atr
                        if current_price > stop_price:
                            should_exit = True

            if should_exit:
                if direction == "long":
                    pnl = (current_price - position["entry_price"]) * position["quantity"]
                else:
                    pnl = (position["entry_price"] - current_price) * position["quantity"]
                fees = position["stake"] * FEE_RATE * 2
                equity += pnl - fees
                trades.append({"pnl": pnl - fees, "win": pnl > fees})
                position = None

        # Entry logic
        if position is None and htf_aligned:
            should_enter = False
            if direction == "long" and current_signal == 1 and prev_signal != 1:
                should_enter = True
            elif direction == "short" and current_signal == -1 and prev_signal != -1:
                should_enter = True

            if should_enter and equity > 100:
                stake = min(equity * 0.9, initial_capital)
                quantity = int(stake / current_price)
                if quantity > 0:
                    position = {
                        "entry_price": current_price,
                        "quantity": quantity,
                        "stake": stake,
                        "bars_held": 0
                    }

        equity_curve.append(equity)

    # Close remaining position
    if position is not None:
        current_price = df["close"].iloc[-1]
        if direction == "long":
            pnl = (current_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - current_price) * position["quantity"]
        fees = position["stake"] * FEE_RATE * 2
        equity += pnl - fees
        trades.append({"pnl": pnl - fees, "win": pnl > fees})

    total_trades = len(trades)
    wins = sum(1 for t in trades if t["win"])
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        "total_trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "final_equity": equity,
        "return_pct": (equity - initial_capital) / initial_capital * 100,
        "max_drawdown": max_drawdown * 100
    }


# =============================================================================
# Parameter Sweep
# =============================================================================

def run_full_sweep(
    symbol: str,
    df_1h: pd.DataFrame,
    indicators: List[str] = None,
    directions: List[str] = None,
    timeframes: List[str] = None,
    initial_capital: float = 1000,
    quick_mode: bool = False,
    show_progress: bool = True
) -> List[Dict]:
    """Run comprehensive parameter sweep."""

    if indicators is None:
        indicators = list(INDICATOR_PARAMS.keys())
    if directions is None:
        directions = DIRECTIONS
    if timeframes is None:
        timeframes = TIMEFRAMES if not quick_mode else ["1h"]

    # Pre-compute resampled dataframes
    df_cache = {"1h": df_1h}
    for tf in timeframes:
        if tf not in df_cache:
            df_cache[tf] = resample_to_timeframe(df_1h, tf)

    # Pre-compute HTF dataframes
    htf_cache = {}
    for htf_config in HTF_CONFIGS:
        if htf_config["enabled"]:
            htf = htf_config["htf"]
            if htf not in htf_cache:
                htf_cache[htf] = resample_to_timeframe(df_1h, htf)

    results = []

    # Reduced grids for quick mode
    atr_mults = ATR_MULTS if not quick_mode else [1.5, 2.0, None]
    min_holds = MIN_HOLD_BARS if not quick_mode else [0, 12, 24]
    htf_configs = HTF_CONFIGS if not quick_mode else [HTF_CONFIGS[0], HTF_CONFIGS[2]]

    # Calculate total combinations for progress
    total_combinations = 0
    for indicator in indicators:
        params = INDICATOR_PARAMS[indicator]
        param_combos = len(params["param_a"]) * len(params["param_b"])
        total_combinations += (param_combos * len(directions) * len(timeframes) *
                               len(atr_mults) * len(min_holds) * len(htf_configs))

    current = 0
    last_pct = -1

    for indicator in indicators:
        params = INDICATOR_PARAMS[indicator]
        param_a_vals = params["param_a"]
        param_b_vals = params["param_b"]

        for direction in directions:
            for timeframe in timeframes:
                df = df_cache[timeframe]

                for param_a, param_b in itertools.product(param_a_vals, param_b_vals):
                    for atr_mult in atr_mults:
                        for min_hold in min_holds:
                            for htf_config in htf_configs:
                                # Get HTF dataframe if needed
                                htf_df = None
                                if htf_config["enabled"]:
                                    htf_df = htf_cache.get(htf_config["htf"])

                                result = run_backtest(
                                    df=df,
                                    indicator=indicator,
                                    direction=direction,
                                    param_a=param_a,
                                    param_b=param_b,
                                    atr_mult=atr_mult,
                                    min_hold_bars=min_hold,
                                    htf_df=htf_df,
                                    htf_config=htf_config,
                                    initial_capital=initial_capital
                                )

                                result.update({
                                    "symbol": symbol,
                                    "indicator": indicator,
                                    "direction": direction,
                                    "timeframe": timeframe,
                                    "param_a": param_a,
                                    "param_b": param_b,
                                    "atr_mult": atr_mult if atr_mult else 0,
                                    "min_hold_bars": min_hold,
                                    "htf_enabled": htf_config["enabled"],
                                    "htf_timeframe": htf_config.get("htf", ""),
                                    "htf_length": htf_config.get("length", 0),
                                    "htf_factor": htf_config.get("factor", 0)
                                })

                                results.append(result)

                                # Progress update
                                current += 1
                                if show_progress:
                                    pct = int(current * 100 / total_combinations)
                                    # Update every 1% or every 500 iterations for large sweeps
                                    should_update = (pct != last_pct) or (current % 500 == 0)
                                    if should_update:
                                        best_so_far = max((r["total_pnl"] for r in results), default=0)
                                        print(f"\r  [{symbol}] {pct:3d}% ({current:,}/{total_combinations:,}) "
                                              f"Best: ${best_so_far:+.2f}    ", end="", flush=True)
                                        last_pct = pct

    if show_progress:
        print()  # Newline after progress

    return results


def find_best_per_symbol(results: List[Dict]) -> Dict[str, Dict]:
    """Find best parameters per symbol (considering both long and short)."""
    best = {}

    for r in results:
        symbol = r["symbol"]
        direction = r["direction"]
        key = f"{symbol}_{direction}"

        if key not in best or r["total_pnl"] > best[key]["total_pnl"]:
            best[key] = r

    return best


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced IB Stock Parameter Sweep")
    parser.add_argument("--period", default="1y", help="Data period: 6mo, 1y, 2y")
    parser.add_argument("--symbols", nargs="+", help="Symbols to test")
    parser.add_argument("--indicator", help="Single indicator (default: all)")
    parser.add_argument("--direction", choices=["long", "short"], help="Single direction")
    parser.add_argument("--quick", action="store_true", help="Quick mode (reduced grid)")
    parser.add_argument("--top", type=int, default=15, help="Show top N results")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    indicators = [args.indicator] if args.indicator else None
    directions = [args.direction] if args.direction else None

    # Calculate estimated combinations
    est_per_symbol = 0
    for ind in (indicators or list(INDICATOR_PARAMS.keys())):
        params = INDICATOR_PARAMS[ind]
        param_combos = len(params["param_a"]) * len(params["param_b"])
        dirs = directions or DIRECTIONS
        tfs = TIMEFRAMES if not args.quick else ["1h"]
        atrs = ATR_MULTS if not args.quick else [1.5, 2.0, None]
        holds = MIN_HOLD_BARS if not args.quick else [0, 12, 24]
        htfs = HTF_CONFIGS if not args.quick else [HTF_CONFIGS[0], HTF_CONFIGS[2]]
        est_per_symbol += param_combos * len(dirs) * len(tfs) * len(atrs) * len(holds) * len(htfs)

    total_est = est_per_symbol * len(symbols)

    print(f"\n{'='*80}")
    print(f"ENHANCED PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"Symbols: {len(symbols)} | Period: {args.period} | Quick: {args.quick}")
    print(f"Indicators: {indicators or 'all'} | Directions: {directions or 'all'}")
    print(f"Timeframes: {TIMEFRAMES if not args.quick else ['1h']} | HTF Configs: {len(HTF_CONFIGS) if not args.quick else 2}")
    print(f"Estimated combinations: {total_est:,} ({est_per_symbol:,} per symbol)")
    if not args.quick and total_est > 100000:
        print(f"⚠️  Large sweep! Consider using --quick for faster results")
    print(f"{'='*80}\n")

    all_results = []
    import time
    start_time = time.time()

    for idx, symbol in enumerate(symbols, 1):
        config = get_ticker_config(symbol)
        capital = config.get("initial_capital_long", 1000)

        # Calculate ETA
        elapsed = time.time() - start_time
        if idx > 1:
            avg_per_symbol = elapsed / (idx - 1)
            remaining = avg_per_symbol * (len(symbols) - idx + 1)
            eta_min = int(remaining / 60)
            eta_sec = int(remaining % 60)
            eta_str = f" | ETA: {eta_min}m {eta_sec}s"
        else:
            eta_str = ""

        print(f"\n[{idx}/{len(symbols)}] {symbol} - Fetching data...{eta_str}")
        df_1h = fetch_data(symbol, args.period, "1h")

        if df_1h is None or len(df_1h) < 100:
            print(f"  Skipped - insufficient data")
            continue

        symbol_start = time.time()
        print(f"  Running sweep ({len(df_1h)} bars)...")
        results = run_full_sweep(
            symbol=symbol,
            df_1h=df_1h,
            indicators=indicators,
            directions=directions,
            initial_capital=capital,
            quick_mode=args.quick
        )
        all_results.extend(results)
        symbol_time = time.time() - symbol_start

        # Show best for this symbol
        best_long = max([r for r in results if r["direction"] == "long"],
                       key=lambda x: x["total_pnl"], default=None)
        best_short = max([r for r in results if r["direction"] == "short"],
                        key=lambda x: x["total_pnl"], default=None)

        print(f"  Completed in {symbol_time:.1f}s ({len(results)} combinations)")
        if best_long:
            print(f"  LONG:  {best_long['indicator']:10} {best_long['timeframe']:3} "
                  f"A={best_long['param_a']:4} B={best_long['param_b']:5} "
                  f"-> ${best_long['total_pnl']:+8.2f} ({best_long['return_pct']:+5.1f}%)")
        if best_short:
            print(f"  SHORT: {best_short['indicator']:10} {best_short['timeframe']:3} "
                  f"A={best_short['param_a']:4} B={best_short['param_b']:5} "
                  f"-> ${best_short['total_pnl']:+8.2f} ({best_short['return_pct']:+5.1f}%)")

    if not all_results:
        print("No results to save.")
        return

    # Save all results
    Path(SWEEP_OUTPUT_DIR).mkdir(exist_ok=True)
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"{SWEEP_OUTPUT_DIR}/sweep_full_{timestamp}.csv", index=False)

    # Show top results
    print(f"\n{'='*80}")
    print(f"TOP {args.top} PARAMETER COMBINATIONS (by P&L)")
    print(f"{'='*80}")

    sorted_results = sorted(all_results, key=lambda x: x["total_pnl"], reverse=True)

    for i, r in enumerate(sorted_results[:args.top], 1):
        htf_str = f"HTF:{r['htf_timeframe']}" if r['htf_enabled'] else "no-HTF"
        print(f"{i:2}. {r['symbol']:5} {r['direction']:5} {r['indicator']:10} {r['timeframe']:3} | "
              f"A={r['param_a']:4} B={r['param_b']:5} ATR={r['atr_mult']:3} Hold={r['min_hold_bars']:2} {htf_str:8} | "
              f"{r['total_trades']:3}T WR={r['win_rate']*100:4.1f}% ${r['total_pnl']:+8.2f}")

    # Best per symbol/direction
    print(f"\n{'='*80}")
    print("BEST PARAMETERS PER SYMBOL")
    print(f"{'='*80}")

    best_configs = []
    for symbol in symbols:
        symbol_results = [r for r in all_results if r["symbol"] == symbol]
        if not symbol_results:
            continue

        # Best overall for this symbol (long or short)
        best = max(symbol_results, key=lambda x: x["total_pnl"])
        best_configs.append(best)

        status = "+" if best["total_pnl"] > 0 else "-"
        htf_str = f"HTF:{best['htf_timeframe']}" if best['htf_enabled'] else ""
        print(f"{status} {best['symbol']:5} {best['direction']:5} {best['indicator']:10} {best['timeframe']:3} | "
              f"A={best['param_a']:4} B={best['param_b']:5} ATR={best['atr_mult']:3} Hold={best['min_hold_bars']:2} {htf_str:8} | "
              f"${best['total_pnl']:+8.2f} ({best['return_pct']:+5.1f}%) WR={best['win_rate']*100:.0f}%")

    # Save best configs
    best_df = pd.DataFrame(best_configs)
    best_df.to_csv(f"{SWEEP_OUTPUT_DIR}/best_params_{timestamp}.csv", index=False)

    # Generate config update suggestion
    print(f"\n{'='*80}")
    print("SUGGESTED tickers_config.py UPDATE:")
    print(f"{'='*80}")

    for b in best_configs:
        if b["total_pnl"] > 0:
            print(f'    "{b["symbol"]}": {{"indicator": "{b["indicator"]}", '
                  f'"direction": "{b["direction"]}", "timeframe": "{b["timeframe"]}", '
                  f'"param_a": {b["param_a"]}, "param_b": {b["param_b"]}, '
                  f'"atr_mult": {b["atr_mult"]}, "min_hold": {b["min_hold_bars"]}}},')

    # Summary
    total_pnl = sum(r["total_pnl"] for r in best_configs)
    profitable = sum(1 for r in best_configs if r["total_pnl"] > 0)
    total_time = time.time() - start_time
    total_min = int(total_time / 60)
    total_sec = int(total_time % 60)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {profitable}/{len(best_configs)} profitable symbols")
    print(f"Total P&L with best params: ${total_pnl:+,.2f}")
    print(f"Total combinations tested: {len(all_results):,}")
    print(f"Total runtime: {total_min}m {total_sec}s")
    print(f"Results saved to {SWEEP_OUTPUT_DIR}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
