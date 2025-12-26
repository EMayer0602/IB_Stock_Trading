#!/usr/bin/env python3
"""
Parameter Sweep for IB Stock Trading - Find optimal parameters.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
from zoneinfo import ZoneInfo
import itertools

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from tickers_config import TICKERS, SYMBOLS, get_ticker_config
import supertrend_strategy as st

NY_TZ = ZoneInfo("America/New_York")
SWEEP_OUTPUT_DIR = "sweep_results"
FEE_RATE = 0.001


def fetch_historical_data(symbol: str, period: str = "1y", interval: str = "1h") -> Optional[pd.DataFrame]:
    """Fetch historical data using yfinance."""
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
        print(f"[Sweep] Error fetching {symbol}: {e}")
        return None


def run_single_backtest(
    df: pd.DataFrame,
    indicator: str,
    param_a: float,
    param_b: float,
    atr_mult: float,
    min_hold_bars: int,
    initial_capital: float = 1000
) -> Dict:
    """Run a single backtest with specific parameters."""

    signals = st.generate_indicator_signals(df, indicator, param_a, param_b)

    equity = initial_capital
    position = None
    trades = []
    max_equity = equity
    max_drawdown = 0

    for i in range(1, len(df)):
        current_price = df["close"].iloc[i]
        current_signal = signals.iloc[i]
        prev_signal = signals.iloc[i - 1]
        current_atr = df["atr"].iloc[i] if "atr" in df.columns else 0

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
                # Trend flip
                if current_signal == -1:
                    should_exit = True
                # ATR stop
                elif atr_mult and current_atr > 0:
                    stop_price = position["entry_price"] - atr_mult * current_atr
                    if current_price < stop_price:
                        should_exit = True

            if should_exit:
                pnl = (current_price - position["entry_price"]) * position["quantity"]
                fees = position["stake"] * FEE_RATE * 2
                equity += pnl - fees
                trades.append({"pnl": pnl - fees, "win": pnl > fees})
                position = None

        # Entry logic (long only)
        if position is None and current_signal == 1 and prev_signal != 1:
            if equity > 100:
                stake = min(equity * 0.9, initial_capital)
                quantity = int(stake / current_price)
                if quantity > 0:
                    position = {
                        "entry_price": current_price,
                        "quantity": quantity,
                        "stake": stake,
                        "bars_held": 0
                    }

    # Close remaining position
    if position is not None:
        current_price = df["close"].iloc[-1]
        pnl = (current_price - position["entry_price"]) * position["quantity"]
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


def run_parameter_sweep(
    symbol: str,
    df: pd.DataFrame,
    indicator: str = "supertrend",
    initial_capital: float = 1000
) -> List[Dict]:
    """Run parameter sweep for a symbol."""

    results = []

    # Parameter ranges based on indicator
    if indicator == "supertrend":
        param_a_values = [7, 10, 14, 20]
        param_b_values = [1.5, 2.0, 2.5, 3.0, 4.0]
    elif indicator == "jma":
        param_a_values = [10, 20, 30, 50]
        param_b_values = [-50, 0, 50]
    elif indicator == "kama":
        param_a_values = [10, 14, 20, 30]
        param_b_values = [20, 30, 40, 50]
    else:
        param_a_values = [10, 20]
        param_b_values = [0]

    atr_mult_values = [1.0, 1.5, 2.0, 2.5, None]
    min_hold_values = [0, 6, 12, 24]

    total_combos = len(param_a_values) * len(param_b_values) * len(atr_mult_values) * len(min_hold_values)

    for param_a, param_b, atr_mult, min_hold in itertools.product(
        param_a_values, param_b_values, atr_mult_values, min_hold_values
    ):
        result = run_single_backtest(
            df, indicator, param_a, param_b, atr_mult, min_hold, initial_capital
        )

        result.update({
            "symbol": symbol,
            "indicator": indicator,
            "param_a": param_a,
            "param_b": param_b,
            "atr_mult": atr_mult if atr_mult else 0,
            "min_hold_bars": min_hold
        })

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="IB Stock Parameter Sweep")
    parser.add_argument("--period", default="1y", help="Data period")
    parser.add_argument("--symbols", nargs="+", help="Symbols to test")
    parser.add_argument("--indicator", default="jma", help="Indicator to optimize")
    parser.add_argument("--top", type=int, default=10, help="Show top N results")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    indicator = args.indicator

    print(f"\n{'='*70}")
    print(f"IB STOCK TRADING - PARAMETER SWEEP")
    print(f"Indicator: {indicator} | Period: {args.period} | Symbols: {len(symbols)}")
    print(f"{'='*70}\n")

    all_results = []

    for symbol in symbols:
        config = get_ticker_config(symbol)
        capital = config.get("initial_capital_long", 1000)

        print(f"[{symbol}] Fetching data...")
        df = fetch_historical_data(symbol, args.period, "1h")

        if df is None or len(df) < 100:
            print(f"[{symbol}] Skipped - insufficient data")
            continue

        print(f"[{symbol}] Running sweep ({len(df)} bars)...")
        results = run_parameter_sweep(symbol, df, indicator, capital)
        all_results.extend(results)

        # Show best for this symbol
        best = max(results, key=lambda x: x["total_pnl"])
        print(f"[{symbol}] Best: ParamA={best['param_a']}, ParamB={best['param_b']}, "
              f"ATR={best['atr_mult']}, Hold={best['min_hold_bars']} -> ${best['total_pnl']:+.2f} ({best['return_pct']:+.1f}%)")

    # Save all results
    Path(SWEEP_OUTPUT_DIR).mkdir(exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{SWEEP_OUTPUT_DIR}/sweep_{indicator}.csv", index=False)

    # Show top results
    print(f"\n{'='*70}")
    print(f"TOP {args.top} PARAMETER COMBINATIONS")
    print(f"{'='*70}")

    sorted_results = sorted(all_results, key=lambda x: x["total_pnl"], reverse=True)

    for i, r in enumerate(sorted_results[:args.top], 1):
        print(f"{i:2}. {r['symbol']:6} | A={r['param_a']:4}, B={r['param_b']:5}, "
              f"ATR={r['atr_mult']:3}, Hold={r['min_hold_bars']:2} | "
              f"{r['total_trades']:3} trades | WR={r['win_rate']*100:4.1f}% | "
              f"${r['total_pnl']:+8.2f} | DD={r['max_drawdown']:4.1f}%")

    # Summary by symbol
    print(f"\n{'='*70}")
    print("BEST PARAMETERS PER SYMBOL (Long Only)")
    print(f"{'='*70}")

    best_params = []
    for symbol in symbols:
        symbol_results = [r for r in all_results if r["symbol"] == symbol]
        if symbol_results:
            best = max(symbol_results, key=lambda x: x["total_pnl"])
            best_params.append(best)
            status = "✓" if best["total_pnl"] > 0 else "✗"
            print(f"{status} {symbol:6} | {indicator} A={best['param_a']:4} B={best['param_b']:5} "
                  f"ATR={best['atr_mult']:3} Hold={best['min_hold_bars']:2} | "
                  f"${best['total_pnl']:+8.2f} ({best['return_pct']:+5.1f}%) | WR={best['win_rate']*100:.0f}%")

    # Save best params
    best_df = pd.DataFrame(best_params)
    best_df.to_csv(f"{SWEEP_OUTPUT_DIR}/best_params_{indicator}.csv", index=False)

    # Total
    total_pnl = sum(r["total_pnl"] for r in best_params)
    total_capital = sum(get_ticker_config(r["symbol"]).get("initial_capital_long", 1000) for r in best_params)
    profitable = sum(1 for r in best_params if r["total_pnl"] > 0)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {profitable}/{len(best_params)} profitable symbols")
    print(f"Total P&L with best params: ${total_pnl:+,.2f} ({total_pnl/total_capital*100:+.1f}%)")
    print(f"Results saved to {SWEEP_OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
