#!/usr/bin/env python3
"""
Parameter Optimization for IB Stock Trading Strategy.

Tests different parameter combinations for long and short trades
to find optimal settings for each direction.

Usage: python optimize_parameters.py
       python optimize_parameters.py --symbols AAPL MSFT --quick
"""

import argparse
import itertools
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

from backtest import (
    run_full_backtest, DirectionConfig, fetch_historical_data,
    run_backtest_for_symbol, SYMBOLS
)
from tickers_config import get_ticker_config
import supertrend_strategy as st


@dataclass
class ParameterSet:
    """A set of parameters to test."""
    indicator: str
    atr_mult: float
    min_hold: int
    htf_filter: bool
    htf_indicator: str = "kama"
    param_a: float = None
    param_b: float = None

    def __str__(self):
        htf = f"+HTF({self.htf_indicator})" if self.htf_filter else ""
        return f"{self.indicator} ATR×{self.atr_mult} Hold={self.min_hold}{htf}"


def generate_parameter_grid(quick=False):
    """Generate parameter combinations to test."""
    if quick:
        # Quick mode - fewer combinations
        indicators = ["supertrend", "kama"]
        atr_mults = [1.5, 2.5]
        min_holds = [6, 12]
        htf_options = [False, True]
    else:
        # Full mode - comprehensive search
        indicators = ["supertrend", "kama", "jma", "psar"]
        atr_mults = [1.0, 1.5, 2.0, 2.5, 3.0]
        min_holds = [4, 6, 8, 12, 18]
        htf_options = [False, True]

    param_sets = []
    for indicator, atr_mult, min_hold, htf in itertools.product(
        indicators, atr_mults, min_holds, htf_options
    ):
        param_sets.append(ParameterSet(
            indicator=indicator,
            atr_mult=atr_mult,
            min_hold=min_hold,
            htf_filter=htf,
            htf_indicator="kama"
        ))

    return param_sets


def test_parameters(symbols: List[str], param_set: ParameterSet,
                    direction: str, period: str = "6mo") -> Dict[str, Any]:
    """Test a single parameter set and return results."""
    all_trades = []
    all_results = []

    for symbol in symbols:
        config = get_ticker_config(symbol)
        df = fetch_historical_data(symbol, period, "1h")
        if df is None or len(df) < 50:
            continue

        # Get HTF signals if needed
        htf_signals = None
        if param_set.htf_filter:
            htf_df = fetch_historical_data(symbol, period, "1d")
            if htf_df is not None and len(htf_df) > 10:
                htf_preset = st.INDICATOR_PRESETS.get(param_set.htf_indicator, {})
                htf_param_a = htf_preset.get("default_a", 14)
                htf_param_b = htf_preset.get("default_b", 30)
                htf_signals = st.generate_indicator_signals(
                    htf_df, param_set.htf_indicator, htf_param_a, htf_param_b
                )

        capital = config.get(f"initial_capital_{direction}", 1000)
        trades, result = run_backtest_for_symbol(
            symbol, df, param_set.indicator, direction, capital,
            param_a=param_set.param_a,
            param_b=param_set.param_b,
            atr_mult=param_set.atr_mult,
            min_hold_bars=param_set.min_hold,
            htf_signals=htf_signals,
            use_htf_filter=param_set.htf_filter
        )
        all_trades.extend(trades)
        all_results.append(result)

    if not all_results:
        return None

    total_trades = sum(r.total_trades for r in all_results)
    total_pnl = sum(r.total_pnl for r in all_results)
    total_capital = sum(r.start_capital for r in all_results)
    winning = sum(r.winning_trades for r in all_results)
    win_rate = winning / total_trades if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    return_pct = (total_pnl / total_capital * 100) if total_capital > 0 else 0

    # Calculate profit factor
    gross_profit = sum(t.net_pnl for t in all_trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in all_trades if t.net_pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        "params": param_set,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "avg_pnl": avg_pnl,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss
    }


def optimize(symbols: List[str] = None, quick: bool = False, period: str = "6mo"):
    """Run full parameter optimization."""
    if symbols is None:
        symbols = SYMBOLS

    param_sets = generate_parameter_grid(quick)
    total_tests = len(param_sets) * 2  # long + short

    print(f"\n{'='*70}")
    print("PARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {period}")
    print(f"Parameter combinations: {len(param_sets)}")
    print(f"Total tests: {total_tests} (long + short)")
    print(f"{'='*70}\n")

    # Test all parameter combinations
    long_results = []
    short_results = []

    print("Testing LONG parameters...")
    for i, param_set in enumerate(param_sets):
        result = test_parameters(symbols, param_set, "long", period)
        if result:
            long_results.append(result)
        print(f"  [{i+1}/{len(param_sets)}] {param_set} -> ", end="")
        if result:
            print(f"PnL: ${result['total_pnl']:+.2f} | WR: {result['win_rate']*100:.1f}% | PF: {result['profit_factor']:.2f}")
        else:
            print("No data")

    print("\nTesting SHORT parameters...")
    for i, param_set in enumerate(param_sets):
        result = test_parameters(symbols, param_set, "short", period)
        if result:
            short_results.append(result)
        print(f"  [{i+1}/{len(param_sets)}] {param_set} -> ", end="")
        if result:
            print(f"PnL: ${result['total_pnl']:+.2f} | WR: {result['win_rate']*100:.1f}% | PF: {result['profit_factor']:.2f}")
        else:
            print("No data")

    # Sort by total P&L
    long_results.sort(key=lambda x: x['total_pnl'], reverse=True)
    short_results.sort(key=lambda x: x['total_pnl'], reverse=True)

    # Display results
    print(f"\n{'='*70}")
    print("TOP 10 LONG PARAMETER SETS (by P&L)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Parameters':<45} {'Trades':>7} {'WinRate':>8} {'P&L':>12} {'PF':>6}")
    print("-" * 85)
    for i, r in enumerate(long_results[:10]):
        print(f"{i+1:<5} {str(r['params']):<45} {r['total_trades']:>7} {r['win_rate']*100:>7.1f}% ${r['total_pnl']:>10.2f} {r['profit_factor']:>6.2f}")

    print(f"\n{'='*70}")
    print("TOP 10 SHORT PARAMETER SETS (by P&L)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Parameters':<45} {'Trades':>7} {'WinRate':>8} {'P&L':>12} {'PF':>6}")
    print("-" * 85)
    for i, r in enumerate(short_results[:10]):
        print(f"{i+1:<5} {str(r['params']):<45} {r['total_trades']:>7} {r['win_rate']*100:>7.1f}% ${r['total_pnl']:>10.2f} {r['profit_factor']:>6.2f}")

    # Best parameters
    print(f"\n{'='*70}")
    print("RECOMMENDED SETTINGS")
    print(f"{'='*70}")

    if long_results:
        best_long = long_results[0]
        print(f"\nBEST LONG:")
        print(f"  Indicator:    {best_long['params'].indicator}")
        print(f"  ATR Mult:     {best_long['params'].atr_mult}")
        print(f"  Min Hold:     {best_long['params'].min_hold}")
        print(f"  HTF Filter:   {best_long['params'].htf_filter} ({best_long['params'].htf_indicator})")
        print(f"  ---")
        print(f"  Total P&L:    ${best_long['total_pnl']:+,.2f}")
        print(f"  Win Rate:     {best_long['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {best_long['profit_factor']:.2f}")

    if short_results:
        best_short = short_results[0]
        print(f"\nBEST SHORT:")
        print(f"  Indicator:    {best_short['params'].indicator}")
        print(f"  ATR Mult:     {best_short['params'].atr_mult}")
        print(f"  Min Hold:     {best_short['params'].min_hold}")
        print(f"  HTF Filter:   {best_short['params'].htf_filter} ({best_short['params'].htf_indicator})")
        print(f"  ---")
        print(f"  Total P&L:    ${best_short['total_pnl']:+,.2f}")
        print(f"  Win Rate:     {best_short['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {best_short['profit_factor']:.2f}")

    # Generate command
    if long_results and short_results:
        bl = best_long['params']
        bs = best_short['params']
        print(f"\n{'='*70}")
        print("BACKTEST COMMAND WITH OPTIMAL SETTINGS")
        print(f"{'='*70}")
        cmd = f"""python backtest.py \\
  --long-indicator {bl.indicator} --long-atr-mult {bl.atr_mult} --long-min-hold {bl.min_hold}"""
        if bl.htf_filter:
            cmd += f" --long-htf --long-htf-indicator {bl.htf_indicator}"
        cmd += f""" \\
  --short-indicator {bs.indicator} --short-atr-mult {bs.atr_mult} --short-min-hold {bs.min_hold}"""
        if bs.htf_filter:
            cmd += f" --short-htf --short-htf-indicator {bs.htf_indicator}"
        print(cmd)

    # Save results to CSV
    if long_results:
        long_df = pd.DataFrame([{
            'indicator': r['params'].indicator,
            'atr_mult': r['params'].atr_mult,
            'min_hold': r['params'].min_hold,
            'htf_filter': r['params'].htf_filter,
            'total_trades': r['total_trades'],
            'win_rate': r['win_rate'],
            'total_pnl': r['total_pnl'],
            'profit_factor': r['profit_factor']
        } for r in long_results])
        long_df.to_csv('optimization_long.csv', index=False)

    if short_results:
        short_df = pd.DataFrame([{
            'indicator': r['params'].indicator,
            'atr_mult': r['params'].atr_mult,
            'min_hold': r['params'].min_hold,
            'htf_filter': r['params'].htf_filter,
            'total_trades': r['total_trades'],
            'win_rate': r['win_rate'],
            'total_pnl': r['total_pnl'],
            'profit_factor': r['profit_factor']
        } for r in short_results])
        short_df.to_csv('optimization_short.csv', index=False)

    print(f"\nResults saved to optimization_long.csv and optimization_short.csv")

    return long_results, short_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Optimization")
    parser.add_argument("--symbols", nargs="+", help="Symbols to test")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer combinations)")
    parser.add_argument("--period", default="6mo", help="Data period")
    args = parser.parse_args()

    optimize(args.symbols, args.quick, args.period)
