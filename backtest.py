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

from tickers_config import (
    TICKERS, SYMBOLS, ENABLED_SYMBOLS, get_ticker_config,
    get_indicator_for_symbol, get_direction_for_symbol, get_best_strategies
)
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

    # Return equity curve with timestamps
    equity_df = pd.DataFrame({
        'timestamp': df.index[:len(equity_curve)],
        'equity': equity_curve
    })

    return trades, result, equity_df


def generate_equity_chart_html(equity_curves: dict, output_path: str):
    """Generate an interactive HTML chart with all equity curves."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Backtest Equity Curves</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1, h2 { color: #00d4ff; }
        .chart { width: 100%; height: 500px; margin-bottom: 30px; }
        .summary { background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
        th { background: #16213e; color: #00d4ff; }
        tr:hover { background: #1f3460; }
        .filter-btn { padding: 8px 16px; margin: 5px; cursor: pointer; border: none; border-radius: 4px; }
        .filter-btn.active { background: #00d4ff; color: #000; }
        .filter-btn:not(.active) { background: #333; color: #fff; }
    </style>
</head>
<body>
    <h1>Backtest Equity Curves</h1>
    <div class="summary">
        <strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
    </div>

    <div>
        <button class="filter-btn active" onclick="filterCharts('all')">All</button>
        <button class="filter-btn" onclick="filterCharts('long')">Long Only</button>
        <button class="filter-btn" onclick="filterCharts('short')">Short Only</button>
    </div>

    <div id="combined-chart" class="chart"></div>
    <h2>Individual Equity Curves</h2>
"""

    # Generate individual chart divs
    for key in equity_curves.keys():
        html_content += f'    <div id="chart-{key.replace("/", "-")}" class="chart" data-direction="{key.split("/")[2]}"></div>\n'

    html_content += """
    <script>
    var equityData = """ + str({k: {'x': [str(t) for t in v['timestamp'].tolist()], 'y': v['equity'].tolist()} for k, v in equity_curves.items()}) + """;

    // Combined chart
    var combinedTraces = [];
    var colors = ['#00d4ff', '#00ff88', '#ff4757', '#ffd93d', '#6c5ce7', '#a29bfe', '#fd79a8', '#00b894'];
    var i = 0;
    for (var key in equityData) {
        combinedTraces.push({
            x: equityData[key].x,
            y: equityData[key].y,
            name: key,
            type: 'scatter',
            mode: 'lines',
            line: {color: colors[i % colors.length], width: 1.5}
        });
        i++;
    }

    Plotly.newPlot('combined-chart', combinedTraces, {
        title: 'All Equity Curves Combined',
        paper_bgcolor: '#1a1a2e',
        plot_bgcolor: '#16213e',
        font: {color: '#eee'},
        xaxis: {gridcolor: '#333'},
        yaxis: {gridcolor: '#333', title: 'Equity ($)'},
        legend: {orientation: 'h', y: -0.2}
    });

    // Individual charts
    for (var key in equityData) {
        var divId = 'chart-' + key.replace(/\\//g, '-');
        var startVal = equityData[key].y[0];
        var endVal = equityData[key].y[equityData[key].y.length - 1];
        var returnPct = ((endVal - startVal) / startVal * 100).toFixed(2);
        var color = endVal >= startVal ? '#00ff88' : '#ff4757';

        Plotly.newPlot(divId, [{
            x: equityData[key].x,
            y: equityData[key].y,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            line: {color: color, width: 2},
            fillcolor: endVal >= startVal ? 'rgba(0,255,136,0.1)' : 'rgba(255,71,87,0.1)'
        }], {
            title: key + ' (Return: ' + returnPct + '%)',
            paper_bgcolor: '#1a1a2e',
            plot_bgcolor: '#16213e',
            font: {color: '#eee'},
            xaxis: {gridcolor: '#333'},
            yaxis: {gridcolor: '#333', title: 'Equity ($)'}
        });
    }

    function filterCharts(direction) {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');

        document.querySelectorAll('.chart[data-direction]').forEach(div => {
            if (direction === 'all' || div.dataset.direction === direction) {
                div.style.display = 'block';
            } else {
                div.style.display = 'none';
            }
        });
    }
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"[Chart] Equity curves saved to {output_path}")


def run_best_strategies_backtest(period="6mo", interval="1h"):
    """Run backtest using only the best indicator/direction per symbol from config."""
    strategies = get_best_strategies()

    print(f"\n{'='*60}")
    print(f"IB STOCK TRADING BACKTEST - BEST STRATEGIES - {period}")
    print(f"{'='*60}\n")

    all_trades, all_results = [], []
    all_equity_curves = {}

    for strat in strategies:
        symbol = strat["symbol"]
        indicator = strat["indicator"]
        param_a = strat.get("param_a")
        param_b = strat.get("param_b")
        direction = strat["direction"]
        capital = strat["capital"]

        print(f"[{symbol}] Fetching data...")
        df = fetch_historical_data(symbol, period, interval)
        if df is None or len(df) < 50:
            print(f"[{symbol}] Skipped")
            continue
        param_str = f"A={param_a} B={param_b}" if param_a is not None else "default"
        print(f"[{symbol}] {len(df)} bars | {indicator} ({param_str}) | {direction}")

        trades, result, equity_df = run_backtest_for_symbol(
            symbol, df, indicator, direction, capital, param_a=param_a, param_b=param_b
        )
        all_trades.extend(trades)
        all_results.append(result)

        # Store equity curve
        curve_key = f"{symbol}/{indicator}/{direction}"
        all_equity_curves[curve_key] = equity_df

        pnl = f"+${result.total_pnl:.2f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):.2f}"
        print(f"  {result.total_trades:3} trades | {result.win_rate*100:5.1f}% win | {pnl:>10}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_pnl = sum(r.total_pnl for r in all_results)
    total_capital = sum(r.start_capital for r in all_results)
    print(f"Total Trades: {len(all_trades)}")
    print(f"Total P&L: ${total_pnl:+,.2f}")
    print(f"Return: {total_pnl/total_capital*100:+.2f}%")

    # Create output directory
    Path(BACKTEST_OUTPUT_DIR).mkdir(exist_ok=True)

    # Save all trades
    all_trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    all_trades_df.to_csv(f"{BACKTEST_OUTPUT_DIR}/backtest_trades.csv", index=False)

    # Save separate trade lists for long and short
    if not all_trades_df.empty:
        long_trades = all_trades_df[all_trades_df['direction'] == 'long']
        short_trades = all_trades_df[all_trades_df['direction'] == 'short']

        if not long_trades.empty:
            long_trades.to_csv(f"{BACKTEST_OUTPUT_DIR}/trades_long.csv", index=False)
            print(f"[Trades] Long trades: {len(long_trades)} saved to trades_long.csv")

        if not short_trades.empty:
            short_trades.to_csv(f"{BACKTEST_OUTPUT_DIR}/trades_short.csv", index=False)
            print(f"[Trades] Short trades: {len(short_trades)} saved to trades_short.csv")

    # Save results
    pd.DataFrame([asdict(r) for r in all_results]).to_csv(f"{BACKTEST_OUTPUT_DIR}/backtest_results.csv", index=False)

    # Generate equity curve charts
    if all_equity_curves:
        generate_equity_chart_html(all_equity_curves, f"{BACKTEST_OUTPUT_DIR}/equity_curves.html")

    print(f"\nResults saved to {BACKTEST_OUTPUT_DIR}/")

    return all_trades, all_results, all_equity_curves


def run_full_backtest(symbols=None, period="6mo", interval="1h", indicators=None, directions=None):
    """Run backtest with all indicator/direction combinations (sweep mode)."""
    if symbols is None:
        symbols = SYMBOLS
    if indicators is None:
        indicators = ["supertrend", "jma", "kama"]
    if directions is None:
        directions = ["long"]

    print(f"\n{'='*60}")
    print(f"IB STOCK TRADING BACKTEST - SWEEP MODE - {period}")
    print(f"{'='*60}\n")

    all_trades, all_results = [], []
    all_equity_curves = {}

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
                trades, result, equity_df = run_backtest_for_symbol(symbol, df, indicator, direction, capital)
                all_trades.extend(trades)
                all_results.append(result)

                # Store equity curve
                curve_key = f"{symbol}/{indicator}/{direction}"
                all_equity_curves[curve_key] = equity_df

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

    # Create output directory
    Path(BACKTEST_OUTPUT_DIR).mkdir(exist_ok=True)

    # Save all trades
    all_trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    all_trades_df.to_csv(f"{BACKTEST_OUTPUT_DIR}/backtest_trades.csv", index=False)

    # Save separate trade lists for long and short
    if not all_trades_df.empty:
        long_trades = all_trades_df[all_trades_df['direction'] == 'long']
        short_trades = all_trades_df[all_trades_df['direction'] == 'short']

        if not long_trades.empty:
            long_trades.to_csv(f"{BACKTEST_OUTPUT_DIR}/trades_long.csv", index=False)
            print(f"[Trades] Long trades: {len(long_trades)} saved to trades_long.csv")

        if not short_trades.empty:
            short_trades.to_csv(f"{BACKTEST_OUTPUT_DIR}/trades_short.csv", index=False)
            print(f"[Trades] Short trades: {len(short_trades)} saved to trades_short.csv")

    # Save results
    pd.DataFrame([asdict(r) for r in all_results]).to_csv(f"{BACKTEST_OUTPUT_DIR}/backtest_results.csv", index=False)

    # Generate equity curve charts
    if all_equity_curves:
        generate_equity_chart_html(all_equity_curves, f"{BACKTEST_OUTPUT_DIR}/equity_curves.html")

    print(f"\nResults saved to {BACKTEST_OUTPUT_DIR}/")

    return all_trades, all_results, all_equity_curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IB Stock Backtest")
    parser.add_argument("--period", default="6mo", help="Period: 1mo, 3mo, 6mo, 1y")
    parser.add_argument("--interval", default="1h", help="Interval: 1h, 1d")
    parser.add_argument("--symbols", nargs="+", help="Symbols to test (sweep mode)")
    parser.add_argument("--indicator", help="Single indicator (sweep mode)")
    parser.add_argument("--long-only", action="store_true", help="Long only (sweep mode)")
    parser.add_argument("--best", action="store_true", help="Use best strategy per symbol from config")
    parser.add_argument("--sweep", action="store_true", help="Run all combinations (sweep mode)")
    args = parser.parse_args()

    if args.best or (not args.sweep and not args.symbols and not args.indicator):
        # Default: use best strategies from config
        run_best_strategies_backtest(args.period, args.interval)
    else:
        # Sweep mode: test all combinations
        symbols = args.symbols or SYMBOLS
        indicators = [args.indicator] if args.indicator else None
        directions = ["long"] if args.long_only else ["long", "short"]
        run_full_backtest(symbols, args.period, args.interval, indicators, directions)
