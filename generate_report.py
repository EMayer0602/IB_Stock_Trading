#!/usr/bin/env python3
"""
Generate detailed HTML reports with equity curves, trade lists and statistics.
Similar to Crypto2's report generation.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[Warning] plotly not installed. Run: pip install plotly")

try:
    import simple_yf as yf
except ImportError:
    import yfinance as yf

from tickers_config import TICKERS, SYMBOLS, get_ticker_config
import supertrend_strategy as st

NY_TZ = ZoneInfo("America/New_York")
REPORT_OUTPUT_DIR = "reports"
FEE_RATE = 0.001


def fetch_data(symbol: str, period: str = "1y", interval: str = "1h",
               start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """Fetch historical data."""
    try:
        if start_date and end_date:
            from simple_yf import fetch_historical_data
            df = fetch_historical_data(symbol, period, interval, start_date, end_date)
        else:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df.columns = df.columns.str.lower()
        df["atr"] = st.calculate_atr(df, st.ATR_WINDOW)
        return df
    except Exception as e:
        print(f"[Report] Error fetching {symbol}: {e}")
        return None


def run_backtest_detailed(symbol: str, df: pd.DataFrame, indicator: str,
                          direction: str, param_a: float, param_b: float,
                          initial_capital: float = 1000, atr_mult: float = 1.5,
                          min_hold_bars: int = 12) -> Dict:
    """Run backtest and return detailed results including equity curve."""

    signals = st.generate_indicator_signals(df, indicator, param_a, param_b)

    trades = []
    equity = initial_capital
    equity_curve = [{"time": df.index[0], "equity": equity}]
    position = None

    for i in range(1, len(df)):
        current_time = df.index[i]
        current_price = df["close"].iloc[i]
        current_signal = signals.iloc[i]
        prev_signal = signals.iloc[i - 1]
        current_atr = df["atr"].iloc[i] if "atr" in df.columns else 0

        # Exit logic
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
                            should_exit, exit_reason = True, "ATR stop"
                    else:
                        if current_price > position["entry_price"] + atr_mult * current_atr:
                            should_exit, exit_reason = True, "ATR stop"

            if should_exit:
                if direction == "long":
                    gross_pnl = (current_price - position["entry_price"]) * position["quantity"]
                else:
                    gross_pnl = (position["entry_price"] - current_price) * position["quantity"]

                fees = position["stake"] * FEE_RATE * 2
                net_pnl = gross_pnl - fees
                equity += net_pnl

                trades.append({
                    "entry_time": position["entry_time"],
                    "entry_price": position["entry_price"],
                    "exit_time": current_time,
                    "exit_price": current_price,
                    "quantity": position["quantity"],
                    "stake": position["stake"],
                    "gross_pnl": gross_pnl,
                    "fees": fees,
                    "net_pnl": net_pnl,
                    "bars_held": position["bars_held"],
                    "reason": exit_reason,
                    "equity_after": equity
                })
                position = None

        # Entry logic
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
                    position = {
                        "entry_time": current_time,
                        "entry_price": current_price,
                        "quantity": quantity,
                        "stake": stake,
                        "bars_held": 0
                    }

        equity_curve.append({"time": current_time, "equity": equity})

    # Close remaining position
    if position is not None:
        current_price = df["close"].iloc[-1]
        if direction == "long":
            gross_pnl = (current_price - position["entry_price"]) * position["quantity"]
        else:
            gross_pnl = (position["entry_price"] - current_price) * position["quantity"]
        fees = position["stake"] * FEE_RATE * 2
        net_pnl = gross_pnl - fees
        equity += net_pnl
        trades.append({
            "entry_time": position["entry_time"],
            "entry_price": position["entry_price"],
            "exit_time": df.index[-1],
            "exit_price": current_price,
            "quantity": position["quantity"],
            "stake": position["stake"],
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "bars_held": position["bars_held"],
            "reason": "End of period",
            "equity_after": equity
        })

    # Calculate statistics
    total_trades = len(trades)
    winning = sum(1 for t in trades if t["net_pnl"] > 0)
    losing = total_trades - winning
    win_rate = winning / total_trades if total_trades > 0 else 0
    total_pnl = sum(t["net_pnl"] for t in trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

    equity_df = pd.DataFrame(equity_curve)
    max_equity = equity_df["equity"].expanding().max()
    drawdown = (equity_df["equity"] - max_equity) / max_equity
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

    return {
        "symbol": symbol,
        "indicator": indicator,
        "direction": direction,
        "param_a": param_a,
        "param_b": param_b,
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_dd * 100,
        "start_capital": initial_capital,
        "end_capital": equity,
        "return_pct": (equity - initial_capital) / initial_capital * 100,
        "trades": trades,
        "equity_curve": equity_df
    }


def create_equity_chart(result: Dict, price_df: pd.DataFrame) -> str:
    """Create equity curve chart with price overlay."""
    if not PLOTLY_AVAILABLE:
        return "<p>Plotly not installed - no chart available</p>"

    equity_df = result["equity_curve"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.4],
                        subplot_titles=(f"{result['symbol']} Price", "Equity Curve"))

    # Price chart
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df["open"],
        high=price_df["high"],
        low=price_df["low"],
        close=price_df["close"],
        name="Price"
    ), row=1, col=1)

    # Mark trades
    for trade in result["trades"]:
        color = "green" if trade["net_pnl"] > 0 else "red"
        fig.add_trace(go.Scatter(
            x=[trade["entry_time"], trade["exit_time"]],
            y=[trade["entry_price"], trade["exit_price"]],
            mode="markers+lines",
            marker=dict(size=8, color=color),
            line=dict(color=color, width=1),
            showlegend=False
        ), row=1, col=1)

    # Equity curve
    fig.add_trace(go.Scatter(
        x=equity_df["time"],
        y=equity_df["equity"],
        mode="lines",
        name="Equity",
        line=dict(color="blue", width=2)
    ), row=2, col=1)

    # Add starting capital line
    fig.add_hline(y=result["start_capital"], line_dash="dash",
                  line_color="gray", row=2, col=1)

    fig.update_layout(
        height=800,
        title=f"{result['symbol']} - {result['indicator'].upper()} {result['direction'].upper()}",
        xaxis_rangeslider_visible=False,
        showlegend=False
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_trade_table(trades: List[Dict]) -> str:
    """Create HTML table of trades."""
    if not trades:
        return "<p>No trades</p>"

    html = """
    <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #f0f0f0;">
        <th>Entry Time</th>
        <th>Entry Price</th>
        <th>Exit Time</th>
        <th>Exit Price</th>
        <th>Qty</th>
        <th>P&L</th>
        <th>Bars</th>
        <th>Reason</th>
    </tr>
    """

    for t in trades:
        pnl_color = "green" if t["net_pnl"] > 0 else "red"
        entry_time = str(t["entry_time"])[:19] if t["entry_time"] else ""
        exit_time = str(t["exit_time"])[:19] if t["exit_time"] else ""

        html += f"""
        <tr>
            <td>{entry_time}</td>
            <td>${t['entry_price']:.2f}</td>
            <td>{exit_time}</td>
            <td>${t['exit_price']:.2f}</td>
            <td>{t['quantity']}</td>
            <td style="color: {pnl_color};">${t['net_pnl']:+.2f}</td>
            <td>{t['bars_held']}</td>
            <td>{t['reason']}</td>
        </tr>
        """

    html += "</table>"
    return html


def create_stats_table(result: Dict) -> str:
    """Create statistics summary table."""
    return f"""
    <table border="1" style="border-collapse: collapse; margin: 10px 0;">
    <tr><td><b>Total Trades</b></td><td>{result['total_trades']}</td></tr>
    <tr><td><b>Winning</b></td><td>{result['winning_trades']}</td></tr>
    <tr><td><b>Losing</b></td><td>{result['losing_trades']}</td></tr>
    <tr><td><b>Win Rate</b></td><td>{result['win_rate']*100:.1f}%</td></tr>
    <tr><td><b>Total P&L</b></td><td style="color: {'green' if result['total_pnl'] > 0 else 'red'};">${result['total_pnl']:+.2f}</td></tr>
    <tr><td><b>Return</b></td><td style="color: {'green' if result['return_pct'] > 0 else 'red'};">{result['return_pct']:+.1f}%</td></tr>
    <tr><td><b>Max Drawdown</b></td><td>{result['max_drawdown']:.1f}%</td></tr>
    <tr><td><b>Start Capital</b></td><td>${result['start_capital']:,.2f}</td></tr>
    <tr><td><b>End Capital</b></td><td>${result['end_capital']:,.2f}</td></tr>
    </table>
    """


def generate_symbol_report(symbol: str, df: pd.DataFrame, indicator: str,
                           param_a: float, param_b: float, capital: float) -> str:
    """Generate full report for one symbol."""

    html_parts = []

    for direction in ["long", "short"]:
        result = run_backtest_detailed(symbol, df, indicator, direction,
                                        param_a, param_b, capital)

        html_parts.append(f"<h2>{symbol} - {indicator.upper()} - {direction.upper()}</h2>")
        html_parts.append(create_stats_table(result))
        html_parts.append(create_equity_chart(result, df))
        html_parts.append("<h3>Trade List</h3>")
        html_parts.append(create_trade_table(result["trades"]))
        html_parts.append("<hr>")

    return "\n".join(html_parts)


def main():
    parser = argparse.ArgumentParser(description="Generate detailed trading reports")
    parser.add_argument("--period", default="1y", help="Data period")
    parser.add_argument("--interval", default="1h", help="Interval: 1h, 1d")
    parser.add_argument("--symbols", nargs="+", help="Symbols to test")
    parser.add_argument("--indicator", default="kama", help="Indicator")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    indicator = args.indicator

    # Get default parameters for indicator
    preset = st.INDICATOR_PRESETS.get(indicator, {})
    param_a = preset.get("default_a", 14)
    param_b = preset.get("default_b", 30)

    print(f"\n{'='*70}")
    print(f"GENERATING DETAILED REPORTS")
    print(f"Indicator: {indicator} | Symbols: {len(symbols)}")
    print(f"{'='*70}\n")

    Path(REPORT_OUTPUT_DIR).mkdir(exist_ok=True)

    all_html = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>IB Stock Trading Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "table { margin: 10px 0; }",
        "th, td { padding: 5px 10px; text-align: left; }",
        "h1 { color: #333; }",
        "h2 { color: #666; border-bottom: 1px solid #ccc; }",
        "hr { margin: 30px 0; }",
        "</style>",
        "</head><body>",
        f"<h1>IB Stock Trading Report - {indicator.upper()}</h1>",
        f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
    ]

    summary_data = []

    for symbol in symbols:
        config = get_ticker_config(symbol)
        capital = config.get("initial_capital_long", 1000)

        print(f"[{symbol}] Fetching data...")
        df = fetch_data(symbol, args.period, args.interval, args.start, args.end)

        if df is None or len(df) < 50:
            print(f"[{symbol}] Skipped - insufficient data")
            continue

        print(f"[{symbol}] Generating report ({len(df)} bars)...")

        # Generate report for this symbol
        symbol_html = generate_symbol_report(symbol, df, indicator, param_a, param_b, capital)
        all_html.append(symbol_html)

        # Collect summary
        for direction in ["long", "short"]:
            result = run_backtest_detailed(symbol, df, indicator, direction,
                                           param_a, param_b, capital)
            summary_data.append({
                "symbol": symbol,
                "direction": direction,
                "trades": result["total_trades"],
                "win_rate": result["win_rate"] * 100,
                "pnl": result["total_pnl"],
                "return_pct": result["return_pct"],
                "max_dd": result["max_drawdown"]
            })

    # Add summary table
    all_html.append("<h1>Summary</h1>")
    all_html.append("<table border='1' style='border-collapse: collapse;'>")
    all_html.append("<tr style='background-color: #f0f0f0;'>")
    all_html.append("<th>Symbol</th><th>Direction</th><th>Trades</th><th>Win Rate</th><th>P&L</th><th>Return</th><th>Max DD</th>")
    all_html.append("</tr>")

    for s in summary_data:
        pnl_color = "green" if s["pnl"] > 0 else "red"
        all_html.append(f"""
        <tr>
            <td>{s['symbol']}</td>
            <td>{s['direction']}</td>
            <td>{s['trades']}</td>
            <td>{s['win_rate']:.1f}%</td>
            <td style="color: {pnl_color};">${s['pnl']:+.2f}</td>
            <td style="color: {pnl_color};">{s['return_pct']:+.1f}%</td>
            <td>{s['max_dd']:.1f}%</td>
        </tr>
        """)

    all_html.append("</table>")
    all_html.append("</body></html>")

    # Save report
    report_file = f"{REPORT_OUTPUT_DIR}/report_{indicator}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_html))

    print(f"\n{'='*70}")
    print(f"Report saved to: {report_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
