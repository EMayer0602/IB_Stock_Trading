#!/usr/bin/env python3
"""
Generate Long vs Short Trading Dashboard

Creates an interactive HTML dashboard comparing long and short trades:
- Separate equity curves for long and short
- P&L per trade charts
- Combined equity curve
- Drawdown comparison
- Detailed statistics

Usage: python generate_long_short_dashboard.py
Output: long_vs_short_dashboard.html
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)


def calc_stats(df, name):
    """Calculate trading statistics for a dataframe of trades."""
    if len(df) == 0:
        return {
            'name': name, 'trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
            'profit_factor': 0, 'max_dd': 0
        }

    wins = df[df['net_pnl'] > 0]
    losses = df[df['net_pnl'] <= 0]
    total_pnl = df['net_pnl'].sum()
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
    profit_factor = wins['net_pnl'].sum() / abs(losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else 0
    peak = df['equity'].cummax()
    max_dd = ((peak - df['equity']) / peak * 100).max() if len(df) > 0 else 0

    return {
        'name': name,
        'trades': len(df),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_dd': max_dd
    }


def generate_long_short_dashboard(output_file='long_vs_short_dashboard.html'):
    """Generate the long vs short comparison dashboard."""

    print("Generating Long vs Short Trading Dashboard...")

    # Check if backtest data exists
    trades_file = Path('backtest_trades.csv')
    if not trades_file.exists():
        print(f"ERROR: {trades_file} not found. Run backtest first.")
        sys.exit(1)

    # Load trade data
    trades_df = pd.read_csv(trades_file).sort_values('exit_time')
    print(f"Loaded {len(trades_df)} total trades")

    # Split by direction
    long_trades = trades_df[trades_df['direction'] == 'long'].copy()
    short_trades = trades_df[trades_df['direction'] == 'short'].copy()

    print(f"  Long trades: {len(long_trades)}")
    print(f"  Short trades: {len(short_trades)}")

    # Calculate cumulative P&L for each
    long_trades['cumulative_pnl'] = long_trades['net_pnl'].cumsum()
    long_trades['equity'] = 7000 + long_trades['cumulative_pnl']  # Half capital for longs

    short_trades['cumulative_pnl'] = short_trades['net_pnl'].cumsum()
    short_trades['equity'] = 7000 + short_trades['cumulative_pnl']  # Half capital for shorts

    # Combined
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
    trades_df['equity'] = 14000 + trades_df['cumulative_pnl']

    # Create figure with 6 rows
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        row_heights=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=(
            'LONG Trades - Equity Curve',
            'SHORT Trades - Equity Curve',
            'LONG Trades - P&L per Trade',
            'SHORT Trades - P&L per Trade',
            'Combined Equity Curve',
            'Drawdown Comparison'
        )
    )

    # ===== ROW 1: Long Equity Curve =====
    long_peak = long_trades['equity'].cummax()
    fig.add_trace(
        go.Scatter(x=list(range(len(long_trades))), y=long_trades['equity'],
                   mode='lines', name='Long Equity', fill='tozeroy',
                   line=dict(color='#4caf50', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(long_trades))), y=long_peak,
                   mode='lines', name='Long Peak',
                   line=dict(color='#81c784', width=1, dash='dot')),
        row=1, col=1
    )

    # ===== ROW 2: Short Equity Curve =====
    short_peak = short_trades['equity'].cummax()
    fig.add_trace(
        go.Scatter(x=list(range(len(short_trades))), y=short_trades['equity'],
                   mode='lines', name='Short Equity', fill='tozeroy',
                   line=dict(color='#f44336', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(short_trades))), y=short_peak,
                   mode='lines', name='Short Peak',
                   line=dict(color='#e57373', width=1, dash='dot')),
        row=2, col=1
    )

    # ===== ROW 3: Long P&L per Trade =====
    long_colors = ['#4caf50' if pnl > 0 else '#c62828' for pnl in long_trades['net_pnl']]
    fig.add_trace(
        go.Bar(x=list(range(len(long_trades))), y=long_trades['net_pnl'],
               name='Long P&L', marker_color=long_colors),
        row=3, col=1
    )

    # ===== ROW 4: Short P&L per Trade =====
    short_colors = ['#4caf50' if pnl > 0 else '#c62828' for pnl in short_trades['net_pnl']]
    fig.add_trace(
        go.Bar(x=list(range(len(short_trades))), y=short_trades['net_pnl'],
               name='Short P&L', marker_color=short_colors),
        row=4, col=1
    )

    # ===== ROW 5: Combined Equity =====
    combined_peak = trades_df['equity'].cummax()
    fig.add_trace(
        go.Scatter(x=list(range(len(trades_df))), y=trades_df['equity'],
                   mode='lines', name='Combined Equity',
                   line=dict(color='#2196f3', width=2)),
        row=5, col=1
    )

    # ===== ROW 6: Drawdown Comparison =====
    long_dd = (long_peak - long_trades['equity']) / long_peak * 100
    short_dd = (short_peak - short_trades['equity']) / short_peak * 100

    fig.add_trace(
        go.Scatter(x=list(range(len(long_trades))), y=-long_dd,
                   mode='lines', name='Long DD',
                   line=dict(color='#4caf50', width=1)),
        row=6, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(short_trades))), y=-short_dd,
                   mode='lines', name='Short DD',
                   line=dict(color='#f44336', width=1)),
        row=6, col=1
    )

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Long vs Short Trading Analysis</b>',
            x=0.5, font=dict(size=20)
        ),
        template='plotly_dark',
        height=1400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5)
    )

    # Axis labels
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=4, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=5, col=1)
    fig.update_yaxes(title_text="DD %", row=6, col=1)
    fig.update_xaxes(title_text="Trade #", row=6, col=1)

    # Calculate stats
    long_stats = calc_stats(long_trades, 'LONG')
    short_stats = calc_stats(short_trades, 'SHORT')

    # Add stats annotations
    long_text = (
        f"<b>LONG TRADES</b><br>"
        f"Trades: {long_stats['trades']} ({long_stats['wins']}W / {long_stats['losses']}L)<br>"
        f"Win Rate: {long_stats['win_rate']:.1f}%<br>"
        f"Total P&L: ${long_stats['total_pnl']:,.2f}<br>"
        f"Avg Win: ${long_stats['avg_win']:,.2f}<br>"
        f"Avg Loss: ${long_stats['avg_loss']:,.2f}<br>"
        f"Profit Factor: {long_stats['profit_factor']:.2f}<br>"
        f"Max Drawdown: {long_stats['max_dd']:.1f}%"
    )

    short_text = (
        f"<b>SHORT TRADES</b><br>"
        f"Trades: {short_stats['trades']} ({short_stats['wins']}W / {short_stats['losses']}L)<br>"
        f"Win Rate: {short_stats['win_rate']:.1f}%<br>"
        f"Total P&L: ${short_stats['total_pnl']:,.2f}<br>"
        f"Avg Win: ${short_stats['avg_win']:,.2f}<br>"
        f"Avg Loss: ${short_stats['avg_loss']:,.2f}<br>"
        f"Profit Factor: {short_stats['profit_factor']:.2f}<br>"
        f"Max Drawdown: {short_stats['max_dd']:.1f}%"
    )

    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
        text=long_text, showarrow=False,
        font=dict(size=10, color="white"), align="left",
        bgcolor="rgba(76,175,80,0.8)", borderpad=6)

    fig.add_annotation(x=0.20, y=0.98, xref="paper", yref="paper",
        text=short_text, showarrow=False,
        font=dict(size=10, color="white"), align="left",
        bgcolor="rgba(244,67,54,0.8)", borderpad=6)

    # Save
    fig.write_html(output_file)
    print(f"\n✅ Dashboard saved to: {output_file}")

    # Print text summary
    print("\n" + "=" * 70)
    print("LONG TRADES STATISTICS")
    print("=" * 70)
    print(f"Total Trades: {long_stats['trades']}")
    print(f"Winners: {long_stats['wins']} | Losers: {long_stats['losses']}")
    print(f"Win Rate: {long_stats['win_rate']:.1f}%")
    print(f"Total P&L: ${long_stats['total_pnl']:,.2f}")
    print(f"Avg Win: ${long_stats['avg_win']:,.2f} | Avg Loss: ${long_stats['avg_loss']:,.2f}")
    print(f"Profit Factor: {long_stats['profit_factor']:.2f}")
    print(f"Max Drawdown: {long_stats['max_dd']:.1f}%")

    print("\n" + "=" * 70)
    print("SHORT TRADES STATISTICS")
    print("=" * 70)
    print(f"Total Trades: {short_stats['trades']}")
    print(f"Winners: {short_stats['wins']} | Losers: {short_stats['losses']}")
    print(f"Win Rate: {short_stats['win_rate']:.1f}%")
    print(f"Total P&L: ${short_stats['total_pnl']:,.2f}")
    print(f"Avg Win: ${short_stats['avg_win']:,.2f} | Avg Loss: ${short_stats['avg_loss']:,.2f}")
    print(f"Profit Factor: {short_stats['profit_factor']:.2f}")
    print(f"Max Drawdown: {short_stats['max_dd']:.1f}%")

    # Top trades
    print("\n" + "=" * 70)
    print("TOP 5 LONG TRADES")
    print("=" * 70)
    top_long = long_trades.nlargest(5, 'net_pnl')[['symbol', 'indicator', 'net_pnl', 'bars_held']]
    for _, t in top_long.iterrows():
        print(f"  {t['symbol']:<6} {t['indicator']:<12} ${t['net_pnl']:>8.2f}  ({t['bars_held']} bars)")

    print("\n" + "=" * 70)
    print("TOP 5 SHORT TRADES")
    print("=" * 70)
    top_short = short_trades.nlargest(5, 'net_pnl')[['symbol', 'indicator', 'net_pnl', 'bars_held']]
    for _, t in top_short.iterrows():
        print(f"  {t['symbol']:<6} {t['indicator']:<12} ${t['net_pnl']:>8.2f}  ({t['bars_held']} bars)")

    # By symbol
    print("\n" + "=" * 70)
    print("LONG P&L BY SYMBOL")
    print("=" * 70)
    long_by_sym = long_trades.groupby('symbol')['net_pnl'].agg(['count', 'sum']).round(2)
    long_by_sym.columns = ['Trades', 'Total P&L']
    long_by_sym = long_by_sym.sort_values('Total P&L', ascending=False)
    for sym, row in long_by_sym.iterrows():
        print(f"  {sym:<6} {int(row['Trades']):>4} trades  ${row['Total P&L']:>10.2f}")

    print("\n" + "=" * 70)
    print("SHORT P&L BY SYMBOL")
    print("=" * 70)
    short_by_sym = short_trades.groupby('symbol')['net_pnl'].agg(['count', 'sum']).round(2)
    short_by_sym.columns = ['Trades', 'Total P&L']
    short_by_sym = short_by_sym.sort_values('Total P&L', ascending=False)
    for sym, row in short_by_sym.iterrows():
        print(f"  {sym:<6} {int(row['Trades']):>4} trades  ${row['Total P&L']:>10.2f}")

    return output_file


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Long vs Short Dashboard')
    parser.add_argument('--output', default='long_vs_short_dashboard.html', help='Output file')
    args = parser.parse_args()

    generate_long_short_dashboard(args.output)
