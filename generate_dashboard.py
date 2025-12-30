#!/usr/bin/env python3
"""
Generate Trading Dashboard - Creates an interactive HTML chart with:
- Candlestick chart with indicators (Supertrend, KAMA)
- RSI panel
- Volume panel
- Equity curve
- Drawdown chart

Usage: python generate_dashboard.py
Output: trading_dashboard.html
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Check for plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

# Import local modules
from supertrend_strategy import calculate_atr, calculate_supertrend, calculate_kama, calculate_rsi


def resample_to_htf(df, htf_multiplier=6):
    """Resample 1h data to higher timeframe (e.g., 6h)."""
    htf_df = df.resample(f'{htf_multiplier}h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return htf_df


def generate_dashboard(symbol='AAPL', indicator='supertrend', output_file='trading_dashboard.html', htf_multiplier=6):
    """Generate the trading dashboard HTML file with HTF indicators."""

    print(f"Generating dashboard for {symbol} with {indicator}...")
    print(f"Using {htf_multiplier}x higher timeframe (HTF = {htf_multiplier}h)")

    # Check if backtest data exists
    trades_file = Path('backtest_trades.csv')
    if not trades_file.exists():
        print(f"ERROR: {trades_file} not found. Run backtest first.")
        sys.exit(1)

    # Load trade data
    trades_df = pd.read_csv(trades_file)
    symbol_trades = trades_df[(trades_df['symbol'] == symbol) & (trades_df['indicator'] == indicator)]
    print(f"Found {len(symbol_trades)} trades for {symbol}/{indicator}")

    # Try to fetch real data, fall back to sample
    df = None
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='6mo', interval='1h')
        if len(df) > 0:
            df.columns = [c.lower() for c in df.columns]
            print(f"Fetched {len(df)} bars from Yahoo Finance")
    except Exception as e:
        print(f"Yahoo Finance not available: {e}")

    if df is None or len(df) == 0:
        print("Using sample data...")
        dates = pd.date_range('2025-06-01', periods=500, freq='h')
        np.random.seed(42)
        close = 200 + np.cumsum(np.random.randn(500) * 2)
        df = pd.DataFrame({
            'open': close + np.random.randn(500),
            'high': close + abs(np.random.randn(500)) * 2,
            'low': close - abs(np.random.randn(500)) * 2,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 500)
        }, index=dates)

    # Calculate 1h indicators
    df['atr'] = calculate_atr(df, 14)
    df['supertrend'], df['st_direction'] = calculate_supertrend(df, length=10, factor=3.0)
    df['kama'] = calculate_kama(df['close'], fast_length=14, slow_length=30)
    df['rsi'] = calculate_rsi(df['close'], window=14)

    # Calculate HTF (6h) indicators
    print(f"Calculating {htf_multiplier}h HTF indicators...")
    htf_df = resample_to_htf(df, htf_multiplier)
    htf_df['atr'] = calculate_atr(htf_df, 14)
    htf_df['supertrend'], htf_df['st_direction'] = calculate_supertrend(htf_df, length=10, factor=3.0)
    htf_df['kama'] = calculate_kama(htf_df['close'], fast_length=14, slow_length=30)

    # Map HTF values back to 1h timeframe (forward fill)
    df['htf_supertrend'] = htf_df['supertrend'].reindex(df.index, method='ffill')
    df['htf_direction'] = htf_df['st_direction'].reindex(df.index, method='ffill')
    df['htf_kama'] = htf_df['kama'].reindex(df.index, method='ffill')

    # Load equity data
    trades_all = pd.read_csv(trades_file).sort_values('exit_time')
    trades_all['cumulative_pnl'] = trades_all['net_pnl'].cumsum()
    trades_all['equity'] = 14000 + trades_all['cumulative_pnl']
    peak = trades_all['equity'].cummax()
    drawdown_pct = (peak - trades_all['equity']) / peak * 100

    # Create combined figure
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.10, 0.10, 0.25, 0.10],
        subplot_titles=(
            f'{symbol} Price Chart (1h + {htf_multiplier}h HTF Indicators)',
            'RSI (1h)',
            'Volume',
            'Portfolio Equity Curve',
            'Drawdown %'
        )
    )

    # ROW 1: Candlestick + Indicators
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Price',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Supertrend (colored segments)
    for i in range(1, len(df)):
        color = '#26a69a' if df['st_direction'].iloc[i] == 1 else '#ef5350'
        fig.add_trace(
            go.Scatter(
                x=[df.index[i-1], df.index[i]],
                y=[df['supertrend'].iloc[i-1], df['supertrend'].iloc[i]],
                mode='lines', line=dict(color=color, width=2),
                showlegend=False, hoverinfo='skip'
            ),
            row=1, col=1
        )

    # KAMA (1h)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['kama'], mode='lines', name='KAMA (1h)',
                   line=dict(color='#ff9800', width=1.5)),
        row=1, col=1
    )

    # HTF Supertrend (6h) - thicker dashed line
    for i in range(1, len(df)):
        if pd.notna(df['htf_direction'].iloc[i]):
            color = '#00bcd4' if df['htf_direction'].iloc[i] == 1 else '#e91e63'
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i-1], df.index[i]],
                    y=[df['htf_supertrend'].iloc[i-1], df['htf_supertrend'].iloc[i]],
                    mode='lines', line=dict(color=color, width=3, dash='dash'),
                    showlegend=False, hoverinfo='skip'
                ),
                row=1, col=1
            )

    # HTF KAMA (6h) - thicker line
    fig.add_trace(
        go.Scatter(x=df.index, y=df['htf_kama'], mode='lines',
                   name=f'KAMA ({htf_multiplier}h HTF)',
                   line=dict(color='#9c27b0', width=2.5, dash='dot')),
        row=1, col=1
    )

    # Trade markers
    long_entries = symbol_trades[symbol_trades['direction'] == 'long']
    short_entries = symbol_trades[symbol_trades['direction'] == 'short']

    if len(long_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(long_entries['entry_time'], utc=True),
                y=long_entries['entry_price'],
                mode='markers', name='Long Entry',
                marker=dict(symbol='triangle-up', size=12, color='#00e676',
                           line=dict(color='white', width=1)),
                text=[f"Long Entry ${p:.2f}" for p in long_entries['entry_price']],
                hoverinfo='text'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(long_entries['exit_time'], utc=True),
                y=long_entries['exit_price'],
                mode='markers', name='Long Exit',
                marker=dict(symbol='triangle-down', size=12, color='#69f0ae',
                           line=dict(color='white', width=1)),
                text=[f"Long Exit ${p:.2f}<br>P&L: ${pnl:.2f}" for p, pnl in
                      zip(long_entries['exit_price'], long_entries['net_pnl'])],
                hoverinfo='text'
            ),
            row=1, col=1
        )

    if len(short_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(short_entries['entry_time'], utc=True),
                y=short_entries['entry_price'],
                mode='markers', name='Short Entry',
                marker=dict(symbol='triangle-down', size=12, color='#ff5252',
                           line=dict(color='white', width=1)),
                text=[f"Short Entry ${p:.2f}" for p in short_entries['entry_price']],
                hoverinfo='text'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(short_entries['exit_time'], utc=True),
                y=short_entries['exit_price'],
                mode='markers', name='Short Exit',
                marker=dict(symbol='triangle-up', size=12, color='#ff8a80',
                           line=dict(color='white', width=1)),
                text=[f"Short Exit ${p:.2f}<br>P&L: ${pnl:.2f}" for p, pnl in
                      zip(short_entries['exit_price'], short_entries['net_pnl'])],
                hoverinfo='text'
            ),
            row=1, col=1
        )

    # ROW 2: RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['rsi'], mode='lines', name='RSI',
                   line=dict(color='#ab47bc', width=1.5)),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # ROW 3: Volume
    colors_vol = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350'
                  for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors_vol, opacity=0.7),
        row=3, col=1
    )

    # ROW 4: Equity Curve
    fig.add_trace(
        go.Scatter(x=list(range(len(trades_all))), y=trades_all['equity'],
                   mode='lines', name='Equity', fill='tozeroy',
                   line=dict(color='#42a5f5', width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(trades_all))), y=peak,
                   mode='lines', name='Peak Equity',
                   line=dict(color='#66bb6a', width=1, dash='dot')),
        row=4, col=1
    )

    # ROW 5: Drawdown
    fig.add_trace(
        go.Scatter(x=list(range(len(trades_all))), y=-drawdown_pct,
                   mode='lines', name='Drawdown', fill='tozeroy',
                   line=dict(color='#ef5350', width=1)),
        row=5, col=1
    )

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>IB Stock Trading Dashboard</b><br>'
                 '<sub>Price Chart + Indicators + Trade Markers + Equity Curve</sub>',
            x=0.5, font=dict(size=20)
        ),
        template='plotly_dark',
        height=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    # Axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=4, col=1)
    fig.update_yaxes(title_text="DD %", row=5, col=1)
    fig.update_xaxes(title_text="Trade #", row=5, col=1)

    # Stats annotation
    total_pnl = trades_all['cumulative_pnl'].iloc[-1]
    final_equity = trades_all['equity'].iloc[-1]
    max_dd = drawdown_pct.max()
    win_rate = len(trades_all[trades_all['net_pnl'] > 0]) / len(trades_all) * 100

    stats_text = (
        f"<b>Summary Stats</b><br>"
        f"HTF: {htf_multiplier}h (6x base)<br>"
        f"Total Trades: {len(trades_all):,}<br>"
        f"Win Rate: {win_rate:.1f}%<br>"
        f"Total P&L: ${total_pnl:,.2f}<br>"
        f"Final Equity: ${final_equity:,.2f}<br>"
        f"Max Drawdown: {max_dd:.1f}%"
    )

    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=stats_text, showarrow=False,
        font=dict(size=11, color="white"), align="left",
        bgcolor="rgba(0,0,0,0.7)", bordercolor="gray",
        borderwidth=1, borderpad=8
    )

    # Save
    fig.write_html(output_file)
    print(f"\n✅ Dashboard saved to: {output_file}")
    return output_file


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Trading Dashboard')
    parser.add_argument('--symbol', default='AAPL', help='Symbol to display (default: AAPL)')
    parser.add_argument('--indicator', default='supertrend', help='Indicator (default: supertrend)')
    parser.add_argument('--output', default='trading_dashboard.html', help='Output file')
    parser.add_argument('--htf', type=int, default=6, help='HTF multiplier (default: 6 = 6h for 1h base)')
    args = parser.parse_args()

    generate_dashboard(args.symbol, args.indicator, args.output, args.htf)
