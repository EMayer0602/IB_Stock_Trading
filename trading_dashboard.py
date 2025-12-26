"""Live Trading Dashboard for Monitoring.

This module provides a web-based monitoring dashboard:
- Real-time status overview
- P&L tracking and visualization
- Open positions display
- Risk metrics monitoring
- Trade history with full details
- Equity curve chart
- Trade detail modal
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
import html

# Try importing optional dependencies
try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    port: int = 8080
    host: str = "localhost"
    refresh_interval_seconds: int = 30
    output_dir: str = "dashboard"
    state_file: str = "paper_trading_state.json"
    risk_state_file: str = "risk_state.json"
    trade_log_file: str = "paper_trading_simulation_log.csv"
    order_log_file: str = "order_log.json"
    initial_capital: float = 16000.0


class TradingDashboard:
    """Trading dashboard generator and server."""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self._running = False
        self._server: Optional[HTTPServer] = None
        self._update_thread: Optional[threading.Thread] = None

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _load_json_file(self, filepath: str) -> Dict:
        """Load a JSON file safely."""
        path = Path(filepath)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _load_csv_file(self, filepath: str) -> List[Dict]:
        """Load a CSV file as list of dicts."""
        if pd is None:
            return []
        path = Path(filepath)
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df.to_dict("records")
            except Exception:
                pass
        return []

    def _get_trading_state(self) -> Dict:
        """Get current trading state."""
        return self._load_json_file(self.config.state_file)

    def _get_risk_state(self) -> Dict:
        """Get current risk state."""
        return self._load_json_file(self.config.risk_state_file)

    def _get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history."""
        trades = self._load_csv_file(self.config.trade_log_file)
        return trades[-limit:] if trades else []

    def _get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get recent order history."""
        orders = self._load_json_file(self.config.order_log_file)
        if isinstance(orders, list):
            return orders[-limit:]
        return []

    def _calculate_metrics(self, state: Dict, risk_state: Dict, trades: List[Dict]) -> Dict:
        """Calculate dashboard metrics."""
        # Basic metrics
        total_capital = float(state.get("total_capital", self.config.initial_capital))
        initial_capital = self.config.initial_capital

        total_pnl = total_capital - initial_capital
        pnl_pct = (total_pnl / initial_capital * 100) if initial_capital else 0

        # Drawdown
        peak_equity = float(risk_state.get("peak_equity", total_capital))
        if peak_equity < total_capital:
            peak_equity = total_capital
        drawdown = peak_equity - total_capital
        drawdown_pct = (drawdown / peak_equity * 100) if peak_equity else 0

        # Position metrics
        positions = state.get("positions", [])
        open_count = len(positions)
        total_exposure = sum(float(p.get("stake", 0)) for p in positions)
        exposure_pct = (total_exposure / total_capital * 100) if total_capital else 0

        # Trade statistics
        if trades:
            wins = sum(1 for t in trades if float(t.get("pnl", t.get("PnL", 0)) or 0) > 0)
            losses = sum(1 for t in trades if float(t.get("pnl", t.get("PnL", 0)) or 0) < 0)
            total_trades = len(trades)
            win_rate = (wins / total_trades * 100) if total_trades else 0

            winning_pnl = sum(float(t.get("pnl", t.get("PnL", 0)) or 0) for t in trades if float(t.get("pnl", t.get("PnL", 0)) or 0) > 0)
            losing_pnl = abs(sum(float(t.get("pnl", t.get("PnL", 0)) or 0) for t in trades if float(t.get("pnl", t.get("PnL", 0)) or 0) < 0))
            profit_factor = (winning_pnl / losing_pnl) if losing_pnl else float('inf')

            avg_win = winning_pnl / wins if wins else 0
            avg_loss = losing_pnl / losses if losses else 0
        else:
            wins = losses = total_trades = 0
            win_rate = profit_factor = avg_win = avg_loss = 0

        # Risk state metrics
        daily_pnl = float(risk_state.get("daily_pnl", 0))
        weekly_pnl = float(risk_state.get("weekly_pnl", 0))
        consecutive_losses = int(risk_state.get("consecutive_losses", 0))
        status = risk_state.get("status", "active")

        return {
            "total_capital": total_capital,
            "initial_capital": initial_capital,
            "total_pnl": total_pnl,
            "total_pnl_pct": pnl_pct,
            "peak_equity": peak_equity,
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct,
            "open_positions": open_count,
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "daily_pnl": daily_pnl,
            "weekly_pnl": weekly_pnl,
            "consecutive_losses": consecutive_losses,
            "status": status,
        }

    def _generate_equity_curve_data(self, trades: List[Dict]) -> List[Dict]:
        """Generate equity curve data from trades."""
        equity_data = []
        current_equity = self.config.initial_capital
        equity_data.append({"time": "Start", "equity": current_equity})

        for i, t in enumerate(trades):
            pnl = float(t.get("pnl", t.get("PnL", 0)) or 0)
            current_equity += pnl
            exit_time = t.get("exit_time", t.get("ExitTime", t.get("ExitZeit", f"Trade {i+1}")))
            if isinstance(exit_time, str) and len(exit_time) > 10:
                exit_time = exit_time[:10]
            equity_data.append({"time": str(exit_time), "equity": current_equity})

        return equity_data

    def generate_html(self) -> str:
        """Generate the dashboard HTML."""
        state = self._get_trading_state()
        risk_state = self._get_risk_state()
        trades = self._get_trade_history(100)
        orders = self._get_order_history(20)
        metrics = self._calculate_metrics(state, risk_state, trades)
        positions = state.get("positions", [])
        equity_data = self._generate_equity_curve_data(trades)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Status color
        status = metrics["status"]
        status_color = {
            "active": "#28a745",
            "paused": "#ffc107",
            "halted": "#dc3545",
            "disabled": "#6c757d",
        }.get(status, "#28a745")

        # Convert equity data to JSON for chart
        equity_json = json.dumps(equity_data)
        trades_json = json.dumps(trades)

        html_content = f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="{self.config.refresh_interval_seconds}">
    <title>Trading Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }}
        h1 {{ color: #fff; font-size: 24px; }}
        .status-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            background: {status_color};
            color: white;
        }}
        .last-update {{ color: #888; font-size: 14px; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .card-title {{
            color: #888;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: bold;
            color: #fff;
        }}
        .card-value.positive {{ color: #28a745; }}
        .card-value.negative {{ color: #dc3545; }}
        .card-subtitle {{
            color: #888;
            font-size: 12px;
            margin-top: 5px;
        }}
        .section-title {{
            font-size: 18px;
            margin-bottom: 15px;
            color: #fff;
            border-left: 4px solid #0d6efd;
            padding-left: 12px;
        }}
        .card-wide {{
            grid-column: span 2;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 8px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            color: #888;
            font-weight: 500;
            text-transform: uppercase;
            font-size: 11px;
            position: sticky;
            top: 0;
            background: #16213e;
        }}
        tr:hover {{ background: #1f2b47; cursor: pointer; }}
        .pnl-positive {{ color: #28a745; }}
        .pnl-negative {{ color: #dc3545; }}
        .risk-meter {{
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .risk-meter-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .risk-low {{ background: linear-gradient(90deg, #28a745, #28a745); }}
        .risk-medium {{ background: linear-gradient(90deg, #28a745, #ffc107); }}
        .risk-high {{ background: linear-gradient(90deg, #ffc107, #dc3545); }}
        .risk-critical {{ background: linear-gradient(90deg, #dc3545, #dc3545); }}
        .chart-container {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            height: 300px;
        }}
        .trade-table-container {{
            max-height: 400px;
            overflow-y: auto;
        }}
        /* Modal Styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
        }}
        .modal-content {{
            background: #16213e;
            margin: 10% auto;
            padding: 30px;
            border-radius: 12px;
            width: 90%;
            max-width: 600px;
            position: relative;
        }}
        .modal-close {{
            position: absolute;
            right: 20px;
            top: 15px;
            font-size: 28px;
            cursor: pointer;
            color: #888;
        }}
        .modal-close:hover {{ color: #fff; }}
        .detail-row {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #333;
        }}
        .detail-label {{ color: #888; }}
        .detail-value {{ font-weight: bold; }}
        .flex-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .flex-row > div {{ flex: 1; min-width: 400px; }}
        @media (max-width: 768px) {{
            .grid {{ grid-template-columns: 1fr 1fr; }}
            .card-value {{ font-size: 20px; }}
            .flex-row > div {{ min-width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>📊 Trading Dashboard</h1>
                <div class="last-update">Letzte Aktualisierung: {now}</div>
            </div>
            <div class="status-badge">{status}</div>
        </header>

        <!-- Key Metrics -->
        <div class="grid">
            <div class="card">
                <div class="card-title">Gesamtkapital</div>
                <div class="card-value">{metrics["total_capital"]:,.2f} €</div>
                <div class="card-subtitle">Start: {metrics["initial_capital"]:,.2f} €</div>
            </div>
            <div class="card">
                <div class="card-title">Gesamt P&L</div>
                <div class="card-value {"positive" if metrics["total_pnl"] >= 0 else "negative"}">
                    {metrics["total_pnl"]:+,.2f} € ({metrics["total_pnl_pct"]:+.2f}%)
                </div>
            </div>
            <div class="card">
                <div class="card-title">Tages P&L</div>
                <div class="card-value {"positive" if metrics["daily_pnl"] >= 0 else "negative"}">
                    {metrics["daily_pnl"]:+,.2f} €
                </div>
                <div class="card-subtitle">Woche: {metrics["weekly_pnl"]:+,.2f} €</div>
            </div>
            <div class="card">
                <div class="card-title">Drawdown</div>
                <div class="card-value {"positive" if metrics["drawdown"] == 0 else "negative"}">
                    {metrics["drawdown"]:,.2f} € ({metrics["drawdown_pct"]:.1f}%)
                </div>
                <div class="risk-meter">
                    <div class="risk-meter-fill {"risk-low" if metrics["drawdown_pct"] < 5 else "risk-medium" if metrics["drawdown_pct"] < 10 else "risk-high" if metrics["drawdown_pct"] < 20 else "risk-critical"}"
                         style="width: {min(metrics["drawdown_pct"] * 5, 100)}%"></div>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Offene Positionen</div>
                <div class="card-value">{metrics["open_positions"]}</div>
                <div class="card-subtitle">Exposure: {metrics["total_exposure"]:,.0f} € ({metrics["exposure_pct"]:.0f}%)</div>
            </div>
            <div class="card">
                <div class="card-title">Trades</div>
                <div class="card-value">{metrics["total_trades"]}</div>
                <div class="card-subtitle">{metrics["wins"]}W / {metrics["losses"]}L</div>
            </div>
            <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value">{metrics["win_rate"]:.1f}%</div>
                <div class="card-subtitle">PF: {metrics["profit_factor"]:.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">Ø Trade</div>
                <div class="card-value positive">+{metrics["avg_win"]:.0f}€</div>
                <div class="card-subtitle negative">-{metrics["avg_loss"]:.0f}€</div>
            </div>
        </div>

        <!-- Equity Chart -->
        <div class="chart-container">
            <h3 class="section-title">Equity-Kurve</h3>
            <canvas id="equityChart" style="width: 100%; height: 220px;"></canvas>
        </div>

        <div class="flex-row">
            <!-- Open Positions -->
            <div class="card">
                <h3 class="section-title">Offene Positionen</h3>
                {self._generate_positions_table(positions)}
            </div>
        </div>

        <!-- Trade History (Full Details) -->
        <div class="card" style="margin-top: 20px;">
            <h3 class="section-title">Trade Historie (Klick für Details)</h3>
            <div class="trade-table-container">
                {self._generate_trades_table(trades)}
            </div>
        </div>

        <!-- Orders -->
        <div class="card" style="margin-top: 20px;">
            <h3 class="section-title">Letzte Orders</h3>
            {self._generate_orders_table(orders)}
        </div>
    </div>

    <!-- Trade Detail Modal -->
    <div id="tradeModal" class="modal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModal()">&times;</span>
            <h2 style="margin-bottom: 20px;">Trade Details</h2>
            <div id="modalContent"></div>
        </div>
    </div>

    <script>
        // Trade data
        const trades = {trades_json};
        const equityData = {equity_json};

        // Draw equity chart
        function drawEquityChart() {{
            const canvas = document.getElementById('equityChart');
            if (!canvas || equityData.length < 2) return;

            const ctx = canvas.getContext('2d');
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width - 40;
            canvas.height = 220;

            const padding = {{ top: 20, right: 20, bottom: 30, left: 60 }};
            const width = canvas.width - padding.left - padding.right;
            const height = canvas.height - padding.top - padding.bottom;

            // Find min/max
            const equities = equityData.map(d => d.equity);
            const minEquity = Math.min(...equities) * 0.99;
            const maxEquity = Math.max(...equities) * 1.01;

            // Clear
            ctx.fillStyle = '#16213e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw grid
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {{
                const y = padding.top + (height * i / 4);
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(padding.left + width, y);
                ctx.stroke();

                const value = maxEquity - (maxEquity - minEquity) * i / 4;
                ctx.fillStyle = '#888';
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(value.toFixed(0) + '€', padding.left - 5, y + 4);
            }}

            // Draw line
            ctx.beginPath();
            ctx.strokeStyle = '#0d6efd';
            ctx.lineWidth = 2;

            equityData.forEach((d, i) => {{
                const x = padding.left + (width * i / (equityData.length - 1));
                const y = padding.top + height - (height * (d.equity - minEquity) / (maxEquity - minEquity));

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});
            ctx.stroke();

            // Draw points
            equityData.forEach((d, i) => {{
                const x = padding.left + (width * i / (equityData.length - 1));
                const y = padding.top + height - (height * (d.equity - minEquity) / (maxEquity - minEquity));

                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fillStyle = d.equity >= {self.config.initial_capital} ? '#28a745' : '#dc3545';
                ctx.fill();
            }});

            // Start line
            const startY = padding.top + height - (height * ({self.config.initial_capital} - minEquity) / (maxEquity - minEquity));
            ctx.beginPath();
            ctx.strokeStyle = '#ffc107';
            ctx.setLineDash([5, 5]);
            ctx.moveTo(padding.left, startY);
            ctx.lineTo(padding.left + width, startY);
            ctx.stroke();
            ctx.setLineDash([]);
        }}

        // Show trade detail modal
        function showTradeDetail(index) {{
            const trade = trades[index];
            if (!trade) return;

            const pnl = parseFloat(trade.pnl || trade.PnL || 0);
            const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

            const content = `
                <div class="detail-row">
                    <span class="detail-label">Symbol</span>
                    <span class="detail-value">${{trade.symbol || trade.Symbol || '?'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Richtung</span>
                    <span class="detail-value">${{(trade.direction || trade.Direction || '?').toUpperCase()}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Indikator</span>
                    <span class="detail-value">${{trade.indicator || trade.Indicator || '?'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">HTF</span>
                    <span class="detail-value">${{trade.htf || trade.HTF || '?'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Parameter</span>
                    <span class="detail-value">${{trade.param_desc || trade.ParamDesc || '?'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Entry Preis</span>
                    <span class="detail-value">${{parseFloat(trade.entry_price || trade.EntryPrice || 0).toFixed(6)}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Exit Preis</span>
                    <span class="detail-value">${{parseFloat(trade.exit_price || trade.ExitPrice || 0).toFixed(6)}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Entry Zeit</span>
                    <span class="detail-value">${{trade.entry_time || trade.EntryTime || '?'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Exit Zeit</span>
                    <span class="detail-value">${{trade.exit_time || trade.ExitTime || '?'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Stake</span>
                    <span class="detail-value">${{parseFloat(trade.stake || trade.Stake || 0).toFixed(2)}} €</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Gebühren</span>
                    <span class="detail-value">${{parseFloat(trade.fees || trade.Fees || 0).toFixed(2)}} €</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">P&L</span>
                    <span class="detail-value ${{pnlClass}}">${{pnl >= 0 ? '+' : ''}}${{pnl.toFixed(2)}} €</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Grund</span>
                    <span class="detail-value">${{trade.reason || trade.Reason || '?'}}</span>
                </div>
            `;

            document.getElementById('modalContent').innerHTML = content;
            document.getElementById('tradeModal').style.display = 'block';
        }}

        function closeModal() {{
            document.getElementById('tradeModal').style.display = 'none';
        }}

        // Close modal on outside click
        window.onclick = function(event) {{
            const modal = document.getElementById('tradeModal');
            if (event.target === modal) {{
                modal.style.display = 'none';
            }}
        }}

        // Initialize
        window.onload = drawEquityChart;
        window.onresize = drawEquityChart;
    </script>
</body>
</html>'''
        return html_content

    def _generate_positions_table(self, positions: List[Dict]) -> str:
        """Generate HTML table for open positions."""
        if not positions:
            return '<p style="color: #888; padding: 20px;">Keine offenen Positionen</p>'

        rows = ""
        for p in positions:
            symbol = html.escape(str(p.get("symbol", "?")))
            direction = html.escape(str(p.get("direction", "?")))
            entry_price = float(p.get("entry_price", 0))
            stake = float(p.get("stake", 0))
            entry_time = html.escape(str(p.get("entry_time", "?"))[:16])
            indicator = html.escape(str(p.get("indicator", "?")))
            htf = html.escape(str(p.get("htf", "?")))

            rows += f'''
            <tr>
                <td><strong>{symbol}</strong></td>
                <td>{direction.upper()}</td>
                <td>{indicator}</td>
                <td>{htf}</td>
                <td>{entry_price:,.6f}</td>
                <td>{stake:,.2f} €</td>
                <td>{entry_time}</td>
            </tr>'''

        return f'''
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Richtung</th>
                    <th>Indikator</th>
                    <th>HTF</th>
                    <th>Entry</th>
                    <th>Stake</th>
                    <th>Zeit</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>'''

    def _generate_trades_table(self, trades: List[Dict]) -> str:
        """Generate HTML table for recent trades with full details."""
        if not trades:
            return '<p style="color: #888; padding: 20px;">Keine Trades vorhanden</p>'

        rows = ""
        for i, t in enumerate(reversed(trades)):  # Most recent first
            idx = len(trades) - 1 - i
            symbol = html.escape(str(t.get("symbol", t.get("Symbol", "?"))))
            direction = html.escape(str(t.get("direction", t.get("Direction", "?"))))
            indicator = html.escape(str(t.get("indicator", t.get("Indicator", "?")))[:10])
            htf = html.escape(str(t.get("htf", t.get("HTF", "?"))))
            entry_price = float(t.get("entry_price", t.get("EntryPrice", 0)) or 0)
            exit_price = float(t.get("exit_price", t.get("ExitPrice", 0)) or 0)
            stake = float(t.get("stake", t.get("Stake", 0)) or 0)
            pnl = float(t.get("pnl", t.get("PnL", 0)) or 0)
            pnl_class = "pnl-positive" if pnl >= 0 else "pnl-negative"
            exit_time = html.escape(str(t.get("exit_time", t.get("ExitTime", t.get("ExitZeit", "?"))))[:16])
            reason = html.escape(str(t.get("reason", t.get("Reason", "?")))[:15])

            rows += f'''
            <tr onclick="showTradeDetail({idx})">
                <td><strong>{symbol}</strong></td>
                <td>{direction.upper()[:1]}</td>
                <td>{indicator}</td>
                <td>{htf}</td>
                <td>{entry_price:,.4f}</td>
                <td>{exit_price:,.4f}</td>
                <td>{stake:,.0f}€</td>
                <td class="{pnl_class}">{pnl:+,.2f}€</td>
                <td>{reason}</td>
                <td>{exit_time}</td>
            </tr>'''

        return f'''
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Dir</th>
                    <th>Indikator</th>
                    <th>HTF</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Stake</th>
                    <th>P&L</th>
                    <th>Grund</th>
                    <th>Exit Zeit</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>'''

    def _generate_orders_table(self, orders: List[Dict]) -> str:
        """Generate HTML table for recent orders."""
        if not orders:
            return '<p style="color: #888; padding: 20px;">Keine Orders vorhanden</p>'

        rows = ""
        for o in reversed(orders):  # Most recent first
            order_id = html.escape(str(o.get("order_id", "?"))[:12])
            success = o.get("success", False)
            status_class = "pnl-positive" if success else "pnl-negative"
            status_text = "OK" if success else "FEHLER"
            filled = float(o.get("filled_amount", 0))
            price = float(o.get("average_price", 0))
            timestamp = html.escape(str(o.get("timestamp", "?"))[:16])
            attempts = int(o.get("attempts", 1))

            rows += f'''
            <tr>
                <td>{order_id}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{filled:,.6f}</td>
                <td>{price:,.4f}</td>
                <td>{attempts}</td>
                <td>{timestamp}</td>
            </tr>'''

        return f'''
        <table>
            <thead>
                <tr>
                    <th>Order ID</th>
                    <th>Status</th>
                    <th>Gefüllt</th>
                    <th>Preis</th>
                    <th>Versuche</th>
                    <th>Zeit</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>'''

    def save_dashboard(self) -> str:
        """Generate and save the dashboard HTML."""
        html_content = self.generate_html()
        output_path = Path(self.config.output_dir) / "index.html"
        output_path.write_text(html_content, encoding="utf-8")
        return str(output_path)

    def start_server(self, blocking: bool = False) -> None:
        """Start the dashboard HTTP server."""
        # Store original directory
        self._original_dir = os.getcwd()

        # Generate initial dashboard
        self.save_dashboard()

        # Create handler that serves from specific directory without changing cwd
        dashboard_dir = os.path.abspath(self.config.output_dir)

        class DashboardHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=dashboard_dir, **kwargs)

            def log_message(self, format, *args):
                pass  # Suppress logging

        self._server = HTTPServer(
            (self.config.host, self.config.port),
            DashboardHandler,
        )
        self._running = True

        print(f"[Dashboard] Server running at http://{self.config.host}:{self.config.port}")

        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

        if blocking:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                self.stop_server()
        else:
            server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            server_thread.start()

    def _update_loop(self) -> None:
        """Periodically update the dashboard."""
        while self._running:
            time.sleep(self.config.refresh_interval_seconds)
            try:
                html_content = self.generate_html()
                output_path = Path(self.config.output_dir) / "index.html"
                output_path.write_text(html_content, encoding="utf-8")
            except Exception as e:
                print(f"[Dashboard] Update error: {e}")

    def stop_server(self) -> None:
        """Stop the dashboard server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            print("[Dashboard] Server stopped")


def generate_dashboard_snapshot(
    output_path: Optional[str] = None,
    state_file: str = "paper_trading_state.json",
    risk_state_file: str = "risk_state.json",
    trade_log_file: str = "paper_trading_simulation_log.csv",
) -> str:
    """Generate a single dashboard snapshot."""
    config = DashboardConfig(
        state_file=state_file,
        risk_state_file=risk_state_file,
        trade_log_file=trade_log_file,
    )
    dashboard = TradingDashboard(config)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        html_content = dashboard.generate_html()
        Path(output_path).write_text(html_content, encoding="utf-8")
        return output_path

    return dashboard.save_dashboard()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Dashboard")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--snapshot", action="store_true", help="Generate single snapshot")
    parser.add_argument("--output", type=str, default=None, help="Output path for snapshot")

    args = parser.parse_args()

    if args.serve:
        config = DashboardConfig(port=args.port)
        dashboard = TradingDashboard(config)
        dashboard.start_server(blocking=True)
    elif args.snapshot:
        path = generate_dashboard_snapshot(args.output)
        print(f"Dashboard generated: {path}")
    else:
        # Default: generate snapshot
        path = generate_dashboard_snapshot()
        print(f"Dashboard generated: {path}")
