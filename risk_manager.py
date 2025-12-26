"""Risk Management Module for Live Trading.

This module provides comprehensive risk management functionality:
- Max Drawdown Limits (daily, weekly, total)
- Daily/Weekly Loss Limits
- Position Sizing based on volatility (ATR)
- Portfolio Heat monitoring (max total exposure)
- Circuit Breaker for critical losses
- Trading session controls
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import threading

# Try to import pandas for DataFrame operations
try:
    import pandas as pd
except ImportError:
    pd = None


class TradingStatus(Enum):
    """Trading system status."""
    ACTIVE = "active"
    PAUSED = "paused"  # Temporary pause (user-initiated or recoverable)
    HALTED = "halted"  # Circuit breaker triggered
    DISABLED = "disabled"  # Manually disabled


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Capital Limits
    initial_capital: float = 14_000.0
    max_portfolio_heat: float = 0.5  # Max 50% of capital at risk
    max_single_position_pct: float = 0.15  # Max 15% per position
    min_position_size: float = 10.0  # Minimum position in quote currency

    # Drawdown Limits
    max_daily_drawdown_pct: float = 0.05  # 5% daily loss limit
    max_weekly_drawdown_pct: float = 0.10  # 10% weekly loss limit
    max_total_drawdown_pct: float = 0.20  # 20% total drawdown from peak

    # Loss Limits (absolute values, calculated from percentages if not set)
    max_daily_loss: Optional[float] = None
    max_weekly_loss: Optional[float] = None
    max_total_loss: Optional[float] = None

    # Position Limits
    max_open_positions: int = 5
    max_positions_per_symbol: int = 1
    max_correlated_positions: int = 3  # Max positions in correlated assets

    # Volatility-based sizing
    use_volatility_sizing: bool = True
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    atr_multiplier_for_stop: float = 2.0  # Stop loss at 2x ATR

    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_cooldown_hours: float = 24.0
    consecutive_loss_limit: int = 5  # Halt after 5 consecutive losses

    # Session Controls
    trading_start_hour: int = 0  # 24h format
    trading_end_hour: int = 24
    trade_on_weekends: bool = True

    def __post_init__(self):
        """Calculate absolute limits from percentages if not set."""
        if self.max_daily_loss is None:
            self.max_daily_loss = self.initial_capital * self.max_daily_drawdown_pct
        if self.max_weekly_loss is None:
            self.max_weekly_loss = self.initial_capital * self.max_weekly_drawdown_pct
        if self.max_total_loss is None:
            self.max_total_loss = self.initial_capital * self.max_total_drawdown_pct


@dataclass
class RiskState:
    """Current risk management state."""
    # Equity tracking
    peak_equity: float = 14_000.0
    current_equity: float = 14_000.0

    # P&L tracking
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    total_pnl: float = 0.0

    # Date tracking
    current_date: str = ""
    week_start_date: str = ""

    # Loss tracking
    consecutive_losses: int = 0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0

    # Status
    status: str = "active"
    status_reason: str = ""
    circuit_breaker_until: Optional[str] = None

    # Open positions tracking
    open_position_count: int = 0
    total_exposure: float = 0.0

    # History for analysis
    daily_pnl_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskState":
        """Create state from dictionary."""
        # Handle enum conversion
        if "status" in data and isinstance(data["status"], str):
            # Keep as string for JSON compatibility
            pass
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RiskManager:
    """Main risk management class."""

    RISK_STATE_FILE = "risk_state.json"
    RISK_CONFIG_FILE = "risk_config.json"

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        state_file: Optional[str] = None,
        config_file: Optional[str] = None,
    ):
        self.config = config or RiskConfig()
        self.state_file = state_file or self.RISK_STATE_FILE
        self.config_file = config_file or self.RISK_CONFIG_FILE
        self.state = self._load_state()
        self._lock = threading.Lock()

        # Update date tracking on init
        self._update_date_tracking()

    def _load_state(self) -> RiskState:
        """Load risk state from file."""
        path = Path(self.state_file)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return RiskState.from_dict(data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[RiskManager] Failed to load state: {e}")
        return RiskState(
            peak_equity=self.config.initial_capital,
            current_equity=self.config.initial_capital,
        )

    def save_state(self) -> None:
        """Save risk state to file."""
        with self._lock:
            try:
                Path(self.state_file).write_text(
                    json.dumps(self.state.to_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
            except OSError as e:
                print(f"[RiskManager] Failed to save state: {e}")

    def save_config(self) -> None:
        """Save current config to file."""
        try:
            Path(self.config_file).write_text(
                json.dumps(asdict(self.config), indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            print(f"[RiskManager] Failed to save config: {e}")

    def load_config(self) -> bool:
        """Load config from file if exists."""
        path = Path(self.config_file)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.config = RiskConfig(**data)
                return True
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[RiskManager] Failed to load config: {e}")
        return False

    def _update_date_tracking(self) -> None:
        """Update daily/weekly date tracking."""
        today = date.today().isoformat()

        # Check if new day
        if self.state.current_date != today:
            # Archive previous day's stats
            if self.state.current_date:
                self.state.daily_pnl_history.append({
                    "date": self.state.current_date,
                    "pnl": self.state.daily_pnl,
                    "trades": self.state.daily_trades,
                    "wins": self.state.daily_wins,
                    "losses": self.state.daily_losses,
                })
                # Keep only last 30 days
                if len(self.state.daily_pnl_history) > 30:
                    self.state.daily_pnl_history = self.state.daily_pnl_history[-30:]

            # Reset daily stats
            self.state.current_date = today
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.daily_wins = 0
            self.state.daily_losses = 0

        # Check if new week (Monday)
        current_week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        if self.state.week_start_date != current_week_start:
            self.state.week_start_date = current_week_start
            self.state.weekly_pnl = 0.0

    def update_equity(self, new_equity: float) -> None:
        """Update current equity and peak tracking."""
        with self._lock:
            self.state.current_equity = new_equity
            if new_equity > self.state.peak_equity:
                self.state.peak_equity = new_equity
            self.state.total_pnl = new_equity - self.config.initial_capital
            self.save_state()

    def record_trade(self, pnl: float, is_win: bool) -> None:
        """Record a completed trade for risk tracking."""
        with self._lock:
            self._update_date_tracking()

            # Update P&L
            self.state.daily_pnl += pnl
            self.state.weekly_pnl += pnl
            self.state.total_pnl += pnl
            self.state.current_equity += pnl

            # Update peak equity
            if self.state.current_equity > self.state.peak_equity:
                self.state.peak_equity = self.state.current_equity

            # Update trade counts
            self.state.daily_trades += 1
            if is_win:
                self.state.daily_wins += 1
                self.state.consecutive_losses = 0
            else:
                self.state.daily_losses += 1
                self.state.consecutive_losses += 1

            # Check circuit breaker conditions
            self._check_circuit_breaker()

            self.save_state()

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be triggered."""
        if not self.config.circuit_breaker_enabled:
            return

        triggered = False
        reason = ""

        # Check consecutive losses
        if self.state.consecutive_losses >= self.config.consecutive_loss_limit:
            triggered = True
            reason = f"Consecutive losses limit reached ({self.state.consecutive_losses})"

        # Check daily loss limit
        if self.state.daily_pnl <= -self.config.max_daily_loss:
            triggered = True
            reason = f"Daily loss limit reached ({self.state.daily_pnl:.2f})"

        # Check weekly loss limit
        if self.state.weekly_pnl <= -self.config.max_weekly_loss:
            triggered = True
            reason = f"Weekly loss limit reached ({self.state.weekly_pnl:.2f})"

        # Check total drawdown
        drawdown = self.state.peak_equity - self.state.current_equity
        if drawdown >= self.config.max_total_loss:
            triggered = True
            reason = f"Max drawdown reached ({drawdown:.2f})"

        if triggered:
            self._trigger_circuit_breaker(reason)

    def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger the circuit breaker."""
        self.state.status = TradingStatus.HALTED.value
        self.state.status_reason = reason
        cooldown_until = datetime.now() + timedelta(hours=self.config.circuit_breaker_cooldown_hours)
        self.state.circuit_breaker_until = cooldown_until.isoformat()
        print(f"[RiskManager] CIRCUIT BREAKER TRIGGERED: {reason}")
        print(f"[RiskManager] Trading halted until {cooldown_until}")

    def check_circuit_breaker_expired(self) -> bool:
        """Check if circuit breaker cooldown has expired."""
        if self.state.status != TradingStatus.HALTED.value:
            return True

        if self.state.circuit_breaker_until:
            try:
                until = datetime.fromisoformat(self.state.circuit_breaker_until)
                if datetime.now() >= until:
                    self.reset_circuit_breaker()
                    return True
            except ValueError:
                pass
        return False

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker (manual or after cooldown)."""
        with self._lock:
            self.state.status = TradingStatus.ACTIVE.value
            self.state.status_reason = ""
            self.state.circuit_breaker_until = None
            self.state.consecutive_losses = 0
            self.save_state()
            print("[RiskManager] Circuit breaker reset - trading resumed")

    def can_open_position(
        self,
        symbol: str,
        stake: float,
        current_positions: List[Dict],
    ) -> Tuple[bool, str]:
        """Check if a new position can be opened based on risk rules."""
        self._update_date_tracking()
        self.check_circuit_breaker_expired()

        # Check trading status
        if self.state.status == TradingStatus.HALTED.value:
            return False, f"Trading halted: {self.state.status_reason}"

        if self.state.status == TradingStatus.DISABLED.value:
            return False, "Trading disabled"

        if self.state.status == TradingStatus.PAUSED.value:
            return False, "Trading paused"

        # Check trading hours
        if not self._is_trading_hours():
            return False, "Outside trading hours"

        # Check max open positions
        if len(current_positions) >= self.config.max_open_positions:
            return False, f"Max open positions ({self.config.max_open_positions}) reached"

        # Check positions per symbol
        symbol_positions = sum(1 for p in current_positions if p.get("symbol") == symbol)
        if symbol_positions >= self.config.max_positions_per_symbol:
            return False, f"Max positions for {symbol} reached"

        # Check single position size
        max_position = self.state.current_equity * self.config.max_single_position_pct
        if stake > max_position:
            return False, f"Position size {stake:.2f} exceeds max {max_position:.2f}"

        # Check portfolio heat
        current_exposure = sum(float(p.get("stake", 0)) for p in current_positions)
        new_exposure = current_exposure + stake
        max_exposure = self.state.current_equity * self.config.max_portfolio_heat
        if new_exposure > max_exposure:
            return False, f"Portfolio heat {new_exposure:.2f} would exceed max {max_exposure:.2f}"

        # Check daily loss limit proximity (warn if close)
        remaining_daily = self.config.max_daily_loss + self.state.daily_pnl
        if remaining_daily < stake * 0.5:  # Less than half stake remaining
            return False, f"Near daily loss limit (remaining: {remaining_daily:.2f})"

        return True, ""

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now()

        # Check weekend
        if not self.config.trade_on_weekends and now.weekday() >= 5:
            return False

        # Check hours
        current_hour = now.hour
        if self.config.trading_end_hour > self.config.trading_start_hour:
            return self.config.trading_start_hour <= current_hour < self.config.trading_end_hour
        else:  # Overnight session
            return current_hour >= self.config.trading_start_hour or current_hour < self.config.trading_end_hour

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        direction: str = "long",
    ) -> float:
        """Calculate position size based on volatility and risk parameters."""
        if not self.config.use_volatility_sizing or atr <= 0:
            # Fallback to simple sizing
            return min(
                self.state.current_equity / self.config.max_open_positions,
                self.state.current_equity * self.config.max_single_position_pct,
            )

        # Risk-based position sizing
        # Position size = (Account Risk) / (Trade Risk per Unit)
        # Trade Risk = ATR * multiplier (distance to stop)

        account_risk = self.state.current_equity * self.config.risk_per_trade_pct
        stop_distance = atr * self.config.atr_multiplier_for_stop

        # Position size in units
        if stop_distance > 0:
            units = account_risk / stop_distance
            position_size = units * entry_price
        else:
            position_size = account_risk

        # Apply limits
        max_position = self.state.current_equity * self.config.max_single_position_pct
        min_position = self.config.min_position_size

        position_size = max(min_position, min(position_size, max_position))

        return position_size

    def get_current_drawdown(self) -> Tuple[float, float]:
        """Get current drawdown in absolute and percentage terms."""
        drawdown = self.state.peak_equity - self.state.current_equity
        drawdown_pct = drawdown / self.state.peak_equity if self.state.peak_equity > 0 else 0
        return drawdown, drawdown_pct

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of current risk metrics."""
        drawdown, drawdown_pct = self.get_current_drawdown()

        return {
            "status": self.state.status,
            "status_reason": self.state.status_reason,
            "current_equity": self.state.current_equity,
            "peak_equity": self.state.peak_equity,
            "total_pnl": self.state.total_pnl,
            "daily_pnl": self.state.daily_pnl,
            "weekly_pnl": self.state.weekly_pnl,
            "current_drawdown": drawdown,
            "current_drawdown_pct": drawdown_pct * 100,
            "max_drawdown_pct": self.config.max_total_drawdown_pct * 100,
            "daily_loss_limit": self.config.max_daily_loss,
            "daily_loss_remaining": self.config.max_daily_loss + self.state.daily_pnl,
            "consecutive_losses": self.state.consecutive_losses,
            "daily_trades": self.state.daily_trades,
            "daily_win_rate": (
                self.state.daily_wins / self.state.daily_trades * 100
                if self.state.daily_trades > 0 else 0
            ),
            "circuit_breaker_until": self.state.circuit_breaker_until,
        }

    def pause_trading(self, reason: str = "User initiated") -> None:
        """Pause trading temporarily."""
        with self._lock:
            self.state.status = TradingStatus.PAUSED.value
            self.state.status_reason = reason
            self.save_state()
            print(f"[RiskManager] Trading paused: {reason}")

    def resume_trading(self) -> None:
        """Resume trading from paused state."""
        with self._lock:
            if self.state.status == TradingStatus.PAUSED.value:
                self.state.status = TradingStatus.ACTIVE.value
                self.state.status_reason = ""
                self.save_state()
                print("[RiskManager] Trading resumed")

    def disable_trading(self, reason: str = "User disabled") -> None:
        """Disable trading completely."""
        with self._lock:
            self.state.status = TradingStatus.DISABLED.value
            self.state.status_reason = reason
            self.save_state()
            print(f"[RiskManager] Trading disabled: {reason}")

    def enable_trading(self) -> None:
        """Enable trading from disabled state."""
        with self._lock:
            if self.state.status == TradingStatus.DISABLED.value:
                self.state.status = TradingStatus.ACTIVE.value
                self.state.status_reason = ""
                self.save_state()
                print("[RiskManager] Trading enabled")

    def update_position_count(self, count: int, total_exposure: float) -> None:
        """Update current open position tracking."""
        with self._lock:
            self.state.open_position_count = count
            self.state.total_exposure = total_exposure
            self.save_state()


# Convenience function for creating a configured risk manager
def create_risk_manager(
    initial_capital: float = 14_000.0,
    max_daily_drawdown_pct: float = 0.05,
    max_total_drawdown_pct: float = 0.20,
    max_open_positions: int = 5,
    use_volatility_sizing: bool = True,
    load_from_file: bool = True,
) -> RiskManager:
    """Create a risk manager with common configuration."""
    config = RiskConfig(
        initial_capital=initial_capital,
        max_daily_drawdown_pct=max_daily_drawdown_pct,
        max_total_drawdown_pct=max_total_drawdown_pct,
        max_open_positions=max_open_positions,
        use_volatility_sizing=use_volatility_sizing,
    )

    manager = RiskManager(config=config)

    if load_from_file:
        manager.load_config()

    return manager


if __name__ == "__main__":
    # Demo/test
    rm = create_risk_manager(initial_capital=10000.0)

    print("=== Risk Manager Demo ===")
    print(f"Config: {asdict(rm.config)}")
    print(f"\nInitial Summary: {rm.get_risk_summary()}")

    # Simulate a losing trade
    rm.record_trade(-200, is_win=False)
    print(f"\nAfter -200 loss: {rm.get_risk_summary()}")

    # Check if we can open a position
    can_open, reason = rm.can_open_position("BTC/EUR", 2000, [])
    print(f"\nCan open position: {can_open}, Reason: {reason}")

    # Calculate position size
    size = rm.calculate_position_size("BTC/EUR", 50000, 1500)
    print(f"Calculated position size: {size:.2f}")
