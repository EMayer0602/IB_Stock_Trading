"""
Interactive Brokers Connector using ib_insync library.
Handles connection to TWS/Gateway, data fetching, and order execution.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from zoneinfo import ZoneInfo

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Order, Trade, Contract
    from ib_insync.util import df as ib_df
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("[Warning] ib_insync not installed. Run: pip install ib_insync")

from tickers_config import TICKERS, get_ticker_config


# Timezone
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Connection settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497  # TWS paper trading. Use 7496 for live, 4001/4002 for Gateway
DEFAULT_CLIENT_ID = 1

# Market hours (US Eastern)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0


@dataclass
class IBConfig:
    """IB Connection configuration."""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    client_id: int = DEFAULT_CLIENT_ID
    readonly: bool = False
    timeout: int = 30
    use_paper: bool = True  # If True, uses paper trading port


class IBConnector:
    """Connector for Interactive Brokers TWS/Gateway."""

    def __init__(self, config: Optional[IBConfig] = None):
        if not IB_AVAILABLE:
            raise ImportError("ib_insync is required. Install with: pip install ib_insync")

        self.config = config or IBConfig()
        self.ib = IB()
        self._connected = False
        self._contracts: Dict[str, Contract] = {}

    def connect(self) -> bool:
        """Connect to TWS/Gateway."""
        if self._connected:
            return True

        port = self.config.port
        if self.config.use_paper and port == 7496:
            port = 7497  # Switch to paper trading port

        try:
            self.ib.connect(
                host=self.config.host,
                port=port,
                clientId=self.config.client_id,
                readonly=self.config.readonly,
                timeout=self.config.timeout
            )
            self._connected = True
            print(f"[IB] Connected to TWS at {self.config.host}:{port}")
            return True
        except Exception as e:
            print(f"[IB] Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from TWS."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            print("[IB] Disconnected from TWS")

    def is_connected(self) -> bool:
        """Check if connected to TWS."""
        return self._connected and self.ib.isConnected()

    def get_contract(self, symbol: str) -> Optional[Contract]:
        """Get or create a contract for a symbol."""
        if symbol in self._contracts:
            return self._contracts[symbol]

        config = get_ticker_config(symbol)
        if not config:
            print(f"[IB] Unknown symbol: {symbol}")
            return None

        contract = Stock(
            symbol=symbol,
            exchange="SMART",
            currency="USD",
            conId=config.get("conID")
        )

        # Qualify the contract to get full details
        try:
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                self._contracts[symbol] = qualified[0]
                return qualified[0]
        except Exception as e:
            print(f"[IB] Failed to qualify contract {symbol}: {e}")

        return contract

    def fetch_historical_data(
        self,
        symbol: str,
        duration: str = "30 D",
        bar_size: str = "1 hour",
        what_to_show: str = "TRADES",
        use_rth: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock symbol
            duration: How far back to fetch (e.g., "30 D", "1 Y")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day")
            what_to_show: Type of data ("TRADES", "MIDPOINT", "BID", "ASK")
            use_rth: Use regular trading hours only

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected():
            if not self.connect():
                return None

        contract = self.get_contract(symbol)
        if not contract:
            return None

        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1
            )

            if not bars:
                print(f"[IB] No data returned for {symbol}")
                return None

            # Convert to DataFrame
            df = ib_df(bars)
            df = df.rename(columns={
                "date": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })

            # Set timestamp as index
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")

            return df

        except Exception as e:
            print(f"[IB] Error fetching data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        if not self.is_connected():
            if not self.connect():
                return None

        contract = self.get_contract(symbol)
        if not contract:
            return None

        try:
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(1)  # Wait for data

            price = ticker.marketPrice()
            if price and price > 0:
                return float(price)

            # Fallback to last price
            if ticker.last and ticker.last > 0:
                return float(ticker.last)

            # Fallback to close
            if ticker.close and ticker.close > 0:
                return float(ticker.close)

        except Exception as e:
            print(f"[IB] Error getting price for {symbol}: {e}")

        return None

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        action: str = "BUY"  # "BUY" or "SELL"
    ) -> Optional[Trade]:
        """Place a market order."""
        if not self.is_connected():
            if not self.connect():
                return None

        contract = self.get_contract(symbol)
        if not contract:
            return None

        order = MarketOrder(action, abs(quantity))

        try:
            trade = self.ib.placeOrder(contract, order)
            print(f"[IB] Placed {action} order for {quantity} {symbol}")
            return trade
        except Exception as e:
            print(f"[IB] Order failed for {symbol}: {e}")
            return None

    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        limit_price: float,
        action: str = "BUY"
    ) -> Optional[Trade]:
        """Place a limit order."""
        if not self.is_connected():
            if not self.connect():
                return None

        contract = self.get_contract(symbol)
        if not contract:
            return None

        order = LimitOrder(action, abs(quantity), limit_price)

        try:
            trade = self.ib.placeOrder(contract, order)
            print(f"[IB] Placed {action} limit order for {quantity} {symbol} @ {limit_price}")
            return trade
        except Exception as e:
            print(f"[IB] Limit order failed for {symbol}: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        if not self.is_connected():
            if not self.connect():
                return []

        positions = []
        for pos in self.ib.positions():
            positions.append({
                "symbol": pos.contract.symbol,
                "conId": pos.contract.conId,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost
            })
        return positions

    def get_account_summary(self) -> Dict:
        """Get account summary."""
        if not self.is_connected():
            if not self.connect():
                return {}

        summary = {}
        for item in self.ib.accountSummary():
            summary[item.tag] = {
                "value": item.value,
                "currency": item.currency
            }
        return summary

    def is_market_open(self) -> bool:
        """Check if US stock market is currently open."""
        now = datetime.now(NY_TZ)

        # Check weekday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:
            return False

        # Check time
        market_open = now.replace(
            hour=MARKET_OPEN_HOUR,
            minute=MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0
        )
        market_close = now.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0
        )

        return market_open <= now <= market_close

    def time_to_market_open(self) -> Optional[timedelta]:
        """Get time until market opens."""
        now = datetime.now(NY_TZ)

        # Find next market open
        next_open = now.replace(
            hour=MARKET_OPEN_HOUR,
            minute=MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0
        )

        # If it's past market open today, move to next trading day
        if now >= next_open:
            next_open += timedelta(days=1)

        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)

        return next_open - now


# Singleton instance
_connector: Optional[IBConnector] = None


def get_connector(config: Optional[IBConfig] = None) -> IBConnector:
    """Get or create the IB connector singleton."""
    global _connector
    if _connector is None:
        _connector = IBConnector(config)
    return _connector


def configure_connection(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    client_id: int = DEFAULT_CLIENT_ID,
    use_paper: bool = True
):
    """Configure the IB connection settings."""
    global _connector
    config = IBConfig(
        host=host,
        port=port,
        client_id=client_id,
        use_paper=use_paper
    )
    _connector = IBConnector(config)
    return _connector
