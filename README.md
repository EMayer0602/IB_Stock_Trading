# IB Stock Trading - Supertrend Strategy

Automated stock trading system using the Supertrend indicator via Interactive Brokers.

## Features

- **Supertrend Strategy**: Multiple indicator support (Supertrend, JMA, KAMA, PSAR)
- **Interactive Brokers Integration**: Direct connection to TWS/Gateway
- **Paper Trading**: Simulate trades before going live
- **Risk Management**: Position sizing, max drawdown limits
- **Web Dashboard**: Real-time monitoring

## Prerequisites

1. **Interactive Brokers Account**
   - TWS (Trader Workstation) or IB Gateway installed
   - API enabled in TWS settings

2. **Python 3.9+**

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## TWS Configuration

1. Open TWS and go to **Edit > Global Configuration**
2. Navigate to **API > Settings**
3. Enable:
   - "Enable ActiveX and Socket Clients"
   - "Allow connections from localhost only"
4. Set Socket Port:
   - **7497** for Paper Trading
   - **7496** for Live Trading

## Quick Start

### 1. Paper Trading (Simulation)

```bash
# Show current status
python paper_trader.py --status

# Run one signal check
python paper_trader.py --check

# Start continuous monitoring (hourly)
python paper_trader.py --monitor --interval 60

# Reset to initial capital
python paper_trader.py --reset
```

### 2. With Dashboard

```bash
python paper_trader.py --monitor --dashboard
# Open http://localhost:8080
```

## Configuration

### Ticker Configuration (`tickers_config.py`)

Each ticker can be configured with:
- `conID`: IB contract ID
- `long`/`short`: Enable/disable direction
- `initial_capital_long/short`: Capital allocation
- `order_round_factor`: Round shares to nearest X
- `trade_on`: "Open" or "Close" timing

### Strategy Settings (`supertrend_strategy.py`)

Key settings:
- `TIMEFRAME`: Default "1h"
- `HIGHER_TIMEFRAME`: HTF filter, default "1d"
- `ENABLE_LONGS`/`ENABLE_SHORTS`: Direction toggles

### Local Config Override

Create `config_local.py` to override settings:
```python
TIMEFRAME = "15m"
HIGHER_TIMEFRAME = "4h"
ENABLE_SHORTS = False
```

## File Structure

```
IB_Stock_Trading/
├── paper_trader.py         # Main trading script
├── supertrend_strategy.py  # Strategy & indicators
├── ib_connector.py         # IB API wrapper
├── tickers_config.py       # Stock configuration
├── risk_manager.py         # Risk management
├── trading_dashboard.py    # Web dashboard
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Tickers Included

| Symbol | Company | Long | Short |
|--------|---------|------|-------|
| AAPL | Apple | ✓ | ✓ |
| GOOGL | Alphabet | ✓ | ✓ |
| NVDA | NVIDIA | ✓ | ✓ |
| MSFT | Microsoft | ✓ | ✓ |
| META | Meta | ✓ | ✓ |
| AMD | AMD | ✓ | ✓ |
| TSLA | Tesla | ✓ | ✓ |
| AMZN | Amazon | ✓ | ✓ |
| NFLX | Netflix | ✓ | ✓ |
| INTC | Intel | ✓ | ✓ |
| MRNA | Moderna | ✓ | ✓ |
| QUBT | Quantum Computing | ✓ | ✓ |
| QBTS | D-Wave | ✓ | ✓ |
| BRRR | Valkyrie Bitcoin | ✓ | ✓ |

## Market Hours

- **US Market**: 9:30 AM - 4:00 PM Eastern
- System automatically waits outside market hours

## Troubleshooting

### Connection Issues
- Ensure TWS is running with API enabled
- Check port (7497 for paper, 7496 for live)
- Only one client can use a client ID at a time

### No Data
- TWS needs market data subscriptions
- Some symbols may need specific subscriptions

### Order Rejected
- Check account permissions
- Verify margin requirements
- Ensure symbol is tradeable
