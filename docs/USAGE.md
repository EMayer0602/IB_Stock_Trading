# IB Stock Trading - Usage Guide

## Backtest Options

### Quick Start

```bash
# Run backtest with best strategy per symbol (default)
python backtest.py

# Run 1-year backtest
python backtest.py --period 1y

# Run sweep mode (all indicator/direction combinations)
python backtest.py --sweep
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--period` | `6mo` | Data period: `1mo`, `3mo`, `6mo`, `1y` |
| `--interval` | `1h` | Bar interval: `1h`, `1d` |
| `--best` | (default) | Use best strategy per symbol from `tickers_config.py` |
| `--sweep` | off | Test all indicator/direction combinations |
| `--symbols` | all | Specific symbols to test (sweep mode) |
| `--indicator` | all | Single indicator to test (sweep mode) |
| `--long-only` | off | Only test long positions (sweep mode) |

### Examples

```bash
# Best strategies, 1 year period
python backtest.py --best --period 1y

# Sweep all combinations for NVDA and AAPL
python backtest.py --sweep --symbols NVDA AAPL

# Test only supertrend indicator, long only
python backtest.py --sweep --indicator supertrend --long-only

# Daily bars instead of hourly
python backtest.py --interval 1d
```

### Output Files

Results are saved to `backtest_results/`:

| File | Description |
|------|-------------|
| `backtest_results.csv` | Summary per symbol/indicator/direction |
| `backtest_trades.csv` | All individual trades |
| `trades_long.csv` | Long trades only |
| `trades_short.csv` | Short trades only |
| `equity_curves.html` | Interactive equity curve charts |

---

## Parameter Sweep

Run comprehensive optimization across multiple dimensions:

```bash
python parameter_sweep.py
```

### Optimization Grid

The sweep tests combinations of:

- **Indicators**: supertrend, jma, kama
- **Directions**: long, short
- **Timeframes**: 1h, 2h, 4h
- **HTF Filters**: None, 3h, 4h, 5h, 6h, 8h, 12h, 1d
- **ATR Multipliers**: None, 1.0, 1.5, 2.0, 2.5
- **Min Hold Bars**: 0, 6, 12, 24, 48

---

## Configuration Files

### `tickers_config.py`
Per-symbol configuration with best indicator/direction:

```python
"NVDA": {
    "enabled": True,           # Include in backtest
    "indicator": "jma",        # Best indicator for this symbol
    "long": True,              # Enable long trading
    "short": False,            # Disable short trading
    "initial_capital_long": 1800,
    ...
}
```

### `config_local.py`
Local overrides for strategy parameters (not tracked in git).

---

## Git Version Control

### Updating Local Repository

```bash
# Fetch and pull latest changes
git pull origin claude/continue-project-G19u5
```

### Common Git Issues

#### 1. Lock File Error
```
fatal: Unable to create '.git/index.lock': File exists.
```

**Solution (Windows):**
```cmd
del ".git\index.lock"
```

**Solution (Linux/Mac):**
```bash
rm .git/index.lock
```

#### 2. Merge Conflicts

When local changes conflict with remote:

**Option A: Discard local changes**
```bash
git reset --hard HEAD
git clean -fd
git pull origin claude/continue-project-G19u5
```

**Option B: Stash local changes**
```bash
git stash
git pull origin claude/continue-project-G19u5
git stash pop  # Re-apply your changes
```

**Option C: Accept remote version for specific files**
```bash
git checkout --theirs path/to/file.csv
git add path/to/file.csv
git commit -m "Accept remote version"
```

#### 3. Conflicts in Output Files (CSVs)

Output files like `backtest_results/*.csv` can be regenerated. Simply accept remote and re-run:

```bash
git checkout --theirs backtest_results/
git add backtest_results/
git commit -m "Accept remote backtest results"
python backtest.py --best  # Regenerate fresh results
```

### Branch Structure

| Branch | Purpose |
|--------|---------|
| `main` | Stable production code |
| `claude/continue-project-*` | Development branches |

### Recommended Workflow

1. Always pull before making changes
2. Don't edit files that are auto-generated (CSVs, HTML reports)
3. Keep `config_local.py` for personal settings (optionally gitignored)
