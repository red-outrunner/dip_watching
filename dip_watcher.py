#!/usr/bin/env python3
"""
dip_watcher.py
Continuously monitor a list of tickers for a configurable "dip entry window".

Requirements:
    pip install yfinance pandas click

Usage examples
--------------
# Real-time polling every 60 s
python dip_watcher.py AAPL MSFT --dip-threshold 0.15 --max-ask-spread 0.02 --lookback 5 20

# One-shot run
python dip_watcher.py TSLA --once --config config.json

# Write to custom CSV
python dip_watcher.py GME --csv gme_alerts.csv
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import click
import pandas as pd
import yfinance as yf


class DipWatcher:
    """Core logic for detecting dip entry windows."""

    def __init__(
        self,
        tickers: List[str],
        dip_threshold: float = 0.15,
        max_ask_spread: float = 0.02,
        lookback_periods: Tuple[int, ...] = (5, 20),
        csv_file: str = "dip_alerts.csv",
    ):
        self.tickers = [self._format_ticker(t) for t in tickers]
        self.dip_threshold = dip_threshold
        self.max_ask_spread = max_ask_spread
        self.lookback_periods = lookback_periods
        self.csv_file = Path(csv_file)
        self._init_csv()
    
    def _format_ticker(self, ticker: str) -> str:
        """Format ticker symbol for the appropriate exchange."""
        ticker = ticker.upper()
        
        # JSE stocks: Add .JO suffix if not present
        # Common JSE stocks: SHP, NPN, ABG, AGL, etc.
        if self._is_jse_stock(ticker):
            if not ticker.endswith('.JO'):
                ticker += '.JO'
        
        return ticker
    
    def _is_jse_stock(self, ticker: str) -> bool:
        """Check if ticker is likely a JSE stock."""
        # Remove .JO suffix for checking
        base_ticker = ticker.replace('.JO', '')
        
        # Known JSE stocks - explicit list to avoid false positives
        jse_stocks = {
            'SHP', 'NPN', 'ABG', 'AGL', 'APN', 'ARI', 'BAW', 'BID', 'BVT', 'CFR',
            'CLS', 'CPI', 'DSY', 'EXX', 'FSR', 'GFI', 'GLN', 'GRT', 'HAR', 'IMP',
            'INL', 'INP', 'JSE', 'KAP', 'LHC', 'MCG', 'MND', 'MNP', 'MRP', 'MSM',
            'MTN', 'NED', 'NHM', 'NTC', 'OML', 'PIK', 'PRX', 'PSG', 'RBX', 'RDF',
            'REM', 'RMH', 'RMI', 'SAP', 'SBK', 'SLM', 'SNH', 'SOL', 'SPP', 'SSW',
            'SUI', 'TBS', 'TKG', 'TRU', 'VOD', 'WHL', 'WBO', 'PPE', 'AMS', 'SOC',
            'SAN', 'KIO', 'OAS', 'RES', 'ANG', 'BHP', 'BTI', 'DRD', 'HAR', 'IGL',
            'INC', 'ITU', 'KST', 'LBT', 'MPC', 'MTA', 'NRP', 'OMU', 'PAN', 'PFG',
            'PPC', 'REI', 'RTO', 'SBP', 'SGL', 'SHF', 'SLR', 'SPG', 'TFG', 'THA',
            'TKG', 'TSG', 'UCT', 'WEQ', 'ZED', 'HIL', 'CTA', 'NVS'
        }
        
        # Only return True if it's a known JSE stock
        return base_ticker in jse_stocks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scan_once(self) -> None:
        """Single pass through all tickers."""
        for symbol in self.tickers:
            try:
                self._process_symbol(symbol)
            except Exception as exc:
                click.echo(f"[{symbol}] Error: {exc}", err=True)

    def run_forever(self, interval: int = 60) -> None:
        """Run continuously until KeyboardInterrupt."""
        click.echo(f"Monitoring {self.tickers} every {interval}s ... (Ctrl-C to stop)")
        try:
            while True:
                self.scan_once()
                time.sleep(interval)
        except KeyboardInterrupt:
            click.echo("\nStopping monitor...")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _init_csv(self) -> None:
        if not self.csv_file.exists():
            header = (
                "timestamp,symbol,last_price,bid,ask,spread,dip_pct,"
                + ",".join([f"SMA_{p}" for p in self.lookback_periods])
                + ",volume,avg_volume\n"
            )
            self.csv_file.write_text(header)

    def _process_symbol(self, symbol: str) -> None:
        tk = yf.Ticker(symbol)

        # 1. Get current price data
        try:
            # Use info instead of fast_info for more reliable data
            info = tk.info
            last_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not last_price:
                # Fallback to history if info doesn't have price
                recent = tk.history(period="1d", interval="1m")
                if recent.empty:
                    raise ValueError("No current price data available")
                last_price = float(recent['Close'].iloc[-1])
            else:
                last_price = float(last_price)
            
            # Get bid/ask if available, otherwise estimate
            bid = info.get('bid', last_price * 0.999)  # Rough estimate
            ask = info.get('ask', last_price * 1.001)  # Rough estimate
            
            # Convert to float and handle None values
            bid = float(bid) if bid else last_price * 0.999
            ask = float(ask) if ask else last_price * 1.001
            
        except Exception as e:
            click.echo(f"[{symbol}] Failed to get current price: {e}", err=True)
            return

        # 2. Historical prices for SMAs & volume
        try:
            # Get more data to ensure we have enough for calculations
            hist = tk.history(period="3mo", interval="1d")
            if hist.empty or len(hist) < max(self.lookback_periods):
                raise ValueError(f"Insufficient historical data (need {max(self.lookback_periods)} days)")
        except Exception as e:
            click.echo(f"[{symbol}] Failed to get historical data: {e}", err=True)
            return

        # Calculate SMAs
        smas = {}
        for period in self.lookback_periods:
            if len(hist) >= period:
                sma_key = f"SMA_{period}"
                smas[sma_key] = hist["Close"].tail(period).mean()
            else:
                click.echo(f"[{symbol}] Warning: Not enough data for {period}-day SMA", err=True)
                return

        # Highest SMA for dip calculation
        highest_sma = max(smas.values())
        dip_pct = (highest_sma - last_price) / highest_sma

        # 3. Bid–ask spread
        spread = (ask - bid) / last_price if last_price > 0 else 0

        # 4. Volume filter (20-day average)
        volume_data = hist["Volume"].tail(20)
        avg_volume = volume_data.mean()
        
        # Get current volume (estimate if not available)
        try:
            current_volume = info.get('volume', avg_volume)
            current_volume = int(current_volume) if current_volume else int(avg_volume)
        except:
            current_volume = int(avg_volume)
        
        # JSE stocks might have lower volume, so adjust threshold
        volume_multiplier = 0.5 if symbol.endswith('.JO') else 0.8

        # 5. Entry window conditions
        window_open = (
            dip_pct >= self.dip_threshold
            and spread <= self.max_ask_spread
            and current_volume >= avg_volume * volume_multiplier  # Adjusted for JSE
        )

        # 6. Log & persist
        if window_open:
            row = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "symbol": symbol,
                "last_price": last_price,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "dip_pct": dip_pct,
                "volume": current_volume,
                "avg_volume": avg_volume,
                **smas,
            }
            self._append_csv(row)
            click.echo(self._format_row(row))
        else:
            # Show status even when no alert - format price based on exchange
            currency = "R" if symbol.endswith('.JO') else "$"
            click.echo(f"[{symbol}] Price: {currency}{last_price:.2f}, Dip: {dip_pct:.2%} (need {self.dip_threshold:.2%})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _append_csv(self, row: Dict) -> None:
        try:
            df = pd.DataFrame([row])
            df.to_csv(self.csv_file, mode="a", header=False, index=False)
        except Exception as e:
            click.echo(f"Failed to write to CSV: {e}", err=True)

    @staticmethod
    def _format_row(row: Dict) -> str:
        ts = row["timestamp"].replace("T", " ")  # prettier
        symbol = row['symbol']
        currency = "R" if symbol.endswith('.JO') else "$"
        return (
            f"🚨 ALERT! {ts}  {symbol:8}  "
            f"price={currency}{row['last_price']:.2f}  "
            f"bid={currency}{row['bid']:.2f}  ask={currency}{row['ask']:.2f}  "
            f"spread={row['spread']:.2%}  "
            f"dip={row['dip_pct']:.2%}  "
            f"vol={row['volume']:,} (avg: {row['avg_volume']:,.0f})"
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def load_config(path: str) -> Dict:
    if Path(path).exists():
        try:
            with open(path) as fp:
                return json.load(fp)
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
    return {}


@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--dip-threshold", type=float, help="Min dip vs SMA to trigger (default: 0.15)")
@click.option("--max-ask-spread", type=float, help="Max ask spread over last price (default: 0.02)")
@click.option(
    "--lookback",
    multiple=True,
    type=int,
    help="Lookback periods for SMA (repeatable, default: 5,20)",
)
@click.option("--csv", "csv_file", default="dip_alerts.csv", help="Output CSV file")
@click.option("--interval", default=60, help="Poll interval in seconds")
@click.option("--config", default=None, help="JSON config file (CLI flags override)")
@click.option("--once", is_flag=True, help="Run once and exit")
def main(tickers, dip_threshold, max_ask_spread, lookback, csv_file, interval, config, once):
    """
    Monitor stock tickers for dip entry opportunities.
    
    Supports both US stocks (e.g., AAPL) and JSE stocks (e.g., SHP, NPN).
    JSE stocks will automatically get .JO suffix added.
    
    Examples:
        python dip_watcher.py AAPL MSFT --dip-threshold 0.10 --interval 30
        python dip_watcher.py SHP NPN ABG --dip-threshold 0.15 --interval 60
        python dip_watcher.py AAPL SHP MSFT NPN --once
    """
    cfg = load_config(config or "")

    # Merge CLI and config file values (CLI wins)
    dip_threshold = dip_threshold or cfg.get("dip_threshold", 0.15)
    max_ask_spread = max_ask_spread or cfg.get("max_ask_spread", 0.02)
    
    # Handle lookback periods correctly - convert tuple to list
    if lookback:
        lookback = list(lookback)
    else:
        lookback = cfg.get("lookback_periods", [5, 20])
    
    csv_file = csv_file or cfg.get("csv_file", "dip_alerts.csv")

    # Validate inputs
    if dip_threshold <= 0 or dip_threshold >= 1:
        click.echo("Error: dip-threshold must be between 0 and 1", err=True)
        return
    
    if max_ask_spread <= 0 or max_ask_spread >= 1:
        click.echo("Error: max-ask-spread must be between 0 and 1", err=True)
        return

    # Validate tickers - only process valid ticker symbols
    valid_tickers = []
    for ticker in tickers:
        # Skip if it's just a number (from CLI parsing confusion)
        if ticker.isdigit():
            click.echo(f"Warning: Skipping '{ticker}' - not a valid ticker symbol", err=True)
            continue
        valid_tickers.append(ticker)
    
    if not valid_tickers:
        click.echo("Error: No valid ticker symbols provided", err=True)
        return

    click.echo(f"Configuration:")
    click.echo(f"  Tickers: {valid_tickers}")
    click.echo(f"  Dip threshold: {dip_threshold:.2%}")
    click.echo(f"  Max spread: {max_ask_spread:.2%}")
    click.echo(f"  Lookback periods: {lookback}")
    click.echo(f"  CSV file: {csv_file}")
    click.echo("")

    watcher = DipWatcher(
        tickers=valid_tickers,
        dip_threshold=dip_threshold,
        max_ask_spread=max_ask_spread,
        lookback_periods=tuple(lookback),
        csv_file=csv_file,
    )

    if once:
        watcher.scan_once()
    else:
        watcher.run_forever(interval=interval)


if __name__ == "__main__":
    main()
