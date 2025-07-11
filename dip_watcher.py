#!/usr/bin/env python3
"""
dip_watcher.py
Continuously monitor a list of tickers for a configurable “dip entry window”.

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
        self.tickers = [t.upper() for t in tickers]
        self.dip_threshold = dip_threshold
        self.max_ask_spread = max_ask_spread
        self.lookback_periods = lookback_periods
        self.csv_file = Path(csv_file)
        self._init_csv()

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
        while True:
            self.scan_once()
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _init_csv(self) -> None:
        if not self.csv_file.exists():
            self.csv_file.write_text(
                "timestamp,symbol,last_price,bid,ask,spread,dip_pct,"
                + ",".join([f"SMA_{p}" for p in self.lookback_periods])
                + ",volume\n"
            )

    def _process_symbol(self, symbol: str) -> None:
        tk = yf.Ticker(symbol)

        # 1. Current quote ------------------------------------------------
        quote = tk.fast_info  # lightweight snapshot
        last_price = float(quote["lastPrice"])
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        volume = int(quote["lastVolume"])

        if not all([last_price, bid, ask, volume]):
            raise ValueError("Incomplete quote data")

        # 2. Historical prices for SMAs & volume -------------------------
        hist = tk.history(period="1mo", interval="1d")
        if hist.empty:
            raise ValueError("No historical data")

        smas = {}
        for period in self.lookback_periods:
            sma_key = f"SMA_{period}"
            smas[sma_key] = hist["Close"].tail(period).mean()

        # Highest SMA for dip calculation
        highest_sma = max(smas.values())
        dip_pct = (highest_sma - last_price) / highest_sma

        # 3. Bid–ask spread
        spread = (ask - bid) / last_price

        # 4. Volume filter (20-day average)
        avg_volume = hist["Volume"].tail(20).mean()

        # 5. Entry window conditions
        window_open = (
            dip_pct >= self.dip_threshold
            and ask <= last_price * (1 + self.max_ask_spread)
            and volume >= avg_volume
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
                "volume": volume,
                **smas,
            }
            self._append_csv(row)
            click.echo(self._format_row(row))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _append_csv(self, row: Dict) -> None:
        df = pd.DataFrame([row])
        df.to_csv(self.csv_file, mode="a", header=False, index=False)

    @staticmethod
    def _format_row(row: Dict) -> str:
        ts = row["timestamp"].replace("T", " ")  # prettier
        return (
            f"{ts}  {row['symbol']:5}  "
            f"price={row['last_price']:.2f}  "
            f"bid={row['bid']:.2f}  ask={row['ask']:.2f}  "
            f"spread={row['spread']:.2%}  "
            f"dip={row['dip_pct']:.2%}"
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def load_config(path: str) -> Dict:
    if Path(path).exists():
        with open(path) as fp:
            return json.load(fp)
    return {}


@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--dip-threshold", type=float, help="Min dip vs SMA to trigger")
@click.option("--max-ask-spread", type=float, help="Max ask spread over last price")
@click.option(
    "--lookback",
    multiple=True,
    type=int,
    help="Lookback periods for SMA (repeatable)",
)
@click.option("--csv", "csv_file", default="dip_alerts.csv", help="Output CSV file")
@click.option("--interval", default=60, help="Poll interval in seconds")
@click.option("--config", default=None, help="JSON config file (CLI flags override)")
@click.option("--once", is_flag=True, help="Run once and exit")
def main(tickers, dip_threshold, max_ask_spread, lookback, csv_file, interval, config, once):
    cfg = load_config(config or "")

    # Merge CLI and config file values (CLI wins)
    dip_threshold = dip_threshold or cfg.get("dip_threshold", 0.15)
    max_ask_spread = max_ask_spread or cfg.get("max_ask_spread", 0.02)
    lookback = lookback or cfg.get("lookback_periods", [5, 20])
    csv_file = csv_file or cfg.get("csv_file", "dip_alerts.csv")

    watcher = DipWatcher(
        tickers=list(tickers),
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
