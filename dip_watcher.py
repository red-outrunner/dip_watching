#!/usr/bin/env python3
"""
dip_watcher.py
Continuously monitor a list of tickers for a configurable "dip entry window" via GUI.

Requirements:
    pip install yfinance pandas PyQt6

Usage:
    python dip_watcher.py
"""

from __future__ import annotations

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yfinance as yf

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QPushButton,
    QVBoxLayout, QWidget, QInputDialog, QMessageBox, QHeaderView, QDialog
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor


class DipWatcher:
    """Core logic for detecting dip entry windows."""

    def __init__(
        self,
        tickers: List[str],
        dip_threshold: float = 0.15,
        max_ask_spread: float = 0.02,
        lookback_periods: Tuple[int, ...] = (5, 20),
        csv_file: str = "dip_alerts.csv",
        volume_multipliers: Dict[str, float] = None,
    ):
        self.tickers = [self._format_ticker(t) for t in tickers]
        self.dip_threshold = dip_threshold
        self.max_ask_spread = max_ask_spread
        self.lookback_periods = lookback_periods
        self.csv_file = Path(csv_file)
        self.volume_multipliers = volume_multipliers or {'US': 0.8, 'JSE': 0.5}
        self._init_csv()
    
    def _format_ticker(self, ticker: str) -> str:
        """Format ticker symbol for the appropriate exchange."""
        ticker = ticker.upper()
        
        # JSE stocks: Add .JO suffix if not present
        if self._is_jse_stock(ticker):
            if not ticker.endswith('.JO'):
                ticker += '.JO'
        
        return ticker
    
    def _is_jse_stock(self, ticker: str) -> bool:
        """Check if ticker is likely a JSE stock."""
        base_ticker = ticker.replace('.JO', '')
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
            'TKG', 'TSG', 'UCT', 'WEQ', 'ZED', 'HIL'
        }
        return base_ticker in jse_stocks

    def _init_csv(self) -> None:
        if not self.csv_file.exists():
            header = (
                "timestamp,symbol,last_price,bid,ask,spread,dip_pct,"
                + ",".join([f"SMA_{p}" for p in self.lookback_periods])
                + ",volume,avg_volume\n"
            )
            self.csv_file.write_text(header)

    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch and compute stock data, return dict or None on error."""
        tk = yf.Ticker(symbol)

        try:
            info = tk.info
            last_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not last_price:
                recent = tk.history(period="1d", interval="1m")
                if recent.empty:
                    raise ValueError("No current price data available")
                last_price = float(recent['Close'].iloc[-1])
            else:
                last_price = float(last_price)
            
            bid = info.get('bid', last_price * 0.999)
            ask = info.get('ask', last_price * 1.001)
            bid = float(bid) if bid else last_price * 0.999
            ask = float(ask) if ask else last_price * 1.001
            
            prev_close = info.get('previousClose')
            change_pct = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0
            
        except Exception:
            return None

        try:
            hist = tk.history(period="3mo", interval="1d")
            if hist.empty or len(hist) < max(self.lookback_periods):
                raise ValueError(f"Insufficient historical data")
        except Exception:
            return None

        smas = {}
        for period in self.lookback_periods:
            if len(hist) >= period:
                smas[f"SMA_{period}"] = hist["Close"].tail(period).mean()
            else:
                return None

        highest_sma = max(smas.values())
        dip_pct = (highest_sma - last_price) / highest_sma
        spread = (ask - bid) / last_price if last_price > 0 else 0
        volume_data = hist["Volume"].tail(20)
        avg_volume = volume_data.mean()
        current_volume = int(info.get('volume', avg_volume)) or int(avg_volume)
        
        exchange = 'JSE' if symbol.endswith('.JO') else 'US'
        volume_multiplier = self.volume_multipliers.get(exchange, 0.8)

        window_open = (
            dip_pct >= self.dip_threshold
            and spread <= self.max_ask_spread
            and current_volume >= avg_volume * volume_multiplier
        )

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

        currency = "R" if symbol.endswith('.JO') else "$"

        return {
            'last_price': last_price,
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'dip_pct': dip_pct,
            'smas': smas,
            'volume': current_volume,
            'avg_volume': avg_volume,
            'previous_close': prev_close,
            'change_pct': change_pct,
            'window_open': window_open,
            'currency': currency,
            'exchange': exchange
        }

    def _append_csv(self, row: Dict) -> None:
        try:
            df = pd.DataFrame([row])
            df.to_csv(self.csv_file, mode="a", header=False, index=False)
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dip Watching")
        self.setGeometry(100, 100, 800, 600)

        self.watchlist_file = Path("watchlist.json")
        self.watchlist: List[Dict] = self.load_watchlist()  # {'ticker': str, 'target': float|None, 'notified_target': bool, 'notified_dip': bool}
        self.watcher = DipWatcher(
            [stock['ticker'] for stock in self.watchlist],
            dip_threshold=0.15,
            max_ask_spread=0.02,
            lookback_periods=(5, 20),
            csv_file="dip_alerts.csv",
            volume_multipliers={'US': 0.8, 'JSE': 0.5}
        )

        self.table = QTableWidget(len(self.watchlist), 5)
        self.table.setHorizontalHeaderLabels(['Ticker', 'Last Price', 'Change %', 'Target Price', 'Status'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for i, stock in enumerate(self.watchlist):
            self.table.setItem(i, 0, QTableWidgetItem(stock['ticker']))
            target_item = QTableWidgetItem(str(stock['target']) if stock['target'] is not None else "")
            target_item.setFlags(Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(i, 3, target_item)

        add_btn = QPushButton("Add Ticker")
        add_btn.clicked.connect(self.add_ticker)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_ticker)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.open_settings)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(add_btn)
        layout.addWidget(remove_btn)
        layout.addWidget(settings_btn)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.interval = 60000  # 60 seconds in ms
        self.timer.start(self.interval)

    def load_watchlist(self) -> List[Dict]:
        """Load watchlist from JSON file."""
        if self.watchlist_file.exists():
            try:
                with self.watchlist_file.open('r') as f:
                    data = json.load(f)
                    # Validate and format watchlist entries
                    watchlist = []
                    for item in data:
                        ticker = self.watcher._format_ticker(item.get('ticker', ''))
                        target = item.get('target')
                        if isinstance(target, (int, float)) or target is None:
                            watchlist.append({
                                'ticker': ticker,
                                'target': target,
                                'notified_target': False,
                                'notified_dip': False
                            })
                    return watchlist[:25]  # Enforce max 25
            except Exception:
                return []
        return []

    def save_watchlist(self) -> None:
        """Save watchlist to JSON file."""
        try:
            with self.watchlist_file.open('w') as f:
                json.dump([
                    {'ticker': stock['ticker'], 'target': stock['target']}
                    for stock in self.watchlist
                ], f, indent=2)
        except Exception:
            pass

    def add_ticker(self):
        if len(self.watchlist) >= 25:
            QMessageBox.warning(self, "Limit Reached", "You can add up to 25 stocks to the watchlist.")
            return

        ticker, ok = QInputDialog.getText(self, "Add Ticker", "Enter ticker symbol:")
        if ok and ticker:
            formatted = self.watcher._format_ticker(ticker)
            if formatted not in [s['ticker'] for s in self.watchlist]:
                self.watchlist.append({'ticker': formatted, 'target': None, 'notified_target': False, 'notified_dip': False})
                self.watcher.tickers.append(formatted)
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(formatted))
                target_item = QTableWidgetItem()
                target_item.setFlags(Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, 3, target_item)
                self.save_watchlist()

    def remove_ticker(self):
        row = self.table.currentRow()
        if row >= 0:
            ticker = self.watchlist[row]['ticker']
            del self.watchlist[row]
            self.watcher.tickers.remove(ticker)
            self.table.removeRow(row)
            self.save_watchlist()

    def update_data(self):
        for i, stock in enumerate(self.watchlist):
            data = self.watcher.get_stock_data(stock['ticker'])
            if data is None:
                self.table.setItem(i, 4, QTableWidgetItem("Error"))
                continue

            # Last Price
            price_str = f"{data['currency']}{data['last_price']:.2f}"
            self.table.setItem(i, 1, QTableWidgetItem(price_str))

            # Change %
            change_str = f"{data['change_pct']:.2f}%"
            change_item = QTableWidgetItem(change_str)
            if data['change_pct'] > 0:
                change_item.setForeground(QColor("green"))
            elif data['change_pct'] < 0:
                change_item.setForeground(QColor("red"))
            self.table.setItem(i, 2, change_item)

            # Target Price (editable)
            target_item = self.table.item(i, 3)
            if target_item and target_item.text():
                try:
                    new_target = float(target_item.text())
                    if stock['target'] != new_target:
                        stock['target'] = new_target
                        stock['notified_target'] = False  # Reset notification if target changes
                        self.save_watchlist()
                except ValueError:
                    stock['target'] = None
                    self.save_watchlist()

            # Status and Notifications
            status = ""
            if data['window_open']:
                status = "Dip Alert!"
                if not stock['notified_dip']:
                    QMessageBox.information(self, "Dip Alert", f"{stock['ticker']} has a dip entry window open!")
                    stock['notified_dip'] = True

            if stock['target'] is not None and data['last_price'] <= stock['target']:
                if not stock['notified_target']:
                    QMessageBox.information(self, "Target Reached", f"{stock['ticker']} reached target price {data['currency']}{stock['target']:.2f}!")
                    stock['notified_target'] = True

            self.table.setItem(i, 4, QTableWidgetItem(status))

    def open_settings(self):
        dip_th, ok = QInputDialog.getDouble(self, "Dip Threshold", "Enter dip threshold (0-1):", self.watcher.dip_threshold, 0, 1, decimals=2)
        if ok:
            self.watcher.dip_threshold = dip_th

        max_sp, ok = QInputDialog.getDouble(self, "Max Ask Spread", "Enter max ask spread (0-1):", self.watcher.max_ask_spread, 0, 1, decimals=2)
        if ok:
            self.watcher.max_ask_spread = max_sp

        lookback_str, ok = QInputDialog.getText(self, "Lookback Periods", "Enter lookback periods (comma separated):", text=",".join(map(str, self.watcher.lookback_periods)))
        if ok:
            try:
                self.watcher.lookback_periods = tuple(map(int, lookback_str.split(',')))
            except ValueError:
                pass

        interval, ok = QInputDialog.getInt(self, "Interval (seconds)", "Enter poll interval:", self.interval // 1000, 10, 3600)
        if ok:
            self.interval = interval * 1000
            self.timer.setInterval(self.interval)

        us_mult, ok = QInputDialog.getDouble(self, "US Volume Multiplier", "Enter US volume multiplier:", self.watcher.volume_multipliers.get('US', 0.8), 0, 2, decimals=2)
        if ok:
            self.watcher.volume_multipliers['US'] = us_mult

        jse_mult, ok = QInputDialog.getDouble(self, "JSE Volume Multiplier", "Enter JSE volume multiplier:", self.watcher.volume_multipliers.get('JSE', 0.5), 0, 2, decimals=2)
        if ok:
            self.watcher.volume_multipliers['JSE'] = jse_mult


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
