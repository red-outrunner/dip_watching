#!/usr/bin/env python3
"""
dip_watcher.py
Continuously monitor a list of tickers for a configurable "dip entry window" via GUI.

Requirements:
    pip install yfinance pandas PyQt6 plotly

Usage:
    python dip_watcher.py

On Pop!_OS/KDE, ensure dependencies:
    sudo apt install libxcb-cursor0 libx11-xcb1 libxcb1 libxcb-xkb1 libxkbcommon-x11-0
    For KDE Wayland: sudo apt install xwayland
"""

from __future__ import annotations

import sys
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QPushButton,
    QVBoxLayout, QWidget, QInputDialog, QMessageBox, QHeaderView, QDialog,
    QFileDialog, QDialogButtonBox, QLabel, QCheckBox
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QAction
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtWebEngineWidgets import QWebEngineView


class DataWorker(QThread):
    data_updated = pyqtSignal(list)

    def __init__(self, watcher: 'DipWatcher', watchlist: List[Dict], interval: int):
        super().__init__()
        self.watcher = watcher
        self.watchlist = watchlist
        self.interval = interval / 1000
        self.running = True

    def run(self):
        while self.running:
            results = []
            for stock in self.watchlist:
                data = self.watcher.get_stock_data(stock['ticker'])
                results.append((stock, data))
            self.data_updated.emit(results)
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.wait()


class DipWatcher:
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
        self.volume_multipliers = volume_multipliers or {'US': 0.8, 'JSE': 0.5, 'LSE': 0.7}
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._init_csv()
    
    def _format_ticker(self, ticker: str) -> str:
        ticker = ticker.upper()
        if self._is_jse_stock(ticker):
            if not ticker.endswith('.JO'):
                ticker += '.JO'
        elif self._is_lse_stock(ticker):
            if not ticker.endswith('.L'):
                ticker += '.L'
        return ticker
    
    def _is_jse_stock(self, ticker: str) -> bool:
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

    def _is_lse_stock(self, ticker: str) -> bool:
        base_ticker = ticker.replace('.L', '')
        lse_stocks = {
            'VOD', 'BP', 'HSBA', 'SHEL', 'AZN', 'GSK', 'BATS', 'DGE', 'ULVR', 'REL',
            'LLOY', 'BARC', 'RBS', 'STAN', 'PRU', 'AV', 'LGEN', 'RIO', 'GLEN', 'AAL',
            'ANTO', 'TSCO', 'SBRY', 'MKS', 'NG', 'SSE', 'CNA', 'BT.A', 'IMB', 'ABF',
            'SKG', 'MNDI', 'PSON', 'RTO', 'ITV', 'WPP', 'BRBY', 'HLN', 'CPG', 'BA',
            'RR', 'IAG', 'EZJ', 'LSEG', 'III', 'ADM', 'SGE', 'EXPN', 'CRDA', 'SPX'
        }
        return base_ticker in lse_stocks

    def _init_csv(self) -> None:
        if not self.csv_file.exists():
            header = (
                "timestamp,symbol,exchange,last_price,bid,ask,spread,dip_pct,"
                + ",".join([f"SMA_{p}" for p in self.lookback_periods])
                + ",volume,avg_volume\n"
            )
            self.csv_file.write_text(header)

    def validate_ticker(self, ticker: str) -> bool:
        try:
            tk = yf.Ticker(self._format_ticker(ticker))
            info = tk.info
            return bool(info.get('currentPrice') or info.get('regularMarketPrice'))
        except Exception:
            return False

    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{symbol.replace('.', '_')}.json"
        tk = yf.Ticker(symbol)

        use_cache = False
        cache_data = None
        if cache_file.exists():
            try:
                with cache_file.open('r') as f:
                    cache_data = json.load(f)
                    cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01T00:00:00'))
                    if datetime.utcnow() - cache_time < timedelta(hours=24):
                        use_cache = True
                    else:
                        cache_file.unlink(missing_ok=True)
            except Exception:
                cache_file.unlink(missing_ok=True)

        if not use_cache:
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

                hist = tk.history(period="3mo", interval="1d")
                if hist.empty or len(hist) < max(self.lookback_periods):
                    raise ValueError("Insufficient historical data")

                cache_data = {
                    'info': info,
                    'history': hist.to_dict(),
                    'timestamp': datetime.utcnow().isoformat()
                }
                try:
                    with cache_file.open('w') as f:
                        json.dump(cache_data, f)
                except Exception:
                    pass
            except Exception:
                if cache_file.exists():
                    try:
                        with cache_file.open('r') as f:
                            cache_data = json.load(f)
                            cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01T00:00:00'))
                            if datetime.utcnow() - cache_time >= timedelta(hours=24):
                                cache_file.unlink(missing_ok=True)
                                return None
                    except Exception:
                        cache_file.unlink(missing_ok=True)
                        return None
                else:
                    return None
        else:
            info = cache_data.get('info', {})
            hist_data = cache_data.get('history', {})
            hist = pd.DataFrame(hist_data)
            if hist.empty:
                return None
            last_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not last_price:
                return None
            last_price = float(last_price)
            bid = float(info.get('bid', last_price * 0.999))
            ask = float(info.get('ask', last_price * 1.001))
            prev_close = info.get('previousClose')
            change_pct = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0

        smas = {}
        for period in self.lookback_periods:
            if len(hist) >= period:
                smas[f"SMA_{period}"] = hist["Close"].tail(period).mean()
            else:
                return None

        exchange = 'JSE' if symbol.endswith('.JO') else 'LSE' if symbol.endswith('.L') else 'US'
        if exchange == 'JSE':
            last_price /= 100
            bid /= 100
            ask /= 100
            prev_close = prev_close / 100 if prev_close else None
            for key in smas:
                smas[key] /= 100
            if prev_close:
                change_pct = ((last_price - prev_close) / prev_close * 100)

        highest_sma = max(smas.values())
        dip_pct = (highest_sma - last_price) / highest_sma
        spread = (ask - bid) / last_price if last_price > 0 else 0
        volume_data = hist["Volume"].tail(20)
        avg_volume = volume_data.mean()
        current_volume = int(info.get('volume', avg_volume)) or int(avg_volume)
        
        volume_multiplier = self.volume_multipliers.get(exchange, 0.8)

        window_open = (
            dip_pct >= self.dip_threshold
            and spread <= self.max_ask_spread
            and current_volume >= avg_volume * volume_multiplier
        )

        # Simple sentiment analysis (mock news-based)
        sentiment_score = self._calculate_sentiment(symbol)

        # Calculate technical indicators
        rsi = self._calculate_rsi(hist["Close"])
        bb_upper, bb_lower = self._calculate_bollinger_bands(hist["Close"])
        fib_levels = self._calculate_fibonacci_retracements(hist["Close"])
        dip_probability = self._calculate_dip_probability(dip_pct)

        if window_open:
            row = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "symbol": symbol,
                "exchange": exchange,
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

        currency = "R" if exchange == 'JSE' else "£" if exchange == 'LSE' else "$"

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
            'exchange': exchange,
            'history': hist,
            'symbol': symbol,
            'rsi': rsi,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'fib_levels': fib_levels,
            'sentiment_score': sentiment_score,
            'dip_probability': dip_probability
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss if loss != 0 else np.inf
        return 100 - (100 / (1 + rs)) if rs != np.inf else 100

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    def _calculate_fibonacci_retracements(self, prices: pd.Series) -> Dict[str, float]:
        high = prices.max()
        low = prices.min()
        diff = high - low
        return {
            '0.0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50.0%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '100.0%': low
        }

    def _calculate_sentiment(self, symbol: str) -> float:
        # Mock sentiment analysis (keyword-based, no external API)
        mock_news = {
            'NED.JO': ['strong earnings', 'growth', 'positive outlook'],
            'VOD.L': ['challenges', 'competition', 'recovery expected'],
            'AAPL': ['innovation', 'record sales', 'bullish'],
            'default': ['stable', 'market uncertainty']
        }
        words = mock_news.get(symbol, mock_news['default'])
        score = sum(1 if 'positive' in w or 'strong' in w or 'growth' in w or 'bullish' in w else -1 if 'challenge' in w or 'weak' in w else 0 for w in words)
        return min(max(score / len(words), -1), 1)

    def _calculate_dip_probability(self, dip_pct: float) -> float:
        # Simple heuristic: higher dip % = higher probability of dip
        return min(dip_pct / self.dip_threshold, 1.0)

    def _append_csv(self, row: Dict) -> None:
        try:
            df = pd.DataFrame([row])
            df.to_csv(self.csv_file, mode="a", header=False, index=False)
        except Exception:
            pass


class StockDetailsDialog(QDialog):
    def __init__(self, data: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Details for {data['symbol']} ({data['exchange']})")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()
        currency = data['currency']
        details = (
            f"Exchange: {data['exchange']}\n"
            f"Last Price: {currency}{data['last_price']:.2f}\n"
            f"Bid: {currency}{data['bid']:.2f}\n"
            f"Ask: {currency}{data['ask']:.2f}\n"
            f"Spread: {data['spread']:.2%}\n"
            f"Dip %: {data['dip_pct']:.2%}\n"
            f"Volume: {data['volume']:,}\n"
            f"Avg Volume (20d): {data['avg_volume']:,.0f}\n"
            f"RSI: {data['rsi']:.2f}\n"
            f"Sentiment Score: {data['sentiment_score']:.2f}\n"
        )
        for period, value in data['smas'].items():
            details += f"{period}: {currency}{value:.2f}\n"
        for level, price in data['fib_levels'].items():
            details += f"Fibonacci {level}: {currency}{price:.2f}\n"

        label = QLabel(details)
        layout.addWidget(label)

        # Plotly chart
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Price and Technicals", "Volume", "Dip Probability Heatmap"),
            row_heights=[0.5, 0.3, 0.2],
            vertical_spacing=0.1
        )

        hist = data['history']
        if not hist.empty:
            # Price chart with SMAs, Bollinger Bands, Fibonacci
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='blue')), row=1, col=1)
            for period, value in data['smas'].items():
                sma_series = hist['Close'].rolling(window=int(period.split('_')[1])).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=sma_series, name=period, line=dict(dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=data['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=data['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dot')), row=1, col=1)
            for level, price in data['fib_levels'].items():
                fig.add_hline(y=price, line_dash="dash", annotation_text=f"Fib {level}", row=1, col=1)
            
            # Volume chart
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume'), row=2, col=1)
            fig.add_hline(y=data['avg_volume'], line_dash="dash", line_color="red", annotation_text="Avg Volume", row=2, col=1)
            
            # Dip probability heatmap
            heatmap_data = [[data['dip_probability']]]
            fig.add_trace(go.Heatmap(z=heatmap_data, colorscale='RdYlGn', showscale=True, zmin=0, zmax=1), row=3, col=1)

            fig.update_layout(
                height=500,
                showlegend=True,
                title_text=f"Analysis for {data['symbol']} ({data['exchange']})",
                template='plotly_dark' if parent.is_dark_mode else 'plotly'
            )
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_xaxes(showticklabels=False, row=3, col=1)
            fig.update_yaxes(showticklabels=False, row=3, col=1)

        chart = QWebEngineView()
        chart.setHtml(fig.to_html(include_plotlyjs='cdn'))
        layout.addWidget(chart)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dip Watching")
        self.setGeometry(100, 100, 1000, 600)
        self.is_dark_mode = False

        self.tray = QSystemTrayIcon(QIcon.fromTheme("stock-chart"), self)
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)
        self.tray.setContextMenu(tray_menu)
        self.tray.show()

        self.settings_file = Path("settings.json")
        self.watchlist_file = Path("watchlist.json")
        self.cloud_file = Path("cloud_sync.json")
        settings = self.load_settings()
        self.watchlist: List[Dict] = self.load_watchlist()

        self.watcher = DipWatcher(
            [stock['ticker'] for stock in self.watchlist],
            dip_threshold=settings.get('dip_threshold', 0.15),
            max_ask_spread=settings.get('max_ask_spread', 0.02),
            lookback_periods=tuple(settings.get('lookback_periods', [5, 20])),
            csv_file="dip_alerts.csv",
            volume_multipliers=settings.get('volume_multipliers', {'US': 0.8, 'JSE': 0.5, 'LSE': 0.7})
        )

        columns = ['Ticker', 'Last Price', 'Change %', 'Dip %', 'Volume', 'SMA 5', 'SMA 20', 'Target Price', 'Status']
        self.table = QTableWidget(len(self.watchlist), len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSortingEnabled(True)
        self.table.setDragEnabled(True)
        self.table.setAcceptDrops(True)
        self.table.setDragDropMode(QTableWidget.DragDropMode.InternalMove)
        self.table.doubleClicked.connect(self.show_details)

        for i, stock in enumerate(self.watchlist):
            self.table.setItem(i, 0, QTableWidgetItem(stock['ticker']))
            target_item = QTableWidgetItem(str(stock['target']) if stock['target'] is not None else "")
            target_item.setFlags(Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(i, 7, target_item)

        add_btn = QPushButton("Add Ticker")
        add_btn.clicked.connect(self.add_ticker)

        import_btn = QPushButton("Import Tickers")
        import_btn.clicked.connect(self.import_tickers)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_ticker)

        refresh_btn = QPushButton("Refresh Now")
        refresh_btn.clicked.connect(self.update_data)

        export_btn = QPushButton("Export Alerts")
        export_btn.clicked.connect(self.export_alerts)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.open_settings)

        dark_mode_btn = QCheckBox("Dark Mode")
        dark_mode_btn.setChecked(self.is_dark_mode)
        dark_mode_btn.stateChanged.connect(self.toggle_dark_mode)

        sync_btn = QPushButton("Sync to Cloud")
        sync_btn.clicked.connect(self.sync_to_cloud)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        button_layout = QVBoxLayout()
        button_layout.addWidget(add_btn)
        button_layout.addWidget(import_btn)
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(export_btn)
        button_layout.addWidget(settings_btn)
        button_layout.addWidget(dark_mode_btn)
        button_layout.addWidget(sync_btn)
        layout.addLayout(button_layout)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.addAction(QAction("Refresh", self, shortcut=QKeySequence("Ctrl+R"), triggered=self.update_data))
        self.addAction(QAction("Add Ticker", self, shortcut=QKeySequence("Ctrl+A"), triggered=self.add_ticker))
        self.addAction(QAction("Import Tickers", self, shortcut=QKeySequence("Ctrl+I"), triggered=self.import_tickers))
        self.addAction(QAction("Export Alerts", self, shortcut=QKeySequence("Ctrl+E"), triggered=self.export_alerts))

        self.interval = settings.get('interval', 10) * 1000
        self.worker = DataWorker(self.watcher, self.watchlist, self.interval)
        self.worker.data_updated.connect(self.handle_data_update)
        self.worker.start()

        # Responsive design
        self.setStyleSheet(self._get_stylesheet())
        self.resizeEvent = self.handle_resize

    def _get_stylesheet(self) -> str:
        base_style = """
            QMainWindow, QDialog { font-size: 14px; }
            QTableWidget { font-size: 12px; }
            QPushButton { padding: 5px; font-size: 12px; }
            QLabel { font-size: 12px; }
        """
        if self.is_dark_mode:
            return base_style + """
                QMainWindow, QDialog { background-color: #2b2b2b; color: #ffffff; }
                QTableWidget { background-color: #333333; color: #ffffff; }
                QPushButton { background-color: #444444; color: #ffffff; }
                QCheckBox { color: #ffffff; }
            """
        return base_style + """
            QMainWindow, QDialog { background-color: #ffffff; color: #000000; }
            QTableWidget { background-color: #f0f0f0; color: #000000; }
            QPushButton { background-color: #e0e0e0; color: #000000; }
            QCheckBox { color: #000000; }
        """

    def toggle_dark_mode(self, state):
        self.is_dark_mode = state == Qt.CheckState.Checked.value
        self.setStyleSheet(self._get_stylesheet())
        self.save_settings()

    def handle_resize(self, event):
        font_size = max(10, min(14, int(self.width() / 80)))
        self.setStyleSheet(self._get_stylesheet().replace('14px', f'{font_size}px').replace('12px', f'{font_size-2}px'))
        super().resizeEvent(event)

    def load_settings(self) -> Dict:
        if self.settings_file.exists():
            try:
                with self.settings_file.open('r') as f:
                    settings = json.load(f)
                    return {
                        'dip_threshold': float(settings.get('dip_threshold', 0.15)),
                        'max_ask_spread': float(settings.get('max_ask_spread', 0.02)),
                        'lookback_periods': [int(p) for p in settings.get('lookback_periods', [5, 20])],
                        'interval': int(settings.get('interval', 10)),
                        'volume_multipliers': {
                            k: float(v) for k, v in settings.get('volume_multipliers', {'US': 0.8, 'JSE': 0.5, 'LSE': 0.7}).items()
                        },
                        'dark_mode': bool(settings.get('dark_mode', False))
                    }
            except Exception:
                pass
        return {'dip_threshold': 0.15, 'max_ask_spread': 0.02, 'lookback_periods': [5, 20], 'interval': 10, 'volume_multipliers': {'US': 0.8, 'JSE': 0.5, 'LSE': 0.7}, 'dark_mode': False}

    def save_settings(self) -> None:
        try:
            with self.settings_file.open('w') as f:
                json.dump({
                    'dip_threshold': self.watcher.dip_threshold,
                    'max_ask_spread': self.watcher.max_ask_spread,
                    'lookback_periods': list(self.watcher.lookback_periods),
                    'interval': self.interval // 1000,
                    'volume_multipliers': self.watcher.volume_multipliers,
                    'dark_mode': self.is_dark_mode
                }, f, indent=2)
        except Exception:
            pass

    def load_watchlist(self) -> List[Dict]:
        # Try cloud sync first
        if self.cloud_file.exists():
            try:
                with self.cloud_file.open('r') as f:
                    data = json.load(f)
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
                    self.watchlist_file.write_text(json.dumps(data, indent=2))
                    return watchlist[:25]
            except Exception:
                pass
        # Fallback to local watchlist
        if self.watchlist_file.exists():
            try:
                with self.watchlist_file.open('r') as f:
                    data = json.load(f)
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
                    return watchlist[:25]
            except Exception:
                return []
        return []

    def save_watchlist(self) -> None:
        try:
            with self.watchlist_file.open('w') as f:
                json.dump([
                    {'ticker': stock['ticker'], 'target': stock['target']}
                    for stock in self.watchlist
                ], f, indent=2)
            self.sync_to_cloud()
        except Exception:
            pass

    def sync_to_cloud(self):
        try:
            with self.cloud_file.open('w') as f:
                json.dump([
                    {'ticker': stock['ticker'], 'target': stock['target']}
                    for stock in self.watchlist
                ], f, indent=2)
            self.save_settings()
            QMessageBox.information(self, "Cloud Sync", "Watchlist and settings synced to cloud.")
        except Exception:
            QMessageBox.critical(self, "Sync Error", "Failed to sync to cloud.")

    def add_ticker(self):
        if len(self.watchlist) >= 25:
            QMessageBox.warning(self, "Limit Reached", "You can add up to 25 stocks to the watchlist.")
            return

        ticker, ok = QInputDialog.getText(self, "Add Ticker", "Enter ticker symbol (e.g., VOD.L for LSE, NED for JSE):")
        if ok and ticker:
            formatted = self.watcher._format_ticker(ticker)
            if not self.watcher.validate_ticker(formatted):
                QMessageBox.critical(self, "Invalid Ticker", f"Ticker {formatted} is invalid or has no data.")
                return
            if formatted not in [s['ticker'] for s in self.watchlist]:
                self.watchlist.append({'ticker': formatted, 'target': None, 'notified_target': False, 'notified_dip': False})
                self.watcher.tickers.append(formatted)
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(formatted))
                target_item = QTableWidgetItem()
                target_item.setFlags(Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, 7, target_item)
                self.save_watchlist()

    def import_tickers(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Tickers", "", "CSV Files (*.csv);;Text Files (*.txt)")
        if file_path:
            try:
                tickers = []
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if 'ticker' in df.columns:
                        tickers = df['ticker'].dropna().tolist()
                else:
                    with open(file_path, 'r') as f:
                        tickers = [line.strip() for line in f if line.strip()]
                
                added = 0
                for ticker in tickers:
                    if len(self.watchlist) >= 25:
                        break
                    formatted = self.watcher._format_ticker(ticker)
                    if not self.watcher.validate_ticker(formatted):
                        continue
                    if formatted not in [s['ticker'] for s in self.watchlist]:
                        self.watchlist.append({'ticker': formatted, 'target': None, 'notified_target': False, 'notified_dip': False})
                        self.watcher.tickers.append(formatted)
                        row = self.table.rowCount()
                        self.table.insertRow(row)
                        self.table.setItem(row, 0, QTableWidgetItem(formatted))
                        target_item = QTableWidgetItem()
                        target_item.setFlags(Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled)
                        self.table.setItem(row, 7, target_item)
                        added += 1
                self.save_watchlist()
                QMessageBox.information(self, "Import Complete", f"Added {added} valid tickers to the watchlist.")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import tickers: {str(e)}")

    def remove_ticker(self):
        row = self.table.currentRow()
        if row >= 0:
            ticker = self.watchlist[row]['ticker']
            del self.watchlist[row]
            self.watcher.tickers.remove(ticker)
            self.table.removeRow(row)
            self.save_watchlist()

    def export_alerts(self):
        if not self.watcher.csv_file.exists():
            QMessageBox.warning(self, "No Alerts", "No dip alerts available to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Alerts", "", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(self.watcher.csv_file)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Complete", f"Alerts exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export alerts: {str(e)}")

    def show_details(self, index):
        row = index.row()
        if row >= 0:
            stock = self.watchlist[row]
            data = self.watcher.get_stock_data(stock['ticker'])
            if data:
                dialog = StockDetailsDialog(data, self)
                dialog.exec()

    def handle_data_update(self, results: List[Tuple[Dict, Optional[Dict]]]):
        for i, (stock, data) in enumerate(results):
            if data is None:
                self.table.setItem(i, 8, QTableWidgetItem("Error"))
                continue

            self.table.setItem(i, 1, QTableWidgetItem(f"{data['currency']}{data['last_price']:.2f}"))
            change_item = QTableWidgetItem(f"{data['change_pct']:.2f}%")
            if data['change_pct'] > 0:
                change_item.setForeground(QColor("green"))
            elif data['change_pct'] < 0:
                change_item.setForeground(QColor("red"))
            self.table.setItem(i, 2, change_item)

            dip_item = QTableWidgetItem(f"{data['dip_pct']:.2%}")
            if data['dip_pct'] >= self.watcher.dip_threshold:
                dip_item.setForeground(QColor("red"))
            elif data['dip_pct'] >= self.watcher.dip_threshold * 0.5:
                dip_item.setForeground(QColor("orange"))
            self.table.setItem(i, 3, dip_item)

            self.table.setItem(i, 4, QTableWidgetItem(f"{data['volume']:,}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{data['currency']}{data['smas'].get('SMA_5', 0):.2f}"))
            self.table.setItem(i, 6, QTableWidgetItem(f"{data['currency']}{data['smas'].get('SMA_20', 0):.2f}"))

            target_item = self.table.item(i, 7)
            if target_item and target_item.text():
                try:
                    new_target = float(target_item.text())
                    if stock['target'] != new_target:
                        stock['target'] = new_target
                        stock['notified_target'] = False
                        self.save_watchlist()
                except ValueError:
                    stock['target'] = None
                    self.save_watchlist()

            status = ""
            status_item = QTableWidgetItem(status)
            if data['window_open']:
                status = "Dip Alert!"
                status_item.setForeground(QColor("red"))
                if not stock['notified_dip']:
                    self.tray.showMessage("Dip Alert", f"{stock['ticker']} has a dip entry window open!", QSystemTrayIcon.MessageIcon.Information, 5000)
                    stock['notified_dip'] = True

            if stock['target'] is not None and data['last_price'] <= stock['target']:
                if not stock['notified_target']:
                    self.tray.showMessage("Target Reached", f"{stock['ticker']} reached target price {data['currency']}{stock['target']:.2f}!", QSystemTrayIcon.MessageIcon.Information, 5000)
                    stock['notified_target'] = True

            self.table.setItem(i, 8, status_item)

    def update_data(self):
        results = [(stock, self.watcher.get_stock_data(stock['ticker'])) for stock in self.watchlist]
        self.handle_data_update(results)

    def open_settings(self):
        dip_th, ok = QInputDialog.getDouble(self, "Dip Threshold", "Enter dip threshold (0-1):", self.watcher.dip_threshold, 0, 1, decimals=2)
        if ok:
            self.watcher.dip_threshold = dip_th
            self.save_settings()

        max_sp, ok = QInputDialog.getDouble(self, "Max Ask Spread", "Enter max ask spread (0-1):", self.watcher.max_ask_spread, 0, 1, decimals=2)
        if ok:
            self.watcher.max_ask_spread = max_sp
            self.save_settings()

        lookback_str, ok = QInputDialog.getText(self, "Lookback Periods", "Enter lookback periods (comma separated):", text=",".join(map(str, self.watcher.lookback_periods)))
        if ok:
            try:
                self.watcher.lookback_periods = tuple(map(int, lookback_str.split(',')))
                self.save_settings()
            except ValueError:
                pass

        interval, ok = QInputDialog.getInt(self, "Interval (seconds)", "Enter poll interval:", self.interval // 1000, 5, 3600)
        if ok:
            self.interval = interval * 1000
            self.worker.interval = interval
            self.save_settings()

        us_mult, ok = QInputDialog.getDouble(self, "US Volume Multiplier", "Enter US volume multiplier:", self.watcher.volume_multipliers.get('US', 0.8), 0, 2, decimals=2)
        if ok:
            self.watcher.volume_multipliers['US'] = us_mult
            self.save_settings()

        jse_mult, ok = QInputDialog.getDouble(self, "JSE Volume Multiplier", "Enter JSE volume multiplier:", self.watcher.volume_multipliers.get('JSE', 0.5), 0, 2, decimals=2)
        if ok:
            self.watcher.volume_multipliers['JSE'] = jse_mult
            self.save_settings()

        lse_mult, ok = QInputDialog.getDouble(self, "LSE Volume Multiplier", "Enter LSE volume multiplier:", self.watcher.volume_multipliers.get('LSE', 0.7), 0, 2, decimals=2)
        if ok:
            self.watcher.volume_multipliers['LSE'] = lse_mult
            self.save_settings()

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray.showMessage("Dip Watching", "Minimized to system tray.", QSystemTrayIcon.MessageIcon.Information, 2000)

    def __del__(self):
        if hasattr(self, 'worker'):
            self.worker.stop()


if __name__ == "__main__":
    os.environ['QT_API'] = 'pyqt6'
    platforms = ['xcb', 'wayland']
    app = None
    for platform in platforms:
        os.environ['QT_QPA_PLATFORM'] = platform
        try:
            app = QApplication(sys.argv)
            break
        except:
            continue
    if not app:
        print("Error: Could not initialize Qt platform. Install libxcb-cursor0: 'sudo apt install libxcb-cursor0 libx11-xcb1'")
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
