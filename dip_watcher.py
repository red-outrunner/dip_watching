#!/usr/bin/env python3
"""
dip_watcher.py
Continuously monitor a list of tickers for a configurable "dip entry window" via GUI.

Requirements:
    pip install yfinance pandas PyQt6 PyQt6-WebEngine plotly

Usage:
    python dip_watcher.py

On Pop!_OS/KDE, ensure dependencies:
    sudo apt install libxcb-cursor0 libx11-xcb1 libxcb1 libxcb-xkb1 libxkbcommon-x11-0 xwayland libqt6webenginecore6 libqt6webenginewidgets6 qt6-webengine-dev

Ensure dips.png is in the project directory for the system tray and app icon.
"""

from __future__ import annotations

import sys
import time
import json
import os
import logging
from logging.handlers import RotatingFileHandler
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
    QFileDialog, QDialogButtonBox, QLabel, QCheckBox, QTabWidget, QTextEdit,
    QLineEdit, QHBoxLayout
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QAction, QTextCursor
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Setup logging with rotation
log_file = 'dip_watcher.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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
                try:
                    data = self.watcher.get_stock_data(stock['ticker'])
                    results.append((stock, data))
                except Exception as e:
                    logger.error(f"Error fetching data for {stock['ticker']}: {str(e)}")
                    results.append((stock, None))
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
        rsi_period: int = 14,
        csv_file: str = "dip_alerts.csv",
        volume_multipliers: Dict[str, float] = None,
    ):
        self.tickers = [self._format_ticker(t) for t in tickers]
        self.dip_threshold = dip_threshold
        self.max_ask_spread = max_ask_spread
        self.lookback_periods = lookback_periods
        self.rsi_period = rsi_period
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
        try:
            if not self.csv_file.exists():
                header = (
                    "timestamp,symbol,exchange,last_price,bid,ask,spread,dip_pct,"
                    + ",".join([f"SMA_{p}" for p in self.lookback_periods])
                    + ",volume,avg_volume\n"
                )
                self.csv_file.write_text(header)
        except Exception as e:
            logger.error(f"Failed to initialize CSV file {self.csv_file}: {str(e)}")

    def validate_ticker(self, ticker: str) -> bool:
        try:
            tk = yf.Ticker(self._format_ticker(ticker))
            info = tk.info
            return bool(info.get('currentPrice') or info.get('regularMarketPrice'))
        except Exception as e:
            logger.error(f"Failed to validate ticker {ticker}: {str(e)}")
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
            except Exception as e:
                logger.error(f"Failed to read cache for {symbol}: {str(e)}")
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
                if hist.empty or len(hist) < max(self.lookback_periods) or len(hist) < self.rsi_period:
                    raise ValueError("Insufficient historical data")

                # Serialize hist with JSON-safe split orient
                hist_json = hist.to_json(orient='split')

                cache_data = {
                    'info': info,
                    'history': hist_json,
                    'timestamp': datetime.utcnow().isoformat()
                }
                try:
                    with cache_file.open('w') as f:
                        json.dump(cache_data, f)
                except Exception as e:
                    logger.error(f"Failed to write cache for {symbol}: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                if cache_file.exists():
                    try:
                        with cache_file.open('r') as f:
                            cache_data = json.load(f)
                            cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01T00:00:00'))
                            if datetime.utcnow() - cache_time >= timedelta(hours=24):
                                cache_file.unlink(missing_ok=True)
                                return None
                    except Exception as e:
                        logger.error(f"Failed to read fallback cache for {symbol}: {str(e)}")
                        cache_file.unlink(missing_ok=True)
                        return None
                else:
                    return None
        else:
            try:
                info = cache_data.get('info', {})
                hist_json = cache_data.get('history', '{}')
                hist = pd.read_json(hist_json, orient='split')
                if hist.empty or len(hist) < max(self.lookback_periods) or len(hist) < self.rsi_period:
                    logger.error(f"Invalid cache data for {symbol}: insufficient history")
                    return None
                last_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if not last_price:
                    logger.error(f"No price data in cache for {symbol}")
                    return None
                last_price = float(last_price)
                bid = float(info.get('bid', last_price * 0.999))
                ask = float(info.get('ask', last_price * 1.001))
                prev_close = info.get('previousClose')
                change_pct = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0
            except Exception as e:
                logger.error(f"Failed to process cache for {symbol}: {str(e)}")
                return None

        smas = {}
        for period in self.lookback_periods:
            if len(hist) >= period:
                smas[f"SMA_{period}"] = hist["Close"].tail(period).mean()
            else:
                logger.error(f"Insufficient data for SMA_{period} for {symbol}")
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

        # Calculate technical indicators
        try:
            rsi = self._calculate_rsi(hist["Close"])
            bb_upper, bb_lower = self._calculate_bollinger_bands(hist["Close"])
            fib_levels = self._calculate_fibonacci_retracements(hist["Close"])
            sentiment_score = self._calculate_sentiment(symbol)
            dip_probability = self._calculate_dip_probability(dip_pct)
        except Exception as e:
            logger.error(f"Failed to calculate indicators for {symbol}: {str(e)}")
            return None

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

    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> float:
        if period is None:
            period = self.rsi_period
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            avg_gain = gain.iloc[-1] if not gain.empty else 0
            avg_loss = loss.iloc[-1] if not loss.empty else 0
            if avg_loss == 0:
                rs = np.inf
            else:
                rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs)) if rs != np.inf else 100
        except Exception as e:
            logger.error(f"Failed to calculate RSI with period {period}: {str(e)}")
            return 0.0

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray]:
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, lower
        except Exception as e:
            logger.error(f"Failed to calculate Bollinger Bands: {str(e)}")
            return np.array([]), np.array([])

    def _calculate_fibonacci_retracements(self, prices: pd.Series) -> Dict[str, float]:
        try:
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
        except Exception as e:
            logger.error(f"Failed to calculate Fibonacci retracements: {str(e)}")
            return {}

    def _calculate_sentiment(self, symbol: str) -> float:
        try:
            mock_news = {
                'NED.JO': ['strong earnings', 'growth', 'positive outlook'],
                'VOD.L': ['challenges', 'competition', 'recovery expected'],
                'AAPL': ['innovation', 'record sales', 'bullish'],
                'default': ['stable', 'market uncertainty']
            }
            words = mock_news.get(symbol, mock_news['default'])
            score = sum(1 if 'positive' in w or 'strong' in w or 'growth' in w or 'bullish' in w else -1 if 'challenge' in w or 'weak' in w else 0 for w in words)
            return min(max(score / len(words), -1), 1)
        except Exception as e:
            logger.error(f"Failed to calculate sentiment for {symbol}: {str(e)}")
            return 0.0

    def _calculate_dip_probability(self, dip_pct: float) -> float:
        try:
            return min(dip_pct / self.dip_threshold, 1.0)
        except Exception as e:
            logger.error(f"Failed to calculate dip probability: {str(e)}")
            return 0.0

    def _append_csv(self, row: Dict) -> None:
        try:
            df = pd.DataFrame([row])
            df.to_csv(self.csv_file, mode="a", header=False, index=False)
        except Exception as e:
            logger.error(f"Failed to append to CSV {self.csv_file}: {str(e)}")


class StockDetailsDialog(QDialog):
    def __init__(self, data: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Details for {data['symbol']} ({data['exchange']})")
        self.setGeometry(200, 200, 800, 600)

        main_layout = QHBoxLayout()

        # Left column: Text details and checkboxes
        left_layout = QVBoxLayout()
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
            f"RSI (Period {parent.watcher.rsi_period}): {data['rsi']:.2f}\n"
            f"Sentiment Score: {data['sentiment_score']:.2f}\n"
        )
        for period, value in data['smas'].items():
            details += f"{period}: {currency}{value:.2f}\n"
        for level, price in data['fib_levels'].items():
            details += f"Fibonacci {level}: {currency}{price:.2f}\n"

        label = QLabel(details)
        left_layout.addWidget(label)

        # Indicator toggle checkboxes
        indicator_layout = QHBoxLayout()
        self.rsi_check = QCheckBox("RSI")
        self.rsi_check.setChecked(True)
        self.rsi_check.stateChanged.connect(self.update_indicators)
        indicator_layout.addWidget(self.rsi_check)

        self.macd_check = QCheckBox("MACD")
        self.macd_check.setChecked(True)
        self.macd_check.stateChanged.connect(self.update_indicators)
        indicator_layout.addWidget(self.macd_check)

        self.vwap_check = QCheckBox("VWAP")
        self.vwap_check.setChecked(True)
        self.vwap_check.stateChanged.connect(self.update_indicators)
        indicator_layout.addWidget(self.vwap_check)

        left_layout.addLayout(indicator_layout)
        main_layout.addLayout(left_layout)

        # Right column: Plotly chart
        self.chart = QWebEngineView()
        self.update_chart()
        main_layout.addWidget(self.chart)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.setLayout(main_layout)

    def update_indicators(self):
        try:
            self.parent.watcher.indicators['rsi'] = self.rsi_check.isChecked()
            self.parent.watcher.indicators['macd'] = self.macd_check.isChecked()
            self.parent.watcher.indicators['vwap'] = self.vwap_check.isChecked()
            self.parent.save_settings()
            logger.info(f"Updated indicators: {self.parent.watcher.indicators}")

            # Update data with new indicator settings
            self.data = self.parent.watcher.get_stock_data(self.data['symbol'])
            currency = self.data['currency']
            details = (
                f"Exchange: {self.data['exchange']}\n"
                f"Last Price: {currency}{self.data['last_price']:.2f}\n"
                f"Bid: {currency}{self.data['bid']:.2f}\n"
                f"Ask: {currency}{self.data['ask']:.2f}\n"
                f"Spread: {self.data['spread']:.2%}\n"
                f"Dip %: {self.data['dip_pct']:.2%}\n"
                f"Volume: {self.data['volume']:,}\n"
                f"Avg Volume (20d): {self.data['avg_volume']:,.0f}\n"
            )
            if self.parent.watcher.indicators['rsi'] and self.data['rsi'] is not None:
                details += f"RSI (Period {self.parent.watcher.rsi_period}): {self.data['rsi']:.2f}\n"
            if self.parent.watcher.indicators['macd'] and self.data['macd'] is not None:
                details += f"MACD: {self.data['macd'].iloc[-1]:.2f} (Signal: {self.data['macd_signal'].iloc[-1]:.2f})\n"
            if self.parent.watcher.indicators['vwap'] and self.data['vwap'] is not None:
                details += f"VWAP: {currency}{self.data['vwap'].iloc[-1] / (100 if self.data['exchange'] == 'JSE' else 1):.2f}\n"
            details += f"Sentiment Score: {self.data['sentiment_score']:.2f}\n"
            for period, value in self.data['smas'].items():
                details += f"{period}: {currency}{value:.2f}\n"
            for level, price in self.data['fib_levels'].items():
                details += f"Fibonacci {level}: {currency}{price:.2f}\n"

            self.layout().itemAt(0).layout().itemAt(0).widget().setText(details)
            self.update_chart()
        except Exception as e:
            logger.error(f"Failed to update indicators: {str(e)}")

    def update_chart(self):
        try:
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Price and Technicals", "Volume", "Dip Probability Heatmap", "MACD" if self.parent.watcher.indicators['macd'] else ""),
                row_heights=[0.4, 0.2, 0.1, 0.3],
                vertical_spacing=0.1
            )

            hist = self.data['history']
            if not hist.empty:
                # Adjust history for JSE if needed
                close_prices = hist['Close']
                if self.data['exchange'] == 'JSE':
                    close_prices = close_prices / 100
                # Price chart with SMAs, Bollinger Bands, Fibonacci, VWAP
                fig.add_trace(go.Scatter(x=hist.index, y=close_prices, name='Price', line=dict(color='blue')), row=1, col=1)
                for period, value in self.data['smas'].items():
                    sma_series = hist['Close'].rolling(window=int(period.split('_')[1])).mean()
                    if self.data['exchange'] == 'JSE':
                        sma_series = sma_series / 100
                    fig.add_trace(go.Scatter(x=hist.index, y=sma_series, name=period, line=dict(dash='dash')), row=1, col=1)
                bb_upper = self.data['bb_upper']
                bb_lower = self.data['bb_lower']
                if self.data['exchange'] == 'JSE':
                    bb_upper = bb_upper / 100
                    bb_lower = bb_lower / 100
                fig.add_trace(go.Scatter(x=hist.index, y=bb_upper, name='BB Upper', line=dict(color='gray', dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=bb_lower, name='BB Lower', line=dict(color='gray', dash='dot')), row=1, col=1)
                for level, price in self.data['fib_levels'].items():
                    fib_price = price / 100 if self.data['exchange'] == 'JSE' else price
                    fig.add_hline(y=fib_price, line_dash="dash", annotation_text=f"Fib {level}", row=1, col=1)
                if self.parent.watcher.indicators['vwap'] and self.data['vwap'] is not None:
                    vwap_series = self.data['vwap']
                    if self.data['exchange'] == 'JSE':
                        vwap_series = vwap_series / 100
                    fig.add_trace(go.Scatter(x=hist.index, y=vwap_series, name='VWAP', line=dict(color='purple', dash='dot')), row=1, col=1)
                
                # Volume chart
                fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume'), row=2, col=1)
                fig.add_hline(y=self.data['avg_volume'], line_dash="dash", line_color="red", annotation_text="Avg Volume", row=2, col=1)
                
                # Dip probability heatmap
                heatmap_data = [[self.data['dip_probability']]]
                fig.add_trace(go.Heatmap(z=heatmap_data, colorscale='RdYlGn', showscale=True, zmin=0, zmax=1), row=3, col=1)

                # MACD chart
                if self.parent.watcher.indicators['macd'] and self.data['macd'] is not None:
                    fig.add_trace(go.Scatter(x=hist.index, y=self.data['macd'], name='MACD', line=dict(color='blue')), row=4, col=1)
                    fig.add_trace(go.Scatter(x=hist.index, y=self.data['macd_signal'], name='Signal', line=dict(color='orange')), row=4, col=1)
                    fig.add_trace(go.Bar(x=hist.index, y=self.data['macd_hist'], name='Histogram'), row=4, col=1)

                fig.update_layout(
                    height=600,
                    showlegend=True,
                    title_text=f"Analysis for {self.data['symbol']} ({self.data['exchange']})",
                    template='plotly_dark' if self.parent.is_dark_mode else 'plotly',
                    legend=dict(
                        orientation='v',
                        yanchor='top',
                        y=1,
                        xanchor='left',
                        x=1.02,
                        bgcolor='rgba(0,0,0,0)',
                        font=dict(size=10)
                    )
                )
                fig.update_xaxes(title_text="Date", row=1, col=1)
                fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                fig.update_xaxes(showticklabels=False, row=3, col=1)
                fig.update_yaxes(showticklabels=False, row=3, col=1)
                if self.parent.watcher.indicators['macd']:
                    fig.update_xaxes(title_text="Date", row=4, col=1)
                    fig.update_yaxes(title_text="MACD", row=4, col=1)

            self.chart.setHtml(fig.to_html(include_plotlyjs='cdn'))
        except Exception as e:
            logger.error(f"Failed to render Plotly chart for {self.data['symbol']}: {str(e)}")
            self.chart.setHtml("<p>Error rendering chart. Check dip_watcher.log for details.</p>")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dip Watching")
        self.setGeometry(100, 100, 1000, 600)
        self.is_dark_mode = False

        # Use custom icon for app and tray
        icon_path = "dips.png"
        icon = QIcon(icon_path)
        if icon.isNull():
            logger.error(f"Failed to load icon {icon_path}")
        self.setWindowIcon(icon)
        self.tray = QSystemTrayIcon(icon, self)
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)
        self.tray.setContextMenu(tray_menu)
        self.tray.show()
        logger.info("System tray and app initialized with dips.png")

        self.settings_file = Path("settings.json")
        self.watchlist_file = Path("watchlist.json")
        self.cloud_file = Path("cloud_sync.json")
        settings = self.load_settings()
        self.is_dark_mode = settings.get('dark_mode', False)

        self.watcher = DipWatcher(
            [stock['ticker'] for stock in self.load_watchlist()],
            dip_threshold=settings.get('dip_threshold', 0.15),
            max_ask_spread=settings.get('max_ask_spread', 0.02),
            lookback_periods=tuple(settings.get('lookback_periods', [5, 20])),
            rsi_period=settings.get('rsi_period', 14),
            csv_file="dip_alerts.csv",
            volume_multipliers=settings.get('volume_multipliers', {'US': 0.8, 'JSE': 0.5, 'LSE': 0.7})
        )
        self.watchlist: List[Dict] = self.load_watchlist()

        # Tab widget for Watchlist and Logs
        self.tabs = QTabWidget()
        self.watchlist_widget = QWidget()
        self.logs_widget = QWidget()

        # Watchlist tab
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

        # Logs tab
        logs_layout = QVBoxLayout()
        self.log_filter = QLineEdit()
        self.log_filter.setPlaceholderText("Filter logs (e.g., ERROR, NED.JO)")
        self.log_filter.textChanged.connect(self.update_logs)
        logs_layout.addWidget(self.log_filter)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        logs_layout.addWidget(self.log_display)
        self.logs_widget.setLayout(logs_layout)

        # Initialize logs
        self.update_logs()
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_logs)
        self.log_timer.start(5000)  # Update every 5 seconds

        watchlist_layout = QVBoxLayout()
        watchlist_layout.addWidget(self.table)
        self.watchlist_widget.setLayout(watchlist_layout)

        self.tabs.addTab(self.watchlist_widget, "Watchlist")
        self.tabs.addTab(self.logs_widget, "Logs")

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
        layout.addWidget(self.tabs)
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
        logger.info("DataWorker thread started")

        # Responsive design
        self.setStyleSheet(self._get_stylesheet())
        self.resizeEvent = self.handle_resize

    def _get_stylesheet(self) -> str:
        base_style = """
            QMainWindow, QDialog { font-size: 14px; }
            QTableWidget, QTextEdit, QLineEdit { font-size: 12px; }
            QPushButton { padding: 5px; font-size: 12px; }
            QLabel, QCheckBox { font-size: 12px; }
            QTabWidget::pane { border: 1px solid #888888; }
        """
        if self.is_dark_mode:
            return base_style + """
                QMainWindow, QDialog { background-color: #2b2b2b; color: #ffffff; }
                QTableWidget, QTextEdit, QLineEdit { background-color: #333333; color: #ffffff; }
                QPushButton { background-color: #444444; color: #ffffff; }
                QCheckBox { color: #ffffff; }
                QTabWidget::pane { border-color: #555555; }
                QTabBar::tab { background: #333333; color: #ffffff; }
                QTabBar::tab:selected { background: #555555; }
            """
        return base_style + """
            QMainWindow, QDialog { background-color: #ffffff; color: #000000; }
            QTableWidget, QTextEdit, QLineEdit { background-color: #f0f0f0; color: #000000; }
            QPushButton { background-color: #e0e0e0; color: #000000; }
            QCheckBox { color: #000000; }
            QTabWidget::pane { border-color: #cccccc; }
            QTabBar::tab { background: #e0e0e0; color: #000000; }
            QTabBar::tab:selected { background: #cccccc; }
        """

    def toggle_dark_mode(self, state):
        self.is_dark_mode = state == Qt.CheckState.Checked.value
        self.setStyleSheet(self._get_stylesheet())
        self.save_settings()
        logger.info(f"Dark mode set to {self.is_dark_mode}")

    def handle_resize(self, event):
        font_size = max(10, min(14, int(self.width() / 80)))
        self.setStyleSheet(self._get_stylesheet().replace('14px', f'{font_size}px').replace('12px', f'{font_size-2}px'))
        super().resizeEvent(event)

    def load_settings(self) -> Dict:
        try:
            if self.settings_file.exists():
                with self.settings_file.open('r') as f:
                    settings = json.load(f)
                    return {
                        'dip_threshold': float(settings.get('dip_threshold', 0.15)),
                        'max_ask_spread': float(settings.get('max_ask_spread', 0.02)),
                        'lookback_periods': [int(p) for p in settings.get('lookback_periods', [5, 20])],
                        'rsi_period': int(settings.get('rsi_period', 14)),
                        'interval': int(settings.get('interval', 10)),
                        'volume_multipliers': {
                            k: float(v) for k, v in settings.get('volume_multipliers', {'US': 0.8, 'JSE': 0.5, 'LSE': 0.7}).items()
                        },
                        'dark_mode': bool(settings.get('dark_mode', False))
                    }
            return {
                'dip_threshold': 0.15,
                'max_ask_spread': 0.02,
                'lookback_periods': [5, 20],
                'rsi_period': 14,
                'interval': 10,
                'volume_multipliers': {'US': 0.8, 'JSE': 0.5, 'LSE': 0.7},
                'dark_mode': False
            }
        except Exception as e:
            logger.error(f"Failed to load settings: {str(e)}")
            return {
                'dip_threshold': 0.15,
                'max_ask_spread': 0.02,
                'lookback_periods': [5, 20],
                'rsi_period': 14,
                'interval': 10,
                'volume_multipliers': {'US': 0.8, 'JSE': 0.5, 'LSE': 0.7},
                'dark_mode': False
            }

    def save_settings(self) -> None:
        try:
            with self.settings_file.open('w') as f:
                json.dump({
                    'dip_threshold': self.watcher.dip_threshold,
                    'max_ask_spread': self.watcher.max_ask_spread,
                    'lookback_periods': list(self.watcher.lookback_periods),
                    'rsi_period': self.watcher.rsi_period,
                    'interval': self.interval // 1000,
                    'volume_multipliers': self.watcher.volume_multipliers,
                    'dark_mode': self.is_dark_mode
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save settings: {str(e)}")

    def update_logs(self):
        try:
            log_file = Path('dip_watcher.log')
            filter_text = self.log_filter.text().lower()
            if log_file.exists():
                with log_file.open('r') as f:
                    lines = f.readlines()[-1000:]  # Last 1000 lines
                    if filter_text:
                        lines = [line for line in lines if filter_text in line.lower()]
                    self.log_display.setText(''.join(lines))
                    cursor = self.log_display.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    self.log_display.setTextCursor(cursor)
            else:
                self.log_display.setText("No logs available.")
        except Exception as e:
            logger.error(f"Failed to update logs: {str(e)}")
            self.log_display.setText("Error reading logs. Check dip_watcher.log for details.")

    def load_watchlist(self) -> List[Dict]:
        try:
            if self.cloud_file.exists():
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
                    logger.info("Loaded watchlist from cloud_sync.json")
                    return watchlist[:25]
        except Exception as e:
            logger.error(f"Failed to load watchlist from cloud: {str(e)}")

        try:
            if self.watchlist_file.exists():
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
                    logger.info("Loaded watchlist from watchlist.json")
                    return watchlist[:25]
            return []
        except Exception as e:
            logger.error(f"Failed to load watchlist: {str(e)}")
            return []

    def save_watchlist(self) -> None:
        try:
            with self.watchlist_file.open('w') as f:
                json.dump([
                    {'ticker': stock['ticker'], 'target': stock['target']}
                    for stock in self.watchlist
                ], f, indent=2)
            self.sync_to_cloud()
            logger.info("Saved watchlist to watchlist.json")
        except Exception as e:
            logger.error(f"Failed to save watchlist: {str(e)}")

    def sync_to_cloud(self):
        try:
            with self.cloud_file.open('w') as f:
                json.dump([
                    {'ticker': stock['ticker'], 'target': stock['target']}
                    for stock in self.watchlist
                ], f, indent=2)
            self.save_settings()
            QMessageBox.information(self, "Cloud Sync", "Watchlist and settings synced to cloud.")
            logger.info("Synced watchlist and settings to cloud_sync.json")
        except Exception as e:
            logger.error(f"Failed to sync to cloud: {str(e)}")
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
                logger.warning(f"Attempted to add invalid ticker: {formatted}")
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
                logger.info(f"Added ticker {formatted} to watchlist")

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
                        logger.warning(f"Skipped invalid ticker during import: {formatted}")
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
                logger.info(f"Imported {added} tickers from {file_path}")
            except Exception as e:
                logger.error(f"Failed to import tickers from {file_path}: {str(e)}")
                QMessageBox.critical(self, "Import Error", f"Failed to import tickers: {str(e)}")

    def remove_ticker(self):
        row = self.table.currentRow()
        if row >= 0:
            ticker = self.watchlist[row]['ticker']
            del self.watchlist[row]
            self.watcher.tickers.remove(ticker)
            self.table.removeRow(row)
            self.save_watchlist()
            logger.info(f"Removed ticker {ticker} from watchlist")

    def export_alerts(self):
        if not self.watcher.csv_file.exists():
            QMessageBox.warning(self, "No Alerts", "No dip alerts available to export.")
            logger.warning("Attempted to export alerts but no alerts available")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Alerts", "", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(self.watcher.csv_file)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Complete", f"Alerts exported to {file_path}")
                logger.info(f"Exported alerts to {file_path}")
            except Exception as e:
                logger.error(f"Failed to export alerts to {file_path}: {str(e)}")
                QMessageBox.critical(self, "Export Error", f"Failed to export alerts: {str(e)}")

    def show_details(self, index):
        row = index.row()
        if row >= 0:
            stock = self.watchlist[row]
            try:
                data = self.watcher.get_stock_data(stock['ticker'])
                if data:
                    dialog = StockDetailsDialog(data, self)
                    dialog.exec()
                else:
                    logger.warning(f"No data available for {stock['ticker']} in details dialog")
            except Exception as e:
                logger.error(f"Failed to show details for {stock['ticker']}: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load details for {stock['ticker']}.")

    def handle_data_update(self, results: List[Tuple[Dict, Optional[Dict]]]):
        for i, (stock, data) in enumerate(results):
            if data is None:
                self.table.setItem(i, 8, QTableWidgetItem("Error"))
                logger.warning(f"No data for {stock['ticker']} in update")
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
                        logger.info(f"Updated target price for {stock['ticker']} to {new_target}")
                except ValueError:
                    stock['target'] = None
                    self.save_watchlist()
                    logger.warning(f"Invalid target price entered for {stock['ticker']}")

            status = ""
            status_item = QTableWidgetItem(status)
            if data['window_open']:
                status = "Dip Alert!"
                status_item.setForeground(QColor("red"))
                if not stock['notified_dip']:
                    try:
                        self.tray.showMessage("Dip Alert", f"{stock['ticker']} has a dip entry window open!", QSystemTrayIcon.MessageIcon.Information, 5000)
                        stock['notified_dip'] = True
                        logger.info(f"Dip alert triggered for {stock['ticker']}")
                    except Exception as e:
                        logger.error(f"Failed to show dip notification for {stock['ticker']}: {str(e)}")

            if stock['target'] is not None and data['last_price'] <= stock['target']:
                if not stock['notified_target']:
                    try:
                        self.tray.showMessage("Target Reached", f"{stock['ticker']} reached target price {data['currency']}{stock['target']:.2f}!", QSystemTrayIcon.MessageIcon.Information, 5000)
                        stock['notified_target'] = True
                        logger.info(f"Target price alert triggered for {stock['ticker']}")
                    except Exception as e:
                        logger.error(f"Failed to show target notification for {stock['ticker']}: {str(e)}")

            self.table.setItem(i, 8, status_item)

    def update_data(self):
        try:
            results = [(stock, self.watcher.get_stock_data(stock['ticker'])) for stock in self.watchlist]
            self.handle_data_update(results)
            logger.info("Manual data update triggered")
        except Exception as e:
            logger.error(f"Failed to update data: {str(e)}")

    def open_settings(self):
        try:
            dip_th, ok = QInputDialog.getDouble(self, "Dip Threshold", "Enter dip threshold (0-1):", self.watcher.dip_threshold, 0, 1, decimals=2)
            if ok:
                self.watcher.dip_threshold = dip_th
                self.save_settings()
                logger.info(f"Updated dip threshold to {dip_th}")

            max_sp, ok = QInputDialog.getDouble(self, "Max Ask Spread", "Enter max ask spread (0-1):", self.watcher.max_ask_spread, 0, 1, decimals=2)
            if ok:
                self.watcher.max_ask_spread = max_sp
                self.save_settings()
                logger.info(f"Updated max ask spread to {max_sp}")

            lookback_str, ok = QInputDialog.getText(self, "Lookback Periods", "Enter lookback periods (comma separated):", text=",".join(map(str, self.watcher.lookback_periods)))
            if ok:
                try:
                    self.watcher.lookback_periods = tuple(map(int, lookback_str.split(',')))
                    self.save_settings()
                    logger.info(f"Updated lookback periods to {self.watcher.lookback_periods}")
                except ValueError as e:
                    logger.error(f"Invalid lookback periods input: {str(e)}")

            rsi_period, ok = QInputDialog.getInt(self, "RSI Period", "Enter RSI period (5-50):", self.watcher.rsi_period, 5, 50)
            if ok:
                self.watcher.rsi_period = rsi_period
                self.save_settings()
                logger.info(f"Updated RSI period to {rsi_period}")

            interval, ok = QInputDialog.getInt(self, "Interval (seconds)", "Enter poll interval:", self.interval // 1000, 5, 3600)
            if ok:
                self.interval = interval * 1000
                self.worker.interval = interval
                self.save_settings()
                logger.info(f"Updated poll interval to {interval} seconds")

            us_mult, ok = QInputDialog.getDouble(self, "US Volume Multiplier", "Enter US volume multiplier:", self.watcher.volume_multipliers.get('US', 0.8), 0, 2, decimals=2)
            if ok:
                self.watcher.volume_multipliers['US'] = us_mult
                self.save_settings()
                logger.info(f"Updated US volume multiplier to {us_mult}")

            jse_mult, ok = QInputDialog.getDouble(self, "JSE Volume Multiplier", "Enter JSE volume multiplier:", self.watcher.volume_multipliers.get('JSE', 0.5), 0, 2, decimals=2)
            if ok:
                self.watcher.volume_multipliers['JSE'] = jse_mult
                self.save_settings()
                logger.info(f"Updated JSE volume multiplier to {jse_mult}")

            lse_mult, ok = QInputDialog.getDouble(self, "LSE Volume Multiplier", "Enter LSE volume multiplier:", self.watcher.volume_multipliers.get('LSE', 0.7), 0, 2, decimals=2)
            if ok:
                self.watcher.volume_multipliers['LSE'] = lse_mult
                self.save_settings()
                logger.info(f"Updated LSE volume multiplier to {lse_mult}")
        except Exception as e:
            logger.error(f"Failed to open settings dialog: {str(e)}")
            QMessageBox.critical(self, "Settings Error", "Failed to update settings.")

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        try:
            self.tray.showMessage("Dip Watching", "Minimized to system tray.", QSystemTrayIcon.MessageIcon.Information, 2000)
            logger.info("Application minimized to system tray")
        except Exception as e:
            logger.error(f"Failed to minimize to tray: {str(e)}")

    def __del__(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
            logger.info("DataWorker thread stopped")
        if hasattr(self, 'log_timer'):
            self.log_timer.stop()
            logger.info("Log timer stopped")


if __name__ == "__main__":
    os.environ['QT_API'] = 'pyqt6'
    platforms = ['xcb', 'wayland']
    app = None
    for platform in platforms:
        os.environ['QT_QPA_PLATFORM'] = platform
        try:
            app = QApplication(sys.argv)
            logger.info(f"Initialized QApplication with platform {platform}")
            break
        except Exception as e:
            logger.error(f"Failed to initialize Qt with platform {platform}: {str(e)}")
            continue
    if not app:
        logger.critical("Could not initialize Qt platform. Install libxcb-cursor0: 'sudo apt install libxcb-cursor0 libx11-xcb1'")
        print("Error: Could not initialize Qt platform. Install libxcb-cursor0: 'sudo apt install libxcb-cursor0 libx11-xcb1'")
        sys.exit(1)

    window = MainWindow()
    window.show()
    logger.info("MainWindow displayed")
    sys.exit(app.exec())
