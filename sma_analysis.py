#%% Imports and Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import struct

# Import the read_ts module for data loading
from read_ts import read_ts_ohlcv_dat, read_ts_pnl_dat_multicore

# Import configuration and strategy
from input import *
from SMA_Strategy import SMAStrategy

#%% Configuration Parameters
# Trading Parameters
TICKER = 'MME'
START_DATE = '2014-01-01'
END_DATE = '2025-01-01'

# SMA Strategy Parameters
SMA_MIN = 10
SMA_MAX = 300
SMA_STEP = 5

# Backtest Settings
TRAIN_TEST_SPLIT = 0.7
ATR_PERIOD = 30
TRADING_CAPITAL = 6000

#%% Configuration
class Config:
    symbol = TICKER  # Use TICKER from input.py
    start_date = START_DATE if 'START_DATE' in globals() else '2010-01-01'
    end_date = END_DATE if 'END_DATE' in globals() else '2023-12-31'
    
    # Strategy parameters
    sma_short = 20
    sma_long = 50
    atr_period = ATR_PERIOD  # Use ATR_PERIOD from input.py
    risk_per_trade = 0.02  # 2% risk per trade
    
    # Backtest settings
    train_test_split = TRAIN_TEST_SPLIT  # Use TRAIN_TEST_SPLIT from input.py
    commission = 0.001  # 0.1% commission per trade

#%% Data Loading Classes and Functions
class mdata:
    """Class for storing market data and metadata"""
    def __init__(self):
        self.tick_size = None
        self.big_point_value = None
        self.country = None
        self.exchange = None
        self.symbol = None
        self.description = None
        self.interval_type = None
        self.interval_span = None
        self.time_zone = None
        self.session = None
        self.data = None

    def __repr__(self):
        return str((self.symbol, self.interval_type, self.interval_span, self.data.shape))

def read_ts_ohlcv_dat_one(fname) -> mdata:
    """Read a single OHLCV data file"""
    try:
        f = open(fname, "rb")
        d = read_ts_ohlcv_dat_one_stream(f)
    finally:
        f.close()
    return d

def read_ts_ohlcv_dat_one_stream(byte_stream) -> mdata:
    """Read OHLCV data from a byte stream"""
    def read_string(f):
        sz = struct.unpack('i', f.read(4))[0]
        s = f.read(sz).decode('ascii')
        return s

    d = mdata()
    try:
        f = byte_stream
        (ones, type_format) = struct.unpack('ii', f.read(8))
        if ones != 1111111111 or type_format != 3:
            return None
            
        d.tick_size = struct.unpack('d', f.read(8))[0]
        d.big_point_value = struct.unpack('d', f.read(8))[0]
        d.country = read_string(f)
        d.exchange = read_string(f)
        d.symbol = read_string(f)
        d.description = read_string(f)
        d.interval_type = read_string(f)
        d.interval_span = struct.unpack('i', f.read(4))[0]
        d.time_zone = read_string(f)
        d.session = read_string(f)

        dt = np.dtype([('date', 'f8'), ('open', 'f8'), ('high', 'f8'), 
                      ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')])
        data = pd.DataFrame.from_records(np.frombuffer(f.read(), dtype=dt))
        
        # Convert dates
        arr = ((data['date']-25569)*24*60*60).round(0).astype(np.int64)*1000000000
        z2 = pd.to_datetime(arr)
        data.insert(0, 'datetime', z2)
        del data['date']
        d.data = data
        
    finally:
        pass
    return d

def read_ts_ohlcv_dat(fnames) -> List[mdata]:
    """Read multiple OHLCV data files"""
    r = []
    for name in glob.glob(fnames):
        z = read_ts_ohlcv_dat_one(name)
        r.append(z)
    return r

#%% Technical Analysis Functions
def mavg(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Moving Average"""
    return np.concatenate([
        np.zeros((period-1)), 
        np.convolve(data, np.ones((period,))/period, mode='valid')
    ])

def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    return true_range.rolling(window=period).mean()

#%% SMA Strategy Class
class SMAStrategy:
    """
    SMA Crossover Trading Strategy with ATR-based Position Sizing
    """
    def __init__(self, short_sma: int, long_sma: int, big_point_value: float, 
                 slippage: float = 0, capital: float = 6000, atr_period: int = 50):
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.big_point_value = big_point_value
        self.slippage = slippage
        self.capital = capital
        self.atr_period = atr_period

    def apply_strategy(self, data: pd.DataFrame, strategy_name: str = "Strategy") -> pd.DataFrame:
        """Apply the strategy to price data"""
        sim_data = data.copy()
        
        # Calculate indicators
        sim_data[f'SMA_Short_{strategy_name}'] = mavg(sim_data['Close'].values, self.short_sma)
        sim_data[f'SMA_Long_{strategy_name}'] = mavg(sim_data['Close'].values, self.long_sma)
        sim_data[f'ATR_{strategy_name}'] = calculate_atr(sim_data, self.atr_period)
        
        # Calculate position sizes
        sim_data[f'Position_Size_{strategy_name}'] = np.round(
            self.capital / (sim_data[f'ATR_{strategy_name}'] * self.big_point_value) + 0.5
        )
        
        # Generate signals
        sim_data[f'Position_Dir_{strategy_name}'] = np.where(
            sim_data[f'SMA_Short_{strategy_name}'] > sim_data[f'SMA_Long_{strategy_name}'], 1, -1
        )
        
        # Calculate P&L
        sim_data[f'Position_Change_{strategy_name}'] = sim_data[f'Position_Dir_{strategy_name}'].diff() != 0
        market_pnl = sim_data['Close'].diff() * self.big_point_value
        sim_data[f'Market_PnL_{strategy_name}'] = market_pnl
        
        sim_data[f'Daily_PnL_{strategy_name}'] = (
            market_pnl * 
            sim_data[f'Position_Dir_{strategy_name}'].shift(1) * 
            sim_data[f'Position_Size_{strategy_name}'].shift(1)
        )
        
        # Apply slippage costs
        position_changed = sim_data[f'Position_Change_{strategy_name}']
        sim_data.loc[position_changed, f'Daily_PnL_{strategy_name}'] -= (
            self.slippage * sim_data[f'Position_Size_{strategy_name}'][position_changed]
        )
        
        # Calculate cumulative P&L
        sim_data[f'Daily_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].fillna(0)
        sim_data[f'Cumulative_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].cumsum()
        
        return sim_data

    def optimize(self, data: pd.DataFrame, sma_range: range, 
                train_test_split: float = 0.7, results_file: str = None, 
                warm_up_idx: int = None) -> Tuple:
        """Find optimal SMA parameters"""
        best_sharpe = -np.inf
        best_sma = (0, 0)
        best_trades = 0
        all_results = []
        
        if results_file:
            with open(results_file, 'w') as f:
                f.write("short_SMA,long_SMA,trades,sharpe_ratio\n")
        
        total_combinations = sum(1 for a, b in [(s, l) for s in sma_range 
                                              for l in sma_range] if a < b)
        completed = 0
        
        print(f"Running {total_combinations} simulations...")
        
        for short_sma in sma_range:
            for long_sma in sma_range:
                if short_sma >= long_sma:
                    continue
                    
                self.short_sma = short_sma
                self.long_sma = long_sma
                
                sim_data = self.apply_strategy(data.copy(), strategy_name="Sim")
                
                if warm_up_idx is not None:
                    sim_data_eval = sim_data.iloc[warm_up_idx:].copy()
                    split_index = int(len(sim_data_eval) * train_test_split)
                else:
                    sim_data_eval = sim_data
                    split_index = int(len(sim_data_eval) * train_test_split)
                
                trade_count = sim_data_eval['Position_Change_Sim'].sum()
                
                in_sample_returns = sim_data_eval['Daily_PnL_Sim'].iloc[:split_index]
                
                if len(in_sample_returns.dropna()) == 0 or in_sample_returns.std() == 0:
                    sharpe_ratio = 0
                else:
                    sharpe_ratio = (in_sample_returns.mean() / 
                                  in_sample_returns.std() * np.sqrt(252))
                
                if results_file:
                    with open(results_file, 'a') as f:
                        f.write(f"{short_sma},{long_sma},{trade_count},{sharpe_ratio:.6f}\n")
                
                all_results.append((short_sma, long_sma, trade_count, sharpe_ratio))
                
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_sma = (short_sma, long_sma)
                    best_trades = trade_count
                
                completed += 1
                if completed % 100 == 0 or completed == total_combinations:
                    print(f"Progress: {completed}/{total_combinations} "
                          f"({(completed/total_combinations*100):.1f}%)")
        
        return best_sma, best_sharpe, best_trades, all_results

    def calculate_performance_metrics(self, data: pd.DataFrame, strategy_name: str = "Strategy", 
                                   train_test_split: float = 0.7) -> Dict:
        """Calculate detailed performance metrics"""
        # Calculate split index for in-sample/out-of-sample
        split_index = int(len(data) * train_test_split)
        split_date = data.index[split_index]

        # Extract daily P&L data
        daily_pnl = data[f'Daily_PnL_{strategy_name}']
        
        # Split returns for performance comparison
        returns_in_sample = daily_pnl.iloc[:split_index]
        returns_out_sample = daily_pnl.iloc[split_index:]

        # Calculate Sharpe ratios
        sharpe_in_sample = returns_in_sample.mean() / returns_in_sample.std() * np.sqrt(252) if returns_in_sample.std() > 0 else 0
        sharpe_out_sample = returns_out_sample.mean() / returns_out_sample.std() * np.sqrt(252) if returns_out_sample.std() > 0 else 0
        sharpe_full = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

        # Calculate trade counts
        position_changes = data[f'Position_Change_{strategy_name}']
        total_trades = position_changes.sum()
        in_sample_trades = position_changes.iloc[:split_index].sum()
        out_sample_trades = position_changes.iloc[split_index:].sum()

        # Calculate max drawdown
        pnl_series = data[f'Cumulative_PnL_{strategy_name}']
        data.loc[:, 'Peak'] = pnl_series.cummax()
        data.loc[:, 'Drawdown_Dollars'] = pnl_series - data['Peak']
        max_drawdown_dollars = data['Drawdown_Dollars'].min()

        # Calculate P&L metrics
        total_pnl = data[f'Cumulative_PnL_{strategy_name}'].iloc[-1]
        in_sample_pnl = data[f'Daily_PnL_{strategy_name}'].iloc[:split_index].sum()
        out_sample_pnl = data[f'Daily_PnL_{strategy_name}'].iloc[split_index:].sum()

        # Calculate position size metrics
        avg_position_size = data[f'Position_Size_{strategy_name}'].mean()
        max_position_size = data[f'Position_Size_{strategy_name}'].max()

        return {
            'split_date': split_date,
            'total_pnl': total_pnl,
            'sharpe_full': sharpe_full,
            'sharpe_in_sample': sharpe_in_sample,
            'sharpe_out_sample': sharpe_out_sample,
            'max_drawdown_dollars': max_drawdown_dollars,
            'total_trades': total_trades,
            'in_sample_trades': in_sample_trades,
            'out_sample_trades': out_sample_trades,
            'in_sample_pnl': in_sample_pnl,
            'out_sample_pnl': out_sample_pnl,
            'avg_position_size': avg_position_size,
            'max_position_size': max_position_size
        }

#%% Helper Functions
def get_slippage_from_excel(symbol, data_dir):
    """Get the slippage value for a specific symbol from the Excel file"""
    excel_path = os.path.join(data_dir, "sessions_slippages.xlsx")
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Slippage Excel file not found at {excel_path}")
    
    lookup_symbol = symbol.replace('=F', '')
    df = pd.read_excel(excel_path)
    
    df['SymbolUpper'] = df.iloc[:, 1].astype(str).str.upper()
    lookup_symbol_upper = lookup_symbol.upper()
    
    matching_rows = df[df['SymbolUpper'] == lookup_symbol_upper]
    
    if matching_rows.empty:
        raise ValueError(f"Symbol '{lookup_symbol}' not found in column B of Excel file")
    
    slippage_value = matching_rows.iloc[0, 3]
    
    if pd.isna(slippage_value) or not isinstance(slippage_value, (int, float)):
        raise ValueError(f"Invalid slippage value for symbol '{lookup_symbol}': {slippage_value}")
    
    return slippage_value

def find_futures_file(symbol, data_dir):
    """Find a data file for the specified futures symbol"""
    patterns = [
        f"*@{symbol}_*.dat",
        f"*_@{symbol}_*.dat",
        f"*_{symbol}_*.dat",
        f"*@{symbol}*.dat"
    ]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        if files:
            return files[0]
    
    raise FileNotFoundError(f"No data file found for {symbol} in {data_dir}")

def plot_results(df: pd.DataFrame, metrics: Dict, title: str):
    """Plot equity curve and key metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot equity curve
    df['Cumulative_PnL_Strategy'].plot(ax=ax1, title=f'{title} - Equity Curve')
    ax1.set_ylabel('Portfolio Value')
    
    # Plot SMAs and price
    df['Close'].plot(ax=ax2, label='Price')
    df['SMA_Short_Strategy'].plot(ax=ax2, label=f'SMA Short')
    df['SMA_Long_Strategy'].plot(ax=ax2, label=f'SMA Long')
    ax2.set_ylabel('Price')
    ax2.legend()
    
    # Add metrics text
    metrics_text = (f"Total Returns: ${metrics['total_pnl']:,.2f}\n"
                   f"Sharpe Ratio: {metrics['sharpe_full']:.2f}\n"
                   f"Max Drawdown: ${abs(metrics['max_drawdown_dollars']):,.2f}\n"
                   f"Total Trades: {metrics['total_trades']}")
    plt.figtext(0.15, 0.15, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

#%% Main Execution
if __name__ == "__main__":
    # Setup paths
    WORKING_DIR = "."
    DATA_DIR = os.path.join(WORKING_DIR, "data")
    
    # Create output directory
    SYMBOL = TICKER.replace('=F', '')
    output_dir = os.path.join(WORKING_DIR, 'output', SYMBOL)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data file
    print("Loading data file...")
    data_file = find_futures_file(SYMBOL, DATA_DIR)
    all_data = read_ts_ohlcv_dat(data_file)
    
    # Extract metadata and OHLCV data
    data_obj = all_data[0]
    tick_size = data_obj.big_point_value * data_obj.tick_size
    big_point_value = data_obj.big_point_value
    
    # Get OHLCV data
    data = data_obj.data.copy()
    
    # Get slippage from Excel
    slippage = get_slippage_from_excel(TICKER, DATA_DIR)
    print(f"Using slippage from Excel: {slippage}")
    
    # Save parameters
    parameters = {
        "big_point_value": big_point_value,
        "slippage": slippage,
        "capital": TRADING_CAPITAL,
        "atr_period": ATR_PERIOD
    }
    with open("parameters.json", "w") as f:
        json.dump(parameters, f)
    
    # Print data info
    print(f"\nSymbol: {data_obj.symbol}")
    print(f"Description: {data_obj.description}")
    print(f"Exchange: {data_obj.exchange}")
    print(f"Interval: {data_obj.interval_type} {data_obj.interval_span}")
    print(f"Tick size: {tick_size}")
    print(f"Big point value: {big_point_value}")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    
    # Prepare data for strategy
    data = data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data.set_index('Date', inplace=True)
    
    # Add warm-up period
    original_start_idx = None
    if START_DATE and END_DATE:
        warm_up_days = SMA_MAX + ATR_PERIOD + 50
        start_date = pd.to_datetime(START_DATE)
        end_date = pd.to_datetime(END_DATE)
        adjusted_start = start_date - pd.Timedelta(days=warm_up_days)
        data = data[(data.index >= adjusted_start) & (data.index <= end_date)]
        original_start_idx = data.index.get_indexer([start_date], method='nearest')[0]
    
    # Initialize strategy with parameters from data
    strategy = SMAStrategy(
        short_sma=0,  # Will be set during optimization
        long_sma=0,  # Will be set during optimization
        big_point_value=big_point_value,
        slippage=slippage,
        capital=TRADING_CAPITAL,
        atr_period=ATR_PERIOD
    )
    
    # Find optimal parameters
    print("\nOptimizing strategy parameters...")
    sma_range = range(SMA_MIN, SMA_MAX + 1, SMA_STEP)
    best_sma, best_sharpe, best_trades, all_results = strategy.optimize(
        data.copy(),
        sma_range,
        train_test_split=TRAIN_TEST_SPLIT,
        results_file='sma_all_results.txt',
        warm_up_idx=original_start_idx
    )
    
    # Update strategy with best parameters
    strategy.short_sma = best_sma[0]
    strategy.long_sma = best_sma[1]
    
    # Apply strategy and calculate metrics
    results = strategy.apply_strategy(data.copy())
    metrics = strategy.calculate_performance_metrics(
        results,
        strategy_name="Strategy",
        train_test_split=TRAIN_TEST_SPLIT
    )
    
    # Plot results
    plot_results(results, metrics, "Strategy Performance")
    
    # Save results
    results = {
        'metrics': metrics,
        'parameters': {
            'sma_short': best_sma[0],
            'sma_long': best_sma[1],
            'atr_period': ATR_PERIOD,
            'risk_per_trade': 0.02
        }
    }
    
    with open('strategy_results.json', 'w') as f:
        json.dump(results, f, indent=4) 