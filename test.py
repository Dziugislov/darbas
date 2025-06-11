import pandas as pd
import numpy as np
import os
import glob
import read_ts
from SMA_Strategy import SMAStrategy

# User parameters - modify these as needed
TICKER = 'MME'
SHORT_SMA = 15  # Your desired short SMA period
LONG_SMA = 40   # Your desired long SMA period
START_DATE = '2014-01-01'
END_DATE = '2025-01-01'
TRAIN_TEST_SPLIT = 0.7
ATR_PERIOD = 30
TRADING_CAPITAL = 6000

def main():
    # Setup paths
    WORKING_DIR = "."
    DATA_DIR = os.path.join(WORKING_DIR, "data")
    
    # Define SYMBOL based on TICKER
    SYMBOL = TICKER.replace('=F', '')
    
    # Function to get slippage from Excel
    def get_slippage_from_excel(symbol, data_dir):
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

    # Find and load the data file
    def find_futures_file(symbol, data_dir):
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

    # Load the data
    data_file = find_futures_file(SYMBOL, DATA_DIR)
    print(f"Loading data from: {data_file}")
    
    all_data = read_ts.read_ts_ohlcv_dat(data_file)
    data_obj = all_data[0]
    
    # Get parameters from data
    big_point_value = data_obj.big_point_value
    slippage = get_slippage_from_excel(TICKER, DATA_DIR)
    
    # Prepare the data
    ohlc_data = data_obj.data.copy()
    data = ohlc_data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data.set_index('Date', inplace=True)
    
    # Add warm-up period
    warm_up_days = max(SHORT_SMA, LONG_SMA) + ATR_PERIOD + 50
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)
    adjusted_start = start_date - pd.Timedelta(days=warm_up_days)
    
    # Filter data with warm-up period
    data = data[(data.index >= adjusted_start) & (data.index <= end_date)]
    original_start_idx = data.index.get_indexer([start_date], method='nearest')[0]
    
    # Initialize strategy
    strategy = SMAStrategy(
        short_sma=SHORT_SMA,
        long_sma=LONG_SMA,
        big_point_value=big_point_value,
        slippage=slippage,
        capital=TRADING_CAPITAL,
        atr_period=ATR_PERIOD
    )
    
    # Apply strategy
    sim_data = strategy.apply_strategy(data.copy())
    
    # Trim warm-up period
    sim_data_eval = sim_data.iloc[original_start_idx:].copy()
    
    # Calculate metrics
    metrics = strategy.calculate_performance_metrics(
        sim_data_eval,
        strategy_name="Strategy",
        train_test_split=TRAIN_TEST_SPLIT
    )
    
    # Print results
    print("\n=== Strategy Performance ===")
    print(f"Parameters: Short SMA = {SHORT_SMA}, Long SMA = {LONG_SMA}")
    print(f"In-sample Sharpe Ratio: {metrics['sharpe_in_sample']:.4f}")
    print(f"Out-of-sample Sharpe Ratio: {metrics['sharpe_out_sample']:.4f}")
    print(f"Full Period Sharpe Ratio: {metrics['sharpe_full']:.4f}")
    print(f"\nAdditional Metrics:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"Maximum Drawdown: ${abs(metrics['max_drawdown_dollars']):,.2f}")

if __name__ == "__main__":
    main() 