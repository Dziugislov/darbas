#!/usr/bin/env python3
"""
Portfolio Comparison Script with Warm-Up Period

This script adds a proper warm-up period implementation to the portfolio comparison script,
ensuring that strategies have enough historical data for calculation of indicators
before the actual analysis period starts.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl
import time
from datetime import datetime
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use('Agg')

# Import local modules
import read_ts
from SMA_Strategy import SMAStrategy
from input_gen import TRAIN_TEST_SPLIT, ATR_PERIOD, TRADING_CAPITAL, START_DATE, END_DATE, SMA_MAX

# Constants
EXCEL_FILE = r"C:\Users\Admin\Documents\darbas\Results.xlsx"
WORKING_DIR = "."
DATA_DIR = os.path.join(WORKING_DIR, "data")
OUTPUT_DIR = os.path.join(WORKING_DIR, 'output', 'portfolios')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_slippage_from_excel(symbol, data_dir):
    """
    Get the slippage value for a specific symbol from the Excel file
    
    Parameters:
    symbol: str - The trading symbol to look up (without '=F')
    data_dir: str - Directory containing the Excel file
    
    Returns:
    float - Slippage value for the symbol
    """
    excel_path = os.path.join(data_dir, "sessions_slippages.xlsx")
    
    # No fallback - if file doesn't exist, crash
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Slippage Excel file not found at {excel_path}")
    
    # Remove '=F' suffix if present for lookup
    lookup_symbol = symbol.replace('=F', '')
    
    # Read the Excel file - will throw exception if any issue
    df = pd.read_excel(excel_path)
    
    # Print the Excel contents for debugging
    print("\nContents of sessions_slippages.xlsx:")
    print(df.head())
    
    # Check if we have at least 4 columns (to access column D)
    if df.shape[1] < 4:
        raise ValueError(f"Excel file has fewer than 4 columns: {df.columns.tolist()}")
        
    # Print column names for debugging
    print(f"Columns: {df.columns.tolist()}")
    
    # Use direct column access - Column B (index 1) for symbol, Column D (index 3) for slippage
    # First convert to uppercase for case-insensitive comparison
    df['SymbolUpper'] = df.iloc[:, 1].astype(str).str.upper()
    lookup_symbol_upper = lookup_symbol.upper()
    
    # Find the matching row
    matching_rows = df[df['SymbolUpper'] == lookup_symbol_upper]
    
    if matching_rows.empty:
        raise ValueError(f"Symbol '{lookup_symbol}' not found in column B of Excel file")
        
    # Get the slippage value from column D (index 3)
    slippage_value = matching_rows.iloc[0, 3]
    
    # Validate the slippage value is numeric
    if pd.isna(slippage_value) or not isinstance(slippage_value, (int, float)):
        raise ValueError(f"Invalid slippage value for symbol '{lookup_symbol}': {slippage_value}")
        
    print(f"Found slippage for {lookup_symbol} in column D: {slippage_value}")
    return slippage_value

def parse_sma_params(param_str):
    """Convert string SMA parameters (format: 'short/long') to tuple of integers"""
    if pd.isna(param_str) or param_str is None:
        return None
    try:
        short, long = str(param_str).split('/')
        return (int(short), int(long))
    except (ValueError, TypeError):
        print(f"Error parsing SMA parameter: {param_str}")
        return None

def find_futures_file(symbol, data_dir):
    """Find a data file for the specified futures symbol"""
    pattern = f"*@{symbol}_*.dat"
    files = glob.glob(os.path.join(data_dir, pattern))
    
    if not files:
        pattern = f"*_@{symbol}_*.dat"
        files = glob.glob(os.path.join(data_dir, pattern))
    
    if not files:
        pattern = f"*_{symbol}_*.dat"
        files = glob.glob(os.path.join(data_dir, pattern))
    
    if not files:
        pattern = f"*@{symbol}*.dat"
        files = glob.glob(os.path.join(data_dir, pattern))
    
    if not files:
        raise FileNotFoundError(f"No data files matching any pattern for symbol {symbol} in {data_dir}")
    
    return files[0]

def load_data(ticker):
    """
    Load market data for a ticker and filter by date range from input.py
    Includes proper warm-up period for SMA and ATR calculation
    """
    print(f"Loading data for {ticker}...")
    symbol = ticker.replace('=F', '')
    data_file = find_futures_file(symbol, DATA_DIR)
    
    if not data_file:
        raise FileNotFoundError(f"No data file found for {ticker} in {DATA_DIR}")
    
    # Load the futures data file - no error handling, let it crash if there's an issue
    all_data = read_ts.read_ts_ohlcv_dat(data_file)
    data_obj = all_data[0]
    
    # Get big point value
    big_point_value = data_obj.big_point_value
    
    # Fetch slippage value from Excel - NO FALLBACK, will crash if it fails
    slippage_value = get_slippage_from_excel(symbol, DATA_DIR)
    slippage = slippage_value  # No longer dividing by big_point_value
    print(f"Using slippage from Excel column D: {slippage_value}")
    
    # Convert OHLCV data
    ohlc_data = data_obj.data.copy()
    data = ohlc_data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Store original data length for reference
    original_data_length = len(data)
    
    # Filter data based on START_DATE and END_DATE from input.py
    # If dates are specified, add a warm-up period for SMA and ATR calculations
    original_start_idx = None
    
    if START_DATE and END_DATE:
        # Calculate warm-up period (longest SMA + buffer for ATR calculation)
        # Use SMA_MAX as reference for longest possible SMA period
        warm_up_days = SMA_MAX + ATR_PERIOD + 50  # Add buffer days for safety
        
        # Convert dates to datetime
        start_date = pd.to_datetime(START_DATE)
        end_date = pd.to_datetime(END_DATE)
        
        # Adjust start date for warm-up
        adjusted_start = start_date - pd.Timedelta(days=warm_up_days)
        
        # Get all available data including warm-up period
        warm_up_data = data[(data.index >= adjusted_start) & (data.index <= end_date)]
        
        # Find index corresponding to original start date
        if warm_up_data.empty:
            raise ValueError(f"No data available for {ticker} including warm-up period")
            
        # Find the closest index to our original start date
        date_positions = warm_up_data.index.get_indexer([start_date], method='nearest')
        if len(date_positions) > 0:
            original_start_idx = date_positions[0]
        else:
            raise ValueError(f"Cannot find index for start date {start_date}")
            
        # Use the extended data with warm-up period
        data = warm_up_data
        
        print(f"Loaded extended data with {warm_up_days} days warm-up period for {ticker}")
        print(f"Original date range: {START_DATE} to {END_DATE}")
        print(f"Adjusted date range: {adjusted_start.strftime('%Y-%m-%d')} to {END_DATE}")
        print(f"Original data length: {original_data_length}, Extended data length: {len(data)}")
        print(f"Start date index: {original_start_idx}")
    
    if data.empty:
        raise ValueError(f"No data after filtering for {ticker}")
    
    # Return data with warm-up period, market parameters, and original start index
    return data, (big_point_value, slippage), original_start_idx

def run_strategy(data, params, big_point_value, slippage, 
                original_start_idx=None, capital=TRADING_CAPITAL, 
                atr_period=ATR_PERIOD, name="Strategy"):
    """
    Run a trading strategy with the given parameters
    
    Parameters:
    data: DataFrame with OHLC data, including warm-up period
    params: Tuple of (short_sma, long_sma)
    big_point_value: Contract point value
    slippage: Slippage amount per contract
    original_start_idx: Index position where actual analysis should start (after warm-up)
    capital: Trading capital
    atr_period: ATR period for position sizing
    name: Strategy name identifier
    
    Returns:
    DataFrame with strategy results
    """
    if data is None or params is None:
        return None
    
    short_sma, long_sma = params
    
    # Create strategy instance
    strategy = SMAStrategy(
        short_sma=short_sma,
        long_sma=long_sma,
        big_point_value=big_point_value,
        slippage=slippage,
        capital=capital,
        atr_period=atr_period
    )
    
    # Apply the strategy to the entire dataset including warm-up period
    try:
        result = strategy.apply_strategy(data.copy(), strategy_name=name)
        
        # If we have a specified start index (after warm-up), update P&L values
        if original_start_idx is not None and original_start_idx > 0:
            # Reset the cumulative P&L to start from zero at the original start date
            pnl_column = f'Daily_PnL_{name}'
            cum_pnl_column = f'Cumulative_PnL_{name}'
            
            # Recalculate cumulative P&L starting from the original start date
            result[cum_pnl_column] = result[pnl_column].cumsum()
            
            # For reporting purposes, we can also create a version that excludes warm-up
            result[f'{cum_pnl_column}_NoWarmup'] = 0.0
            result.loc[result.index[original_start_idx:], f'{cum_pnl_column}_NoWarmup'] = \
                result.loc[result.index[original_start_idx:], pnl_column].cumsum()
            
            print(f"Adjusted cumulative P&L for {name} to start from original start date")
        
        return result
    except Exception as e:
        print(f"Error running strategy with params {params}: {e}")
        return None

def build_portfolios():
    """Build the three portfolios based on Excel parameters with proper warm-up period"""
    print("Building portfolios...")
    
    # Load parameters from Excel
    parameters = load_excel_parameters()
    if not parameters:
        print("Failed to load parameters. Cannot build portfolios.")
        return None
    
    # Initialize portfolio data
    portfolio_data = {
        'Best_Sharpe': [],
        'Kmeans': [],
        'Hierarchy': []
    }
    
    # Process each instrument
    for params in parameters:
        ticker = params['Ticker']
        print(f"\nProcessing {ticker}...")
        
        # Load market data with warm-up period
        try:
            data, market_params, original_start_idx = load_data(ticker)
            
            if data is None or market_params is None:
                print(f"Skipping {ticker} due to data loading error")
                continue
                
            big_point_value, slippage = market_params
            
            # Parse parameters
            best_sharpe_params = parse_sma_params(params['Best_Sharpe'])
            
            kmeans_params = []
            for param in params['Kmeans_Clusters']:
                parsed = parse_sma_params(param)
                if parsed:
                    kmeans_params.append(parsed)
            
            hierarchy_params = []
            for param in params['Hierarchy_Clusters']:
                parsed = parse_sma_params(param)
                if parsed:
                    hierarchy_params.append(parsed)
            
            # Run best Sharpe strategy
            if best_sharpe_params:
                print(f"Running Best Sharpe strategy for {ticker} with params {best_sharpe_params}")
                best_sharpe_result = run_strategy(
                    data, 
                    best_sharpe_params, 
                    big_point_value, 
                    slippage,
                    original_start_idx,
                    name=f"BestSharpe_{ticker}"
                )
                
                if best_sharpe_result is not None:
                    portfolio_data['Best_Sharpe'].append({
                        'ticker': ticker,
                        'data': best_sharpe_result,
                        'pnl_column': f'Daily_PnL_BestSharpe_{ticker}',
                        'original_start_idx': original_start_idx
                    })
                else:
                    print(f"Best Sharpe strategy failed for {ticker}")
            
            # Run K-means strategies
            kmeans_results = []
            base_data = data.copy()
            for i, k_params in enumerate(kmeans_params, 1):  # Start indexing at 1
                strategy_name = f"Kmeans_{i}_{ticker}"
                print(f"Running K-means strategy {i} for {ticker} with params {k_params}")
                result = run_strategy(
                    base_data,  # Use the same base data to accumulate all Daily_PnL columns
                    k_params, 
                    big_point_value, 
                    slippage,
                    original_start_idx,
                    name=strategy_name
                )
                
                if result is not None:
                    base_data = result  # Update base_data with the new Daily_PnL column
                    kmeans_results.append({
                        'data': result,
                        'pnl_column': f'Daily_PnL_{strategy_name}'
                    })
            
            # Only proceed if we have K-means results
            if kmeans_results:
                # Average the K-means results
                base_data[f'Daily_PnL_Kmeans_Avg_{ticker}'] = 0
                for result in kmeans_results:
                    pnl_column = result['pnl_column']
                    base_data[f'Daily_PnL_Kmeans_Avg_{ticker}'] += base_data[pnl_column]
                
                base_data[f'Daily_PnL_Kmeans_Avg_{ticker}'] /= len(kmeans_results)
                
                # Calculate cumulative P&L, with proper handling for warm-up period
                if original_start_idx is not None and original_start_idx > 0:
                    # Recalculate cumulative P&L starting from the analysis period
                    base_data[f'Cumulative_PnL_Kmeans_Avg_{ticker}'] = base_data[f'Daily_PnL_Kmeans_Avg_{ticker}'].cumsum()
                else:
                    base_data[f'Cumulative_PnL_Kmeans_Avg_{ticker}'] = base_data[f'Daily_PnL_Kmeans_Avg_{ticker}'].cumsum()
                
                portfolio_data['Kmeans'].append({
                    'ticker': ticker,
                    'data': base_data,
                    'pnl_column': f'Daily_PnL_Kmeans_Avg_{ticker}',
                    'original_start_idx': original_start_idx
                })
            else:
                print(f"No K-means results for {ticker}")
            
            # Run Hierarchical strategies
            hierarchy_results = []
            base_data = data.copy()
            for i, h_params in enumerate(hierarchy_params, 1):  # Start indexing at 1
                strategy_name = f"Hierarchy_{i}_{ticker}"
                print(f"Running Hierarchical strategy {i} for {ticker} with params {h_params}")
                result = run_strategy(
                    base_data,  # Use the same base data to accumulate all Daily_PnL columns
                    h_params, 
                    big_point_value, 
                    slippage,
                    original_start_idx,
                    name=strategy_name
                )
                
                if result is not None:
                    base_data = result  # Update base_data with the new Daily_PnL column
                    hierarchy_results.append({
                        'data': result,
                        'pnl_column': f'Daily_PnL_{strategy_name}'
                    })
            
            # Only proceed if we have Hierarchical results
            if hierarchy_results:
                # Average the Hierarchical results
                base_data[f'Daily_PnL_Hierarchy_Avg_{ticker}'] = 0
                for result in hierarchy_results:
                    pnl_column = result['pnl_column']
                    base_data[f'Daily_PnL_Hierarchy_Avg_{ticker}'] += base_data[pnl_column]
                
                base_data[f'Daily_PnL_Hierarchy_Avg_{ticker}'] /= len(hierarchy_results)
                
                # Calculate cumulative P&L, with proper handling for warm-up period
                if original_start_idx is not None and original_start_idx > 0:
                    # Recalculate cumulative P&L starting from the analysis period
                    base_data[f'Cumulative_PnL_Hierarchy_Avg_{ticker}'] = base_data[f'Daily_PnL_Hierarchy_Avg_{ticker}'].cumsum()
                else:
                    base_data[f'Cumulative_PnL_Hierarchy_Avg_{ticker}'] = base_data[f'Daily_PnL_Hierarchy_Avg_{ticker}'].cumsum()
                
                portfolio_data['Hierarchy'].append({
                    'ticker': ticker,
                    'data': base_data,
                    'pnl_column': f'Daily_PnL_Hierarchy_Avg_{ticker}',
                    'original_start_idx': original_start_idx
                })
            else:
                print(f"No Hierarchical results for {ticker}")
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    print(f"\nPortfolio summary:")
    print(f"Best Sharpe portfolio: {len(portfolio_data['Best_Sharpe'])} instruments")
    print(f"K-means portfolio: {len(portfolio_data['Kmeans'])} instruments")
    print(f"Hierarchical portfolio: {len(portfolio_data['Hierarchy'])} instruments")
    
    return portfolio_data

def create_combined_portfolio_data(portfolio_data):
    """
    Combine all portfolio data into a single DataFrame with combined P&L
    Takes into account warm-up periods for proper analysis
    """
    print("Creating combined portfolio data...")
    
    # Identify common date range across all portfolios
    all_dates = set()
    for portfolio_type in portfolio_data.values():
        for instrument in portfolio_type:
            # If there's a warm-up period, only include dates after the warm-up
            if instrument.get('original_start_idx') is not None and instrument['original_start_idx'] > 0:
                # Use only dates after warm-up period
                analysis_dates = instrument['data'].index[instrument['original_start_idx']:]
                all_dates.update(analysis_dates)
            else:
                # Use all dates
                all_dates.update(instrument['data'].index)
    
    # Sort dates and create empty combined DataFrame
    dates = sorted(all_dates)
    combined_data = pd.DataFrame(index=dates)
    combined_data.index.name = 'Date'
    
    # Initialize portfolio P&L columns
    initial_columns = {
        'Best_Sharpe_Daily_PnL': pd.Series(0, index=dates),
        'Kmeans_Daily_PnL': pd.Series(0, index=dates),
        'Hierarchy_Daily_PnL': pd.Series(0, index=dates)
    }
    combined_data = pd.concat([combined_data, pd.DataFrame(initial_columns)], axis=1)
    
    # Prepare dictionaries to collect all instrument columns before adding them
    best_sharpe_columns = {}
    kmeans_columns = {}
    hierarchy_columns = {}
    
    # Add P&L from each instrument to the respective portfolio
    # Best Sharpe Portfolio
    for instrument in portfolio_data['Best_Sharpe']:
        ticker = instrument['ticker']
        data = instrument['data']
        pnl_column = instrument['pnl_column']
        original_start_idx = instrument.get('original_start_idx')
        
        # Determine which part of the data to use (after warm-up if applicable)
        if original_start_idx is not None and original_start_idx > 0:
            instrument_data = data.iloc[original_start_idx:]
        else:
            instrument_data = data
        
        # Align dates and add P&L to the combined data
        common_dates = combined_data.index.intersection(instrument_data.index)
        combined_data.loc[common_dates, 'Best_Sharpe_Daily_PnL'] += instrument_data.loc[common_dates, pnl_column]
        
        # Prepare individual instrument P&L column for later addition
        temp_series = pd.Series(0, index=combined_data.index)
        temp_series.loc[common_dates] = instrument_data.loc[common_dates, pnl_column]
        best_sharpe_columns[f'Best_Sharpe_{ticker}_PnL'] = temp_series
    
    # K-means Portfolio
    for instrument in portfolio_data['Kmeans']:
        ticker = instrument['ticker']
        data = instrument['data']
        pnl_column = instrument['pnl_column']
        original_start_idx = instrument.get('original_start_idx')
        
        # Determine which part of the data to use (after warm-up if applicable)
        if original_start_idx is not None and original_start_idx > 0:
            instrument_data = data.iloc[original_start_idx:]
        else:
            instrument_data = data
        
        # Align dates and add P&L to the combined data
        common_dates = combined_data.index.intersection(instrument_data.index)
        combined_data.loc[common_dates, 'Kmeans_Daily_PnL'] += instrument_data.loc[common_dates, pnl_column]
        
        # Prepare individual instrument P&L column for later addition
        temp_series = pd.Series(0, index=combined_data.index)
        temp_series.loc[common_dates] = instrument_data.loc[common_dates, pnl_column]
        kmeans_columns[f'Kmeans_{ticker}_PnL'] = temp_series
    
    # Hierarchical Portfolio
    for instrument in portfolio_data['Hierarchy']:
        ticker = instrument['ticker']
        data = instrument['data']
        pnl_column = instrument['pnl_column']
        original_start_idx = instrument.get('original_start_idx')
        
        # Determine which part of the data to use (after warm-up if applicable)
        if original_start_idx is not None and original_start_idx > 0:
            instrument_data = data.iloc[original_start_idx:]
        else:
            instrument_data = data
        
        # Align dates and add P&L to the combined data
        common_dates = combined_data.index.intersection(instrument_data.index)
        combined_data.loc[common_dates, 'Hierarchy_Daily_PnL'] += instrument_data.loc[common_dates, pnl_column]
        
        # Prepare individual instrument P&L column for later addition
        temp_series = pd.Series(0, index=combined_data.index)
        temp_series.loc[common_dates] = instrument_data.loc[common_dates, pnl_column]
        hierarchy_columns[f'Hierarchy_{ticker}_PnL'] = temp_series
    
    # Add all individual instrument P&L columns to the combined DataFrame at once
    # Add Best Sharpe columns
    if best_sharpe_columns:
        combined_data = pd.concat([combined_data, pd.DataFrame(best_sharpe_columns)], axis=1)
        
    # Add K-means columns
    if kmeans_columns:
        combined_data = pd.concat([combined_data, pd.DataFrame(kmeans_columns)], axis=1)
        
    # Add Hierarchical columns
    if hierarchy_columns:
        combined_data = pd.concat([combined_data, pd.DataFrame(hierarchy_columns)], axis=1)
    
    # Calculate cumulative P&L for each portfolio
    combined_data['Best_Sharpe_Cumulative_PnL'] = combined_data['Best_Sharpe_Daily_PnL'].cumsum()
    combined_data['Kmeans_Cumulative_PnL'] = combined_data['Kmeans_Daily_PnL'].cumsum()
    combined_data['Hierarchy_Cumulative_PnL'] = combined_data['Hierarchy_Daily_PnL'].cumsum()
    
    # Calculate cumulative P&L for individual instruments
    # Collect all ticker names across all portfolios
    all_tickers = set()
    for portfolio_type in portfolio_data.values():
        for instrument in portfolio_type:
            all_tickers.add(instrument['ticker'])
    
    # Prepare cumulative P&L columns
    cumulative_columns = {}
    
    for ticker in all_tickers:
        # Best Sharpe
        if f'Best_Sharpe_{ticker}_PnL' in combined_data.columns:
            cumulative_columns[f'Best_Sharpe_{ticker}_Cumulative_PnL'] = combined_data[f'Best_Sharpe_{ticker}_PnL'].cumsum()
        
        # K-means
        if f'Kmeans_{ticker}_PnL' in combined_data.columns:
            cumulative_columns[f'Kmeans_{ticker}_Cumulative_PnL'] = combined_data[f'Kmeans_{ticker}_PnL'].cumsum()
        
        # Hierarchical
        if f'Hierarchy_{ticker}_PnL' in combined_data.columns:
            cumulative_columns[f'Hierarchy_{ticker}_Cumulative_PnL'] = combined_data[f'Hierarchy_{ticker}_PnL'].cumsum()
    
    # Add all cumulative columns at once
    if cumulative_columns:
        combined_data = pd.concat([combined_data, pd.DataFrame(cumulative_columns)], axis=1)
    
    # Calculate in-sample/out-of-sample split
    split_index = int(len(combined_data) * TRAIN_TEST_SPLIT)
    split_date = combined_data.index[split_index]
    
    print(f"Created combined portfolio data with {len(combined_data)} days")
    print(f"In-sample/out-of-sample split date: {split_date}")
    
    return combined_data, split_date

def load_excel_parameters():
    """Load parameters from Excel file"""
    print(f"Loading parameters from {EXCEL_FILE}...")
    
    if not os.path.exists(EXCEL_FILE):
        print(f"Error: Excel file not found at {EXCEL_FILE}")
        return None
    
    try:
        # Load workbook
        wb = openpyxl.load_workbook(EXCEL_FILE)
        sheet = wb.active
        
        # Create DataFrame to store parameters
        parameters = []
        
        # Start from row 3 (after headers)
        for row in range(3, sheet.max_row + 1):
            ticker = sheet.cell(row=row, column=1).value
            
            # Skip empty rows
            if not ticker:
                continue
            
            # Extract parameters
            params = {
                'Ticker': ticker,
                'Best_Sharpe': sheet.cell(row=row, column=13).value,  # Column M
                'Kmeans_Clusters': [
                    sheet.cell(row=row, column=5).value,  # Column E
                    sheet.cell(row=row, column=6).value,  # Column F
                    sheet.cell(row=row, column=7).value,  # Column G
                ],
                'Hierarchy_Clusters': [
                    sheet.cell(row=row, column=9).value,  # Column I
                    sheet.cell(row=row, column=10).value, # Column J
                    sheet.cell(row=row, column=11).value, # Column K
                ]
            }
            
            parameters.append(params)
        
        print(f"Loaded parameters for {len(parameters)} instruments")
        return parameters
    
    except Exception as e:
        print(f"Error loading Excel parameters: {e}")
        return None

def calculate_performance_metrics(combined_data, split_date):
    """Calculate in-sample and out-of-sample P&L and Sharpe ratios for each portfolio"""
    # Define portfolios to analyze
    portfolios = ['Best_Sharpe', 'Kmeans', 'Hierarchy']
    performance_data = []
    
    # Split data into in-sample and out-of-sample
    in_sample = combined_data[combined_data.index < split_date]
    out_sample = combined_data[combined_data.index >= split_date]
    
    for portfolio in portfolios:
        # In-sample metrics
        in_sample_pnl = in_sample[f'{portfolio}_Daily_PnL'].sum()
        in_sample_sharpe = (in_sample[f'{portfolio}_Daily_PnL'].mean() / 
                           in_sample[f'{portfolio}_Daily_PnL'].std() * np.sqrt(252)) if in_sample[f'{portfolio}_Daily_PnL'].std() > 0 else 0
        
        # Out-of-sample metrics
        out_sample_pnl = out_sample[f'{portfolio}_Daily_PnL'].sum()
        out_sample_sharpe = (out_sample[f'{portfolio}_Daily_PnL'].mean() / 
                            out_sample[f'{portfolio}_Daily_PnL'].std() * np.sqrt(252)) if out_sample[f'{portfolio}_Daily_PnL'].std() > 0 else 0
        
        # Full period metrics
        full_pnl = combined_data[f'{portfolio}_Daily_PnL'].sum()
        full_sharpe = (combined_data[f'{portfolio}_Daily_PnL'].mean() / 
                      combined_data[f'{portfolio}_Daily_PnL'].std() * np.sqrt(252)) if combined_data[f'{portfolio}_Daily_PnL'].std() > 0 else 0
        
        # Store metrics
        performance_data.append({
            'Portfolio': portfolio.replace('_', ' '),
            'Period': 'In-Sample',
            'P&L ($)': in_sample_pnl,
            'Sharpe Ratio': in_sample_sharpe
        })
        performance_data.append({
            'Portfolio': portfolio.replace('_', ' '),
            'Period': 'Out-of-Sample',
            'P&L ($)': out_sample_pnl,
            'Sharpe Ratio': out_sample_sharpe
        })
        performance_data.append({
            'Portfolio': portfolio.replace('_', ' '),
            'Period': 'Full',
            'P&L ($)': full_pnl,
            'Sharpe Ratio': full_sharpe
        })
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # Save to Excel
    output_file = os.path.join(OUTPUT_DIR, 'Portfolio_Performance_Metrics.xlsx')
    performance_df.to_excel(output_file, index=False)
    print(f"Performance metrics saved to: {output_file}")
    
    return performance_df

def plot_portfolio_comparison(combined_data, split_date):
    """Create a performance comparison chart for all three portfolios"""
    print("Creating portfolio performance comparison chart...")
    
    # Calculate performance metrics for the legend
    performance_df = calculate_performance_metrics(combined_data, split_date)
    
    # Extract in-sample and out-of-sample metrics for each portfolio
    metrics = {}
    for portfolio in ['Best Sharpe', 'Kmeans', 'Hierarchy']:
        metrics[portfolio] = {
            'in_sample_pnl': performance_df[(performance_df['Portfolio'] == portfolio) & (performance_df['Period'] == 'In-Sample')]['P&L ($)'].iloc[0],
            'in_sample_sharpe': performance_df[(performance_df['Portfolio'] == portfolio) & (performance_df['Period'] == 'In-Sample')]['Sharpe Ratio'].iloc[0],
            'out_sample_pnl': performance_df[(performance_df['Portfolio'] == portfolio) & (performance_df['Period'] == 'Out-of-Sample')]['P&L ($)'].iloc[0],
            'out_sample_sharpe': performance_df[(performance_df['Portfolio'] == portfolio) & (performance_df['Period'] == 'Out-of-Sample')]['Sharpe Ratio'].iloc[0]
        }
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot cumulative P&L for each portfolio
    plt.plot(combined_data.index, combined_data['Best_Sharpe_Cumulative_PnL'], 
             label='Best Sharpe Portfolio', color='blue', linewidth=2)
    plt.plot(combined_data.index, combined_data['Kmeans_Cumulative_PnL'], 
             label='K-means Portfolio', color='green', linewidth=2)
    plt.plot(combined_data.index, combined_data['Hierarchy_Cumulative_PnL'], 
             label='Hierarchical Portfolio', color='purple', linewidth=2)
    
    # Highlight out-of-sample period with thicker lines
    plt.plot(combined_data.index[combined_data.index >= split_date], 
             combined_data.loc[combined_data.index >= split_date, 'Best_Sharpe_Cumulative_PnL'], 
             color='blue', linewidth=3, alpha=0.8)
    plt.plot(combined_data.index[combined_data.index >= split_date], 
             combined_data.loc[combined_data.index >= split_date, 'Kmeans_Cumulative_PnL'], 
             color='green', linewidth=3, alpha=0.8)
    plt.plot(combined_data.index[combined_data.index >= split_date], 
             combined_data.loc[combined_data.index >= split_date, 'Hierarchy_Cumulative_PnL'], 
             color='purple', linewidth=3, alpha=0.8)
    
    # Add split line and zero line
    plt.axvline(x=split_date, color='black', linestyle='--', alpha=0.7,
                label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='Break-even')
    
    # Create custom legend with in-sample and out-of-sample metrics
    custom_lines = [
        plt.Line2D([0], [0], color='blue', lw=2),
        plt.Line2D([0], [0], color='green', lw=2),
        plt.Line2D([0], [0], color='purple', lw=2),
        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.7),
        plt.Line2D([0], [0], color='gray', alpha=0.5)
    ]
    
    custom_labels = [
        f'Best Sharpe Portfolio\nIn-Sample: P&L=${metrics["Best Sharpe"]["in_sample_pnl"]:,.2f}, Sharpe={metrics["Best Sharpe"]["in_sample_sharpe"]:.2f}\nOut-of-Sample: P&L=${metrics["Best Sharpe"]["out_sample_pnl"]:,.2f}, Sharpe={metrics["Best Sharpe"]["out_sample_sharpe"]:.2f}',
        f'K-means Portfolio\nIn-Sample: P&L=${metrics["Kmeans"]["in_sample_pnl"]:,.2f}, Sharpe={metrics["Kmeans"]["in_sample_sharpe"]:.2f}\nOut-of-Sample: P&L=${metrics["Kmeans"]["out_sample_pnl"]:,.2f}, Sharpe={metrics["Kmeans"]["out_sample_sharpe"]:.2f}',
        f'Hierarchical Portfolio\nIn-Sample: P&L=${metrics["Hierarchy"]["in_sample_pnl"]:,.2f}, Sharpe={metrics["Hierarchy"]["in_sample_sharpe"]:.2f}\nOut-of-Sample: P&L=${metrics["Hierarchy"]["out_sample_pnl"]:,.2f}, Sharpe={metrics["Hierarchy"]["out_sample_sharpe"]:.2f}',
        f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)',
        'Break-even'
    ]
    
    plt.legend(custom_lines, custom_labels, loc='best', fontsize=10)
    
    # Add labels and title
    plt.title('Portfolio Performance Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative P&L ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Remove the text box (metrics are now in the legend)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Portfolio_Performance_Comparison.png'))
    plt.close()
    
    print("Portfolio performance comparison chart saved to:", 
          os.path.join(OUTPUT_DIR, 'Portfolio_Performance_Comparison.png'))

def create_bimonthly_comparison(combined_data, split_date, portfolio1='Best_Sharpe', portfolio2='Kmeans'):
    """Create bimonthly comparison between two portfolios"""
    p1_name = 'Best Sharpe' if portfolio1 == 'Best_Sharpe' else 'K-means' if portfolio1 == 'Kmeans' else 'Hierarchical'
    p2_name = 'Best Sharpe' if portfolio2 == 'Best_Sharpe' else 'K-means' if portfolio2 == 'Kmeans' else 'Hierarchical'
    
    print(f"Creating bimonthly comparison between {p1_name} and {p2_name} portfolios...")
    
    # Get out-of-sample data
    oos_data = combined_data[combined_data.index >= split_date].copy()
    
    # Add a year and bimonthly period columns for grouping
    oos_data['year'] = oos_data.index.year.astype(int)
    oos_data['bimonthly'] = ((oos_data.index.month - 1) // 2 + 1).astype(int)
    
    # Create simplified period labels (YYYY-MM)
    oos_data['period_label'] = oos_data.apply(
        lambda row: f"{int(row['year'])}-{int((row['bimonthly'] - 1) * 2 + 1):02d}",
        axis=1
    )
    
    # Create DataFrame to store bimonthly comparisons
    bimonthly_data = []
    
    # Group by bimonthly period
    for period_label, group in oos_data.groupby('period_label'):
        # Skip periods with too few trading days
        if len(group) < 10:
            continue
        
        # Get first date in period for sorting
        year, month = period_label.split('-')
        period_date = pd.Timestamp(year=int(year), month=int(month), day=15)
        
        # Calculate metrics for this period
        p1_returns = group[f'{portfolio1}_Daily_PnL']
        p2_returns = group[f'{portfolio2}_Daily_PnL']
        
        # Calculate Sharpe ratios
        p1_sharpe = p1_returns.mean() / p1_returns.std() * np.sqrt(252) if p1_returns.std() > 0 else 0
        p2_sharpe = p2_returns.mean() / p2_returns.std() * np.sqrt(252) if p2_returns.std() > 0 else 0
        
        # Calculate total returns
        p1_return = p1_returns.sum()
        p2_return = p2_returns.sum()
        
        # Create period data entry
        period_data = {
            'period_label': period_label,
            'date': period_date,
            'trading_days': len(group),
            f'{portfolio1}_sharpe': p1_sharpe,
            f'{portfolio2}_sharpe': p2_sharpe,
            f'{portfolio1}_return': p1_return,
            f'{portfolio2}_return': p2_return
        }
        
        bimonthly_data.append(period_data)
    
    # Convert to DataFrame and sort by date
    bimonthly_df = pd.DataFrame(bimonthly_data)
    if bimonthly_df.empty:
        print("No bimonthly periods found in out-of-sample data!")
        return
    
    bimonthly_df = bimonthly_df.sort_values('date')
    
    # Count periods where portfolio2 outperforms portfolio1
    outperform_count = sum(bimonthly_df[f'{portfolio2}_sharpe'] > bimonthly_df[f'{portfolio1}_sharpe'])
    total_periods = len(bimonthly_df)
    outperform_pct = (outperform_count / total_periods) * 100 if total_periods > 0 else 0
    
    # Create the visualization
    plt.figure(figsize=(14, 8))
    
    # Set up x-axis dates
    x = np.arange(len(bimonthly_df))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, bimonthly_df[f'{portfolio1}_sharpe'], width,
            label=f'{p1_name} Portfolio', color='blue')
    plt.bar(x + width/2, bimonthly_df[f'{portfolio2}_sharpe'], width,
            label=f'{p2_name} Portfolio', color='green' if portfolio2 == 'Kmeans' else 'purple')
    
    # Add horizontal line at Sharpe = 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add title and labels
    plt.title(f'Bimonthly Sharpe Ratio Comparison (Out-of-Sample Period)\n'
              f'{p2_name} Portfolio outperformed {p1_name} Portfolio {outperform_pct:.1f}% of the time',
              fontsize=14)
    plt.xlabel('Bimonthly Period', fontsize=12)
    plt.ylabel('Sharpe Ratio (Annualized)', fontsize=12)
    
    # Add x-ticks with period labels
    plt.xticks(x, bimonthly_df['period_label'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    
    # Add text box with summary
    plt.annotate(
        f'{p2_name} Portfolio Win Rate: {outperform_pct:.1f}%\n'
        f'({outperform_count} out of {total_periods} periods)\n'
        f'Out-of-Sample Period Only',
        xy=(0.7, 0.95), xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    filename = f'Bimonthly_Comparison_{portfolio1}_vs_{portfolio2}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    
    print(f"Bimonthly comparison chart saved to: {os.path.join(OUTPUT_DIR, filename)}")
    
    return bimonthly_df

def main():
    """Main function to run the portfolio comparison with warm-up period"""
    print("\n=== PORTFOLIO COMPARISON ANALYSIS (WITH WARM-UP PERIOD) ===\n")
    
    start_time = time.time()
    
    # Build portfolios with warm-up period
    portfolio_data = build_portfolios()
    if not portfolio_data:
        print("Failed to build portfolios. Exiting.")
        return
    
    # Create combined portfolio data (automatically handles warm-up period)
    combined_data, split_date = create_combined_portfolio_data(portfolio_data)
    
    # Create visualizations (using existing functions)
    plot_portfolio_comparison(combined_data, split_date)
    
    # Create bimonthly comparisons
    create_bimonthly_comparison(combined_data, split_date, 'Best_Sharpe', 'Kmeans')
    create_bimonthly_comparison(combined_data, split_date, 'Best_Sharpe', 'Hierarchy')
    
    # Calculate total runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nPortfolio comparison completed in {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()