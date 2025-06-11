import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import time
from pathlib import Path

# Import the read_ts module for data loading
import read_ts

# Import configuration
from input import *
from SMA_Strategy import SMAStrategy


def main():
    # Start overall execution timer
    overall_start_time = time.time()
    
    # Setup paths using relative directories
    SCRIPT_DIR = Path(__file__).parent.absolute()  # Get the directory where this script is located
    DATA_DIR = SCRIPT_DIR / "data"  # Use pathlib for cross-platform compatibility
    
    # Ensure data directory exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}. Please create a 'data' directory and place your data files there.")

    # Define SYMBOL based on TICKER
    SYMBOL = TICKER.replace('=F', '')  # This strips the '=F' part if present in the ticker symbol

    # Create the output directory for each symbol (only if not running in Jupyter)
    import sys
    is_jupyter = hasattr(sys, 'ps1') or any('jupyter' in arg or 'ipykernel' in arg for arg in sys.argv)
    
    if not is_jupyter:
        output_dir = SCRIPT_DIR / 'output' / SYMBOL  # Symbol-specific folder
        output_dir.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist
        
        # Function to save plots in the created folder
        def save_plot(plot_name):
            plt.savefig(output_dir / plot_name)  # Save plot to the symbol-specific folder
            plt.close()  # Close the plot to free up memory
    else:
        # In Jupyter, we'll display plots inline instead of saving them
        def save_plot(plot_name):
            plt.show()  # Display the plot inline
            plt.close()  # Close the plot to free up memory

    # Function to save parameters to a JSON file
    def save_parameters():
        """Save the parameters to a JSON file in the script directory."""
        parameters = {
            "big_point_value": big_point_value,
            "slippage": slippage,
            "capital": TRADING_CAPITAL,
            "atr_period": ATR_PERIOD
        }

        with open(SCRIPT_DIR / "parameters.json", "w") as file:
            json.dump(parameters, file)

    # Function to get slippage value from an Excel file - STRICT VERSION
    def get_slippage_from_excel(symbol, data_dir):
        """
        Get the slippage value for a specific symbol from the Excel file
        
        Parameters:
        symbol: str - The trading symbol to look up (without '=F')
        data_dir: Path - Directory containing the Excel file
        
        Returns:
        float - Slippage value for the symbol
        """
        excel_path = data_dir / "sessions_slippages.xlsx"
        
        # No fallback - if file doesn't exist, crash with helpful message
        if not excel_path.exists():
            raise FileNotFoundError(
                f"Slippage Excel file not found at {excel_path}. "
                "Please place 'sessions_slippages.xlsx' in the 'data' directory."
            )
        
        # Remove '=F' suffix if present for lookup
        lookup_symbol = symbol.replace('=F', '')
        
        try:
            # Read the Excel file
            df = pd.read_excel(excel_path)
        except Exception as e:
            raise Exception(f"Error reading Excel file: {e}. Please ensure the file is not corrupted or open in another program.")
        
        # Print the Excel contents for debugging
        print("\nContents of sessions_slippages.xlsx:")
        print(df.head())
        
        # Check if we have at least 4 columns (to access column D)
        if df.shape[1] < 4:
            raise ValueError(
                f"Excel file has fewer than 4 columns: {df.columns.tolist()}. "
                "Please ensure the file has at least 4 columns with slippage values in column D."
            )
            
        # Print column names for debugging
        print(f"Columns: {df.columns.tolist()}")
        
        # Use direct column access - Column B (index 1) for symbol, Column D (index 3) for slippage
        # First convert to uppercase for case-insensitive comparison
        df['SymbolUpper'] = df.iloc[:, 1].astype(str).str.upper()
        lookup_symbol_upper = lookup_symbol.upper()
        
        # Find the matching row
        matching_rows = df[df['SymbolUpper'] == lookup_symbol_upper]
        
        if matching_rows.empty:
            raise ValueError(
                f"Symbol '{lookup_symbol}' not found in column B of Excel file. "
                "Please ensure the symbol is correctly listed in the Excel file."
            )
            
        # Get the slippage value from column D (index 3)
        slippage_value = matching_rows.iloc[0, 3]
        
        # Validate the slippage value is numeric
        if pd.isna(slippage_value) or not isinstance(slippage_value, (int, float)):
            raise ValueError(
                f"Invalid slippage value for symbol '{lookup_symbol}': {slippage_value}. "
                "Please ensure the slippage value in column D is a valid number."
            )
            
        print(f"Found slippage for {lookup_symbol} in column D: {slippage_value}")
        return slippage_value

    # Function to find the data file for the specified futures symbol
    def find_futures_file(symbol, data_dir):
        """Find a data file for the specified futures symbol"""
        # First try a pattern that specifically looks for @SYMBOL
        pattern = f"*@{symbol}_*.dat"
        files = list(data_dir.glob(pattern))
        
        if not files:
            # Try a pattern that looks for the symbol with an underscore or at boundary
            pattern = f"*_@{symbol}_*.dat"
            files = list(data_dir.glob(pattern))
        
        if not files:
            # Try a more specific boundary pattern for the symbol
            pattern = f"*_{symbol}_*.dat"
            files = list(data_dir.glob(pattern))
        
        if not files:
            # Last resort: less specific but better than nothing
            pattern = f"*@{symbol}*.dat"
            files = list(data_dir.glob(pattern))
        
        # No fallback - if no file found, crash with helpful message
        if not files:
            raise FileNotFoundError(
                f"No data file found for {symbol} in {data_dir}. "
                "Please ensure your data files are placed in the 'data' directory "
                "and follow the naming pattern: *@SYMBOL_*.dat or *_SYMBOL_*.dat"
            )
            
        return str(files[0])  # Return the first matching file as string for compatibility

    # Get symbol from the TICKER variable (remove '=F' if it exists)
    SYMBOL = TICKER.replace('=F', '')

    # Load local data instead of downloading from Yahoo Finance
    print(f"Loading {TICKER} data from local files...")
    data_file = find_futures_file(SYMBOL, DATA_DIR)

    print(f"Found data file: {os.path.basename(data_file)}")
    print(f"File size: {os.path.getsize(data_file)} bytes")

    # Load the futures data file
    print("Loading data file...")
    load_start_time = time.time()
    all_data = read_ts.read_ts_ohlcv_dat(data_file)
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    print(f"Data loaded successfully in {load_time:.2f} seconds! Number of items: {len(all_data)}")
    
    # Extract metadata and OHLCV data from the first data object
    data_obj = all_data[0]
    tick_size = data_obj.big_point_value * data_obj.tick_size
    
    # Get big point value from data
    big_point_value = data_obj.big_point_value

    # Fetch slippage value from Excel - NO FALLBACK
    slippage_value = get_slippage_from_excel(TICKER, DATA_DIR)
    slippage = slippage_value
    print(f"Using slippage from Excel column D: {slippage}")

    # Save the parameters to a JSON file
    save_parameters()
    
    # Start timing data preparation
    prep_start_time = time.time()
    
    ohlc_data = data_obj.data.copy()  # Make a copy to avoid modifying original data
    
    # Print information about the data
    print(f"\nSymbol: {data_obj.symbol}")
    print(f"Description: {data_obj.description}")
    print(f"Exchange: {data_obj.exchange}")
    print(f"Interval: {data_obj.interval_type} {data_obj.interval_span}")
    print(f"Tick size: {tick_size}")
    print(f"Big point value: {big_point_value}")
    print(f"Data shape: {ohlc_data.shape}")
    print(f"Date range: {ohlc_data['datetime'].min()} to {ohlc_data['datetime'].max()}")
    
    # Display the first few rows of data
    print("\nFirst few rows of OHLCV data:")
    print(ohlc_data.head())
    
    # Convert the OHLCV data to the format expected by the SMA strategy
    # First, rename columns to match what yfinance provides
    data = ohlc_data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Set the datetime column as the index
    data.set_index('Date', inplace=True)
    
    # Add warm-up period for SMA calculation
    original_start_idx = None
    
    # Filter data to match the date range if specified in input.py
    if START_DATE and END_DATE:
        # Calculate warm-up period (longest SMA + buffer for ATR calculation)
        warm_up_days = SMA_MAX + ATR_PERIOD + 50  # Add buffer days for safety
        
        # Convert dates to datetime
        start_date = pd.to_datetime(START_DATE)
        end_date = pd.to_datetime(END_DATE)
        
        # Adjust start date for warm-up
        adjusted_start = start_date - pd.Timedelta(days=warm_up_days)
        
        # Load more data for warm-up
        data = data[(data.index >= adjusted_start) & 
                    (data.index <= end_date)]
        
        # Store the original start date index for later use
        if data.empty:
            raise ValueError(f"No data available for the specified date range: {START_DATE} to {END_DATE}")
            
        # Find the closest index to our original start date
        original_start_idx = data.index.get_indexer([start_date], method='nearest')[0]
        
        print(f"Loaded extended data with {warm_up_days} days warm-up period")
        print(f"Original date range: {START_DATE} to {END_DATE}")
        print(f"Adjusted date range: {adjusted_start.strftime('%Y-%m-%d')} to {END_DATE}")
        print(f"Original start index: {original_start_idx}")
    
    prep_end_time = time.time()
    prep_time = prep_end_time - prep_start_time
    print(f"Data preparation completed in {prep_time:.2f} seconds")
    
    # Define the range of SMA periods to test
    sma_range = range(SMA_MIN, SMA_MAX + 1, SMA_STEP)
    
    print(f"Optimizing SMA parameters using range from {SMA_MIN} to {SMA_MAX} with step {SMA_STEP}...")
    print(f"Trading with big point value from data: {big_point_value}")
    print(f"Using capital allocation: ${TRADING_CAPITAL:,} with ATR period: {ATR_PERIOD}")
    
    # Initialize the ATR-based strategy using the big point value from the data
    strategy = SMAStrategy(
        short_sma=0,  # Will be set during optimization
        long_sma=0,  # Will be set during optimization
        big_point_value=big_point_value,  # Use the big point value from data
        slippage=slippage,  # Use dynamically calculated slippage
        capital=TRADING_CAPITAL,  # Capital allocation for position sizing
        atr_period=ATR_PERIOD  # ATR period for position sizing
    )
    
    # Start timing the optimization process
    print("\nStarting optimization process...")
    optimization_start_time = time.time()
    
    # Run the optimization function to find the best SMA parameters
    # Save results file for data_analysis.py to use
    best_sma, best_sharpe, best_trades, all_results = strategy.optimize(
        data.copy(),
        sma_range,
        train_test_split=TRAIN_TEST_SPLIT,
        results_file='sma_all_results.txt',  # Relative path to current directory
        warm_up_idx=original_start_idx  # Pass the warm-up index to ensure consistent Sharpe calculation
    )
    
    # Calculate and display the time taken for optimization
    optimization_end_time = time.time()
    optimization_time = optimization_end_time - optimization_start_time
    print(f"\nOptimization completed in {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    
    print(f"Optimal SMA parameters: Short = {best_sma[0]} days, Long = {best_sma[1]} days")
    print(f"In-sample Sharpe ratio = {best_sharpe:.4f}")
    print(f"Number of trades with optimal parameters = {best_trades}")
    print(f"Optimization results saved to 'sma_all_results.txt' for further analysis")
    
    # Update strategy with the best parameters
    strategy.short_sma = best_sma[0]
    strategy.long_sma = best_sma[1]
    
    # Apply the best SMA parameters found from optimization to the dataset
    print("\nApplying best strategy parameters...")
    apply_start_time = time.time()
    data = strategy.apply_strategy(data.copy())
    apply_end_time = time.time()
    apply_time = apply_end_time - apply_start_time
    print(f"Strategy application completed in {apply_time:.2f} seconds")
    
    # Trim data back to the original date range for evaluation
    if original_start_idx is not None:
        print("Trimming warm-up period for final evaluation and visualization...")
        data_for_evaluation = data.iloc[original_start_idx:]
        print(f"Original data length: {len(data)}, Evaluation data length: {len(data_for_evaluation)}")
    else:
        data_for_evaluation = data
    
    # Calculate split index for in-sample/out-of-sample
    split_index = int(len(data_for_evaluation) * TRAIN_TEST_SPLIT)
    
    # Print detailed verification information about the data split
    print("\n--- DATA SPLIT VERIFICATION ---")
    print(f"Total data points after trimming: {len(data_for_evaluation)}")
    print(f"Split index: {split_index}")
    print(f"In-sample data points: {split_index}")
    print(f"Out-of-sample data points: {len(data_for_evaluation) - split_index}")
    
    # Start timing the visualization process
    viz_start_time = time.time()
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(14, 16))
    
    # Plot price and SMAs
    plt.subplot(3, 1, 1)
    plt.plot(data_for_evaluation.index, data_for_evaluation['Close'], label=f'{data_obj.symbol} Price', color='blue')
    plt.plot(data_for_evaluation.index, data_for_evaluation['SMA_Short_Strategy'], label=f'{best_sma[0]}-day SMA', color='orange')
    plt.plot(data_for_evaluation.index, data_for_evaluation['SMA_Long_Strategy'], label=f'{best_sma[1]}-day SMA', color='red')
    
    # Plot position changes (using vectorized identification of changes)
    long_entries = (data_for_evaluation['Position_Dir_Strategy'] == 1) & data_for_evaluation['Position_Change_Strategy']
    short_entries = (data_for_evaluation['Position_Dir_Strategy'] == -1) & data_for_evaluation['Position_Change_Strategy']
    
    # Plot the entries
    plt.scatter(data_for_evaluation.index[long_entries], data_for_evaluation.loc[long_entries, 'Close'], 
                color='green', marker='^', s=50, label='Long Entry')
    plt.scatter(data_for_evaluation.index[short_entries], data_for_evaluation.loc[short_entries, 'Close'], 
                color='red', marker='v', s=50, label='Short Entry')
    
    plt.legend()
    plt.title(f'{data_obj.symbol} with Optimized SMA Strategy ({best_sma[0]}, {best_sma[1]})')
    plt.grid(True)
    
    # Plot position size based on ATR with dual y-axes
    ax1 = plt.subplot(3, 1, 2)
    ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
    
    # Plot position size on the left y-axis
    ax1.plot(data_for_evaluation.index, data_for_evaluation['Position_Size_Strategy'], 
             label='Position Size (# Contracts)', color='purple')
    ax1.set_ylabel('Position Size (# Contracts)', color='purple')
    ax1.tick_params(axis='y', colors='purple')
    
    # Plot ATR on the right y-axis
    ax2.plot(data_for_evaluation.index, data_for_evaluation['ATR_Strategy'], 
             label=f'ATR ({ATR_PERIOD}-day)', color='orange')
    ax2.set_ylabel(f'ATR ({ATR_PERIOD}-day)', color='orange')
    ax2.tick_params(axis='y', colors='orange')
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Position Sizing Based on {ATR_PERIOD}-day ATR')
    ax1.grid(True)
    
    # Plot the performance (P&L)
    plt.subplot(3, 1, 3)
    
    # Reset P&L to zero at start of evaluation period for cleaner visualization
    strategy_pnl_cumulative = data_for_evaluation['Cumulative_PnL_Strategy'] - data_for_evaluation['Cumulative_PnL_Strategy'].iloc[0]
    
    # Plot the cumulative P&L of the strategy (removed market P&L)
    plt.plot(data_for_evaluation.index, strategy_pnl_cumulative, 
             label='Strategy P&L (full period)', color='green')
    
    # Highlight out-of-sample period
    plt.plot(data_for_evaluation.index[split_index:], strategy_pnl_cumulative.iloc[split_index:],
            label=f'Strategy P&L (last {int((1 - TRAIN_TEST_SPLIT) * 100)}% out-of-sample)', color='purple')
    
    # Add split line and zero line
    plt.axvline(x=data_for_evaluation.index[split_index], color='black', linestyle='--',
                label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
    plt.axhline(y=0.0, color='gray', linestyle='-', label='Break-even')
    
    plt.legend()
    plt.title('Strategy Performance (Dollar P&L)')
    plt.ylabel('P&L ($)')
    plt.grid(True)
    
    plt.tight_layout()
    save_plot('Optimized_Strategy_Plot.png')
    
    viz_end_time = time.time()
    viz_time = viz_end_time - viz_start_time
    print(f"Visualization completed in {viz_time:.2f} seconds")
    
    # Calculate performance metrics
    metrics_start_time = time.time()
    metrics = strategy.calculate_performance_metrics(
        data_for_evaluation,  # Use the trimmed data for metrics
        strategy_name="Strategy",
        train_test_split=TRAIN_TEST_SPLIT
    )
    metrics_end_time = time.time()
    metrics_time = metrics_end_time - metrics_start_time
    print(f"Performance metrics calculation completed in {metrics_time:.2f} seconds")
    
    # Calculate market performance for comparison (for reporting only, not plotting)
    market_cumulative_pnl = data_for_evaluation['Market_PnL_Strategy'].cumsum().iloc[-1]
    
    # Print summary statistics
    print("\n--- PERFORMANCE SUMMARY OF ATR-BASED SMA STRATEGY ---")
    print(f"Symbol: {data_obj.symbol}")
    print(f"Big Point Value (from data): {big_point_value}")
    print(f"ATR Period for Position Sizing: {ATR_PERIOD} days")
    print(f"Capital Allocation: ${TRADING_CAPITAL:,}")
    print(f"Average Position Size: {metrics['avg_position_size']:.2f} contracts")
    print(f"Maximum Position Size: {metrics['max_position_size']:.0f} contracts")
    print(f"Strategy Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"Market Buy & Hold P&L: ${market_cumulative_pnl:,.2f}")
    print(f"Outperformance: ${(metrics['total_pnl'] - market_cumulative_pnl):,.2f}")
    
    # *** SHARPE RATIO VERIFICATION ***
    print("\n--- SHARPE RATIO COMPARISON VERIFICATION ---")
    print(f"Optimization in-sample Sharpe ratio: {best_sharpe:.6f}")
    print(f"Final in-sample Sharpe ratio: {metrics['sharpe_in_sample']:.6f}")
    print(f"Difference: {abs(best_sharpe - metrics['sharpe_in_sample']):.6f}")
    if abs(best_sharpe - metrics['sharpe_in_sample']) < 0.001:
        print("✓ SHARPE RATIOS MATCH (within 0.001 tolerance)")
    else:
        print("✗ SHARPE RATIOS DO NOT MATCH")
    
    print(f"Sharpe ratio (entire period, annualized): {metrics['sharpe_full']:.4f}")
    print(f"Sharpe ratio (in-sample, annualized): {metrics['sharpe_in_sample']:.4f}")
    print(f"Sharpe ratio (out-of-sample, annualized): {metrics['sharpe_out_sample']:.4f}")
    print(f"Maximum Drawdown: ${abs(metrics['max_drawdown_dollars']):,.2f}")
    print("\n--- TRADE COUNT SUMMARY ---")
    print(f"In-sample period trades: {metrics['in_sample_trades']}")
    print(f"Out-of-sample period trades: {metrics['out_sample_trades']}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"In-sample P&L: ${metrics['in_sample_pnl']:,.2f}")
    print(f"Out-of-sample P&L: ${metrics['out_sample_pnl']:,.2f}")
    
    print(f"\nBest parameters: Short SMA = {best_sma[0]}, Long SMA = {best_sma[1]}, Sharpe = {best_sharpe:.6f}, Trades = {best_trades}")
    
    # Calculate overall execution time
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    # Print timing summary
    print("\n--- EXECUTION TIME SUMMARY (Vectorized Implementation) ---")
    print(f"Data loading time: {load_time:.2f} seconds")
    print(f"Data preparation time: {prep_time:.2f} seconds")
    print(f"Optimization time: {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    print(f"Strategy application time: {apply_time:.2f} seconds")
    print(f"Visualization time: {viz_time:.2f} seconds")
    print(f"Metrics calculation time: {metrics_time:.2f} seconds")
    print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()