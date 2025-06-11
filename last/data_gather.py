import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json  # Import json for saving parameters

# Import the read_ts module for data loading
import read_ts

# Import configuration
from input import *
from SMA_Strategy import SMAStrategy


def main():
    # Setup paths
    WORKING_DIR = r"D:\dziug\Documents\darbas\last"
    DATA_DIR = os.path.join(WORKING_DIR, "data")


    # Define SYMBOL based on TICKER
    SYMBOL = TICKER.replace('=F', '')  # This strips the '=F' part if present in the ticker symbol

    # Create the output directory for each symbol
    output_dir = os.path.join(WORKING_DIR, 'output', SYMBOL)  # Symbol-specific folder
    os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

    # Function to save plots in the created folder
    def save_plot(plot_name):
        plt.savefig(os.path.join(output_dir, plot_name))  # Save plot to the symbol-specific folder
        plt.close()  # Close the plot to free up memory

    # Function to save parameters to a JSON file
    def save_parameters():
        """Save the contract_multiplier and dynamic_slippage to a JSON file."""
        parameters = {
            "contract_multiplier": contract_multiplier,
            "dynamic_slippage": dynamic_slippage
        }

        with open("parameters.json", "w") as file:
            json.dump(parameters, file)

    # Function to get slippage value from an Excel file
    def get_slippage_from_excel(symbol, data_dir):
        """
        Get the slippage value for a specific symbol from the Excel file
        
        Parameters:
        symbol: str - The trading symbol to look up (without '=F')
        data_dir: str - Directory containing the Excel file
        
        Returns:
        float - Slippage value for the symbol, or None if not found
        """
        excel_path = os.path.join(data_dir, "sessions_slippages.xlsx")
        
        if not os.path.exists(excel_path):
            print(f"Warning: Slippage Excel file not found at {excel_path}")
            return None
        
        try:
            # Remove '=F' suffix if present for lookup
            lookup_symbol = symbol.replace('=F', '')
            
            # Read the Excel file
            df = pd.read_excel(excel_path)
            
            # Print the Excel contents for debugging
            print("\nContents of sessions_slippages.xlsx:")
            print(df.head())
            
            # Check if we have at least 4 columns (to access column D)
            if df.shape[1] < 4:
                print(f"Warning: Excel file has fewer than 4 columns: {df.columns.tolist()}")
                return None
                
            # Print column names for debugging
            print(f"Columns: {df.columns.tolist()}")
            
            # Use direct column access - Column B (index 1) for symbol, Column D (index 3) for slippage
            # First convert to uppercase for case-insensitive comparison
            df['SymbolUpper'] = df.iloc[:, 1].astype(str).str.upper()
            lookup_symbol_upper = lookup_symbol.upper()
            
            # Find the matching row
            matching_rows = df[df['SymbolUpper'] == lookup_symbol_upper]
            
            if not matching_rows.empty:
                # Get the slippage value from column D (index 3)
                slippage_value = matching_rows.iloc[0, 3]
                print(f"Found slippage for {lookup_symbol} in column D: {slippage_value}")
                return slippage_value
            
            print(f"Warning: Symbol '{lookup_symbol}' not found in column B of Excel file")
            return None
            
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Function to find the data file for the specified futures symbol
    def find_futures_file(symbol, data_dir):
        """Find a data file for the specified futures symbol"""
        # First try a pattern that specifically looks for @SYMBOL
        pattern = f"*@{symbol}_*.dat"
        files = glob.glob(os.path.join(data_dir, pattern))
        
        if not files:
            # Try a pattern that looks for the symbol with an underscore or at boundary
            pattern = f"*_@{symbol}_*.dat"
            files = glob.glob(os.path.join(data_dir, pattern))
        
        if not files:
            # Try a more specific boundary pattern for the symbol
            pattern = f"*_{symbol}_*.dat"
            files = glob.glob(os.path.join(data_dir, pattern))
        
        if not files:
            # Last resort: less specific but better than nothing
            pattern = f"*@{symbol}*.dat"
            files = glob.glob(os.path.join(data_dir, pattern))
        
        return files[0] if files else None

    # Get symbol from the TICKER variable (remove '=F' if it exists)
    SYMBOL = TICKER.replace('=F', '')

    # Load local data instead of downloading from Yahoo Finance
    print(f"Loading {TICKER} data from local files...")
    data_file = find_futures_file(SYMBOL, DATA_DIR)

    if not data_file:
        print(f"Error: No data file found for {TICKER} in {DATA_DIR}")
        print("Available files:")
        all_files = glob.glob(os.path.join(DATA_DIR, "*.dat"))
        for file in all_files:
            print(f"  - {os.path.basename(file)}")
        exit(1)

    print(f"Found data file: {os.path.basename(data_file)}")
    print(f"File size: {os.path.getsize(data_file)} bytes")

    try:
        # Load the futures data file
        print("Loading data file...")
        all_data = read_ts.read_ts_ohlcv_dat(data_file)
        
        print(f"Data loaded successfully! Number of items: {len(all_data)}")
        
        # Extract metadata and OHLCV data from the first data object
        data_obj = all_data[0]
        tick_size = data_obj.big_point_value * data_obj.tick_size
        
        # Get contract multiplier from the data file's big_point_value
        contract_multiplier = data_obj.big_point_value

        # Fetch slippage value from Excel
        slippage_value = get_slippage_from_excel(TICKER, DATA_DIR)

        # Calculate dynamic slippage based on contract multiplier
        if slippage_value is not None:
            dynamic_slippage = slippage_value / contract_multiplier
            print(f"Using slippage from Excel column D: {slippage_value} adjusted by contract multiplier: {dynamic_slippage}")
        else:
            # Fallback to the original formula if not found in Excel
            dynamic_slippage = 14 / contract_multiplier
            print(f"Slippage not found in Excel, using default value: 14 adjusted by contract multiplier: {dynamic_slippage}")

        # Save the parameters to a JSON file
        save_parameters()

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()


    # Get symbol from the TICKER variable (remove '=F' if it exists)
    SYMBOL = TICKER.replace('=F', '')

    # Load local data instead of downloading from Yahoo Finance
    print(f"Loading {TICKER} data from local files...")
    data_file = find_futures_file(SYMBOL, DATA_DIR)

    if not data_file:
        print(f"Error: No data file found for {TICKER} in {DATA_DIR}")
        print("Available files:")
        all_files = glob.glob(os.path.join(DATA_DIR, "*.dat"))
        for file in all_files:
            print(f"  - {os.path.basename(file)}")
        exit(1)

    print(f"Found data file: {os.path.basename(data_file)}")
    print(f"File size: {os.path.getsize(data_file)} bytes")

    try:
        # Load the futures data file
        print("Loading data file...")
        all_data = read_ts.read_ts_ohlcv_dat(data_file)
        
        print(f"Data loaded successfully! Number of items: {len(all_data)}")
        
        # Extract metadata and OHLCV data from the first data object
        data_obj = all_data[0]
        tick_size = data_obj.big_point_value * data_obj.tick_size
        
        # Get contract multiplier from the data file's big_point_value
        contract_multiplier = data_obj.big_point_value
        
        ohlc_data = data_obj.data.copy()  # Make a copy to avoid modifying original data
        
        # Print information about the data
        print(f"\nSymbol: {data_obj.symbol}")
        print(f"Description: {data_obj.description}")
        print(f"Exchange: {data_obj.exchange}")
        print(f"Interval: {data_obj.interval_type} {data_obj.interval_span}")
        print(f"Tick size: {tick_size}")
        print(f"Big point value (Contract Multiplier): {contract_multiplier}")
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
        
        # Filter data to match the date range if specified in input.py
        if START_DATE and END_DATE:
            data = data[(data.index >= pd.to_datetime(START_DATE)) & 
                        (data.index <= pd.to_datetime(END_DATE))]
            print(f"Filtered data to date range: {START_DATE} to {END_DATE}")
        
        # Define the range of SMA periods to test
        sma_range = range(SMA_MIN, SMA_MAX + 1, SMA_STEP)
        
        print(f"Optimizing SMA parameters using range from {SMA_MIN} to {SMA_MAX} with step {SMA_STEP}...")
        print(f"Trading with contract multiplier from data: {contract_multiplier}")
        
        # Look up slippage in Excel file (pass TICKER directly to handle symbol properly)
        slippage_value = get_slippage_from_excel(TICKER, DATA_DIR)

        # If slippage was found in Excel, use it; otherwise fall back to calculation
        if slippage_value is not None:
            dynamic_slippage = slippage_value / contract_multiplier
            print(f"Using slippage from Excel column D: {slippage_value} adjusted by contract multiplier: {dynamic_slippage}")
        else:
            # Fallback to the original formula if not found in Excel
            dynamic_slippage = 14 / contract_multiplier
            print(f"Slippage not found in Excel, using default value: 14 adjusted by contract multiplier: {dynamic_slippage}")
            
        
        # Initialize the strategy using the contract multiplier from the data
        strategy = SMAStrategy(
            short_sma=0,  # Will be set during optimization
            long_sma=0,  # Will be set during optimization
            contract_size=contract_multiplier,  # Use the actual contract multiplier from data
            slippage=dynamic_slippage  # Use dynamically calculated slippage
        )
        
        # Run the optimization function to find the best SMA parameters
        # Save only the results file for data_analysis.py to use
        best_sma, best_sharpe, best_trades, all_results = strategy.optimize(
            data.copy(),
            sma_range,
            train_test_split=TRAIN_TEST_SPLIT,
            initial_capital=INITIAL_CAPITAL,
            results_file=r'D:\dziug\Documents\darbas\last\sma_all_results.txt'
                # Save results file for data_analysis.py
        )
        
        print(f"Optimal SMA parameters: Short = {best_sma[0]} days, Long = {best_sma[1]} days")
        print(f"In-sample Sharpe ratio = {best_sharpe:.4f}")
        print(f"Number of trades with optimal parameters = {best_trades}")
        print(f"Optimization results saved to 'sma_all_results.txt' for further analysis")
        
        # Update strategy with the best parameters
        strategy.short_sma = best_sma[0]
        strategy.long_sma = best_sma[1]
        
        # Apply the best SMA parameters found from optimization to the dataset
        data = strategy.apply_strategy(data.copy(), initial_capital=INITIAL_CAPITAL)
        
        # Calculate split index for in-sample/out-of-sample
        split_index = int(len(data) * TRAIN_TEST_SPLIT)
        
        # Plot the price and SMA lines
        plt.figure(figsize=(14, 10))
        
        # Plot price and SMAs
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label=f'{data_obj.symbol} Price', color='blue')
        plt.plot(data.index, data['SMA_Short_Strategy'], label=f'{best_sma[0]}-day SMA', color='orange')
        plt.plot(data.index, data['SMA_Long_Strategy'], label=f'{best_sma[1]}-day SMA', color='red')
        
        # Plot long position entries
        long_entries = (data['Position_Strategy'] == 1) & (data['Position_Strategy'].shift(1) != 1)
        plt.scatter(data.index[long_entries], data['Entry_Price_Strategy'][long_entries],
                    color='green', marker='^', s=50, label='Long Entry')
        
        # Plot short position entries
        short_entries = (data['Position_Strategy'] == -1) & (data['Position_Strategy'].shift(1) != -1)
        plt.scatter(data.index[short_entries], data['Entry_Price_Strategy'][short_entries],
                    color='red', marker='v', s=50, label='Short Entry')
        
        plt.legend()
        plt.title(f'{data_obj.symbol} with Optimized SMA Strategy ({best_sma[0]}, {best_sma[1]})')
        
        # Plot the performance
        plt.subplot(2, 1, 2)
        
        # Plot the cumulative returns
        plt.plot(data.index, data['Cumulative_Returns_Strategy'], label='Cumulative Returns (full period)', color='green')
        plt.plot(data.index[split_index:], data['Cumulative_Returns_Strategy'].iloc[split_index:],
                label=f'Cumulative Returns (last {int((1 - TRAIN_TEST_SPLIT) * 100)}% out-of-sample)', color='purple')
        plt.axvline(x=data.index[split_index], color='black', linestyle='--',
                    label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
        plt.axhline(y=1.0, color='gray', linestyle='-', label=f'Initial Capital (${INITIAL_CAPITAL:,})')
        plt.legend()
        plt.title('Strategy Performance (Multiple of Initial Capital)')
        plt.ylabel('Cumulative Return (x initial)')
        
        plt.tight_layout()
        save_plot('Optimized_Strategy_Plot.png')
        
        # Calculate performance metrics
        metrics = strategy.calculate_performance_metrics(
            data,
            strategy_name="Strategy",
            train_test_split=TRAIN_TEST_SPLIT,
            initial_capital=INITIAL_CAPITAL
        )
        
        # Print summary statistics
        print("\n--- PERFORMANCE SUMMARY OF SMA STRATEGY ---")
        print(f"Symbol: {data_obj.symbol}")
        print(f"Contract Multiplier (from data): {contract_multiplier}")
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Total Profit/Loss: ${metrics['total_return_dollars']:,.2f} ({metrics['total_return_percent']:.2f}%)")
        print(f"Final Sharpe ratio (entire period, annualized): {metrics['sharpe_full']:.4f}")
        print(f"Sharpe ratio (in-sample, annualized): {metrics['sharpe_in_sample']:.4f}")
        print(f"Sharpe ratio (out-of-sample, annualized): {metrics['sharpe_out_sample']:.4f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print("\n--- TRADE COUNT SUMMARY ---")
        print(f"In-sample period trades: {metrics['in_sample_trades']}")
        print(f"Out-of-sample period trades: {metrics['out_sample_trades']}")
        print(f"Total trades: {metrics['total_trades']}")
        
        print(f"\nBest parameters: Short SMA = {best_sma[0]}, Long SMA = {best_sma[1]}, Sharpe = {best_sharpe:.6f}, Trades = {best_trades}")
        print("\nAnalysis complete!")

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
