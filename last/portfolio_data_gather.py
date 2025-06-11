import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json

# Import the read_ts module for data loading
import read_ts

# Import configuration
from input import *
from PortfolioSMAStrategy import PortfolioSMAStrategy


def main():
    # Setup paths
    WORKING_DIR = r"D:\dziug\Documents\darbas\last"
    DATA_DIR = os.path.join(WORKING_DIR, "data")
    
    # List of underlyings to include in the portfolio
    underlyings = ['ES=F', 'SI=F', 'GC=F', 'CC']
    
    # Create the output directory for portfolio
    output_dir = os.path.join(WORKING_DIR, 'output', 'PORTFOLIO')
    os.makedirs(output_dir, exist_ok=True)
    
    # Function to save plots in the created folder
    def save_plot(plot_name):
        plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()
    
    # Function to save parameters to a JSON file
    def save_parameters(contract_multipliers, dynamic_slippages):
        """Save the contract_multipliers and dynamic_slippages to a JSON file."""
        parameters = {
            "contract_multipliers": contract_multipliers,
            "dynamic_slippages": dynamic_slippages
        }

        with open(os.path.join(output_dir, "portfolio_parameters.json"), "w") as file:
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
    
    # Initialize dictionaries to store data, contract multipliers, and slippages
    data_dict = {}
    contract_multipliers = {}
    dynamic_slippages = {}
    
    # Load data for each underlying
    for ticker in underlyings:
        # Get symbol from the ticker (remove '=F' if it exists)
        symbol = ticker.replace('=F', '')
        
        print(f"\n=== Loading {ticker} data from local files... ===")
        data_file = find_futures_file(symbol, DATA_DIR)
        
        if not data_file:
            print(f"Error: No data file found for {ticker} in {DATA_DIR}")
            print("Available files:")
            all_files = glob.glob(os.path.join(DATA_DIR, "*.dat"))
            for file in all_files:
                print(f"  - {os.path.basename(file)}")
            continue
        
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
            contract_multipliers[ticker] = contract_multiplier
            
            # Fetch slippage value from Excel
            slippage_value = get_slippage_from_excel(ticker, DATA_DIR)
            
            # Calculate dynamic slippage based on contract multiplier
            if slippage_value is not None:
                dynamic_slippage = slippage_value / contract_multiplier
                print(f"Using slippage from Excel column D: {slippage_value} adjusted by contract multiplier: {dynamic_slippage}")
            else:
                # Fallback to the original formula if not found in Excel
                dynamic_slippage = 14 / contract_multiplier
                print(f"Slippage not found in Excel, using default value: 14 adjusted by contract multiplier: {dynamic_slippage}")
            
            dynamic_slippages[ticker] = dynamic_slippage
            
            # Extract OHLC data
            ohlc_data = data_obj.data.copy()
            
            # Print information about the data
            print(f"\nSymbol: {data_obj.symbol}")
            print(f"Description: {data_obj.description}")
            print(f"Exchange: {data_obj.exchange}")
            print(f"Interval: {data_obj.interval_type} {data_obj.interval_span}")
            print(f"Tick size: {tick_size}")
            print(f"Big point value (Contract Multiplier): {contract_multiplier}")
            print(f"Data shape: {ohlc_data.shape}")
            print(f"Date range: {ohlc_data['datetime'].min()} to {ohlc_data['datetime'].max()}")
            
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
            
            # Store the prepared data
            data_dict[ticker] = data
            
        except Exception as e:
            print(f"Error processing data for {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save the contract multipliers and dynamic slippages
    save_parameters(contract_multipliers, dynamic_slippages)
    
    # Check if we were able to load data for at least one instrument
    if not data_dict:
        print("Error: Could not load data for any of the instruments in the portfolio!")
        exit(1)
    
    print(f"\n=== Successfully loaded data for {len(data_dict)} instruments ===")
    
    # Define the range of SMA periods to test
    sma_range = range(SMA_MIN, SMA_MAX + 1, SMA_STEP)
    
    print(f"Optimizing SMA parameters for portfolio using range from {SMA_MIN} to {SMA_MAX} with step {SMA_STEP}...")
    
    # Initialize the portfolio strategy with empty parameters (will be set during optimization)
    portfolio_strategy = PortfolioSMAStrategy(
        instruments={},  # Will be populated during optimization
        contract_sizes=contract_multipliers,
        slippages=dynamic_slippages
    )
    
    # Run the optimization to find the best SMA parameters for each instrument
    best_params, best_sharpe, best_trades, all_results = portfolio_strategy.optimize(
        data_dict,
        sma_range,
        train_test_split=TRAIN_TEST_SPLIT,
        initial_capital=INITIAL_CAPITAL,
        results_file=os.path.join(output_dir, 'portfolio_sma_all_results.txt')
    )
    
    print(f"\nOptimal SMA parameters found for each instrument:")
    for symbol, params in best_params.items():
        print(f"{symbol}: Short = {params['short_sma']} days, Long = {params['long_sma']} days")
    print(f"Portfolio in-sample Sharpe ratio = {best_sharpe:.4f}")
    print(f"Optimization results saved to '{os.path.join(output_dir, 'portfolio_sma_all_results.txt')}' for further analysis")
    
    # Update the portfolio strategy with the best parameters
    portfolio_strategy = PortfolioSMAStrategy(
        instruments=best_params,
        contract_sizes=contract_multipliers,
        slippages=dynamic_slippages
    )
    
    # Apply the best parameters to the portfolio
    combined_data, instrument_data = portfolio_strategy.apply_strategy(
        data_dict.copy(),
        initial_capital=INITIAL_CAPITAL
    )
    
    # Calculate split index for in-sample/out-of-sample
    split_index = int(len(combined_data) * TRAIN_TEST_SPLIT)
    
    # Save the combined performance data for analysis
    combined_data.to_csv(os.path.join(output_dir, 'portfolio_performance.csv'))
    
    # Plot the portfolio performance
    plt.figure(figsize=(14, 10))
    
    # Skip the price/SMA chart since we have multiple instruments
    # Instead, focus on the performance metrics
    
    # Plot the performance
    # Plot the cumulative returns
    plt.plot(combined_data.index, combined_data['Cumulative_Returns_Strategy'], 
             label='Cumulative Returns (full period)', color='green')
    plt.plot(combined_data.index[split_index:], combined_data['Cumulative_Returns_Strategy'].iloc[split_index:],
            label=f'Cumulative Returns (last {int((1 - TRAIN_TEST_SPLIT) * 100)}% out-of-sample)', color='purple')
    plt.axvline(x=combined_data.index[split_index], color='black', linestyle='--',
                label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
    plt.axhline(y=1.0, color='gray', linestyle='-', label=f'Initial Capital (${INITIAL_CAPITAL:,})')
    plt.legend()
    plt.title('Portfolio Strategy Performance (Multiple of Initial Capital)')
    plt.ylabel('Cumulative Return (x initial)')
    
    plt.tight_layout()
    save_plot('Portfolio_Strategy_Performance.png')
    
    # Calculate overall performance metrics
    metrics = portfolio_strategy.calculate_performance_metrics(
        combined_data,
        instrument_data,
        strategy_name="Strategy",
        train_test_split=TRAIN_TEST_SPLIT,
        initial_capital=INITIAL_CAPITAL
    )
    
    # Print summary statistics
    print("\n--- PERFORMANCE SUMMARY OF PORTFOLIO SMA STRATEGY ---")
    print(f"Number of Instruments: {len(data_dict)}")
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
    
    print("\n--- INSTRUMENT BREAKDOWN ---")
    for symbol, instrument_metrics in metrics['instruments'].items():
        print(f"{symbol}: PnL=${instrument_metrics['pnl']:,.2f}, Trades={instrument_metrics['total_trades']}")
    
    # Save metrics to JSON file
    with open(os.path.join(output_dir, 'portfolio_metrics.json'), 'w') as f:
        # Convert non-serializable objects to strings or standard Python types
        serializable_metrics = {}
        
        # Process top-level metrics
        for key, value in metrics.items():
            if key == 'split_date':
                serializable_metrics[key] = str(value)
            elif key == 'instruments':
                # Handle instruments separately
                continue
            elif hasattr(value, 'dtype'):  # Handle numpy values
                if np.issubdtype(value.dtype, np.integer):
                    serializable_metrics[key] = int(value)
                elif np.issubdtype(value.dtype, np.floating):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            else:
                serializable_metrics[key] = value
        
        # Process instruments metrics
        serializable_metrics['instruments'] = {}
        for symbol, instr_metrics in metrics['instruments'].items():
            serializable_metrics['instruments'][symbol] = {}
            for key, value in instr_metrics.items():
                if hasattr(value, 'dtype'):  # Handle numpy values
                    if np.issubdtype(value.dtype, np.integer):
                        serializable_metrics['instruments'][symbol][key] = int(value)
                    elif np.issubdtype(value.dtype, np.floating):
                        serializable_metrics['instruments'][symbol][key] = float(value)
                    else:
                        serializable_metrics['instruments'][symbol][key] = value
                else:
                    serializable_metrics['instruments'][symbol][key] = value
        
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"\nPerformance metrics saved to '{os.path.join(output_dir, 'portfolio_metrics.json')}'")
    print("\nPortfolio analysis complete!")

if __name__ == "__main__":
    main()