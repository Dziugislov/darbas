import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import time
import random
import datetime
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# Import the read_ts module for data loading
import read_ts

# Import configuration
from input_gen import *
from SMA_Strategy import SMAStrategy

# Global variables for caching and statistics
evaluated_combinations = {}
evaluation_counter = 0
cache_hit_counter = 0

def main():
    # Start overall execution timer
    overall_start_time = time.time()
    
    # Setup paths using relative directories
    WORKING_DIR = "."  # Current directory
    DATA_DIR = os.path.join(WORKING_DIR, "data")

    # Define SYMBOL based on TICKER
    SYMBOL = TICKER.replace('=F', '')  # This strips the '=F' part if present in the ticker symbol

    # Create the output directory for each symbol
    output_dir = os.path.join(WORKING_DIR, 'output2', SYMBOL)  # Symbol-specific folder
    os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

    # Function to save plots in the created folder
    def save_plot(plot_name):
        plt.savefig(os.path.join(output_dir, plot_name))  # Save plot to the symbol-specific folder
        plt.close()  # Close the plot to free up memory

    # Function to save parameters to a JSON file
    def save_parameters():
        """Save the big_point_value and dynamic_slippage to a JSON file."""
        parameters = {
            "big_point_value": big_point_value,
            "slippage": slippage,
            "capital": TRADING_CAPITAL,
            "atr_period": ATR_PERIOD
        }

        with open("parameters.json", "w") as file:
            json.dump(parameters, file)

    # Function to get slippage value from an Excel file - STRICT VERSION
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
        
        # No fallback - if no file found, crash
        if not files:
            raise FileNotFoundError(f"No data file found for {symbol} in {data_dir}")
            
        return files[0]  # Return the first matching file

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
        # Calculate warm-up period (longest possible SMA + buffer for ATR calculation)
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
    
    prep_end_time = time.time()
    prep_time = prep_end_time - prep_start_time
    print(f"Data preparation completed in {prep_time:.2f} seconds")
    
    # Initialize the ATR-based strategy
    strategy = SMAStrategy(
        short_sma=0,  # Will be set during optimization
        long_sma=0,  # Will be set during optimization
        big_point_value=big_point_value,  # Use the big point value from data
        slippage=slippage,  # Use dynamically calculated slippage
        capital=TRADING_CAPITAL,  # Capital allocation for position sizing
        atr_period=ATR_PERIOD  # ATR period for position sizing
    )
    
    # Set up the genetic algorithm for SMA optimization
    print("\nStarting genetic algorithm optimization...")
    print(f"Population size: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
    print(f"Using genetic algorithm parameters from input.py")
    optimization_start_time = time.time()
    
    # Fix random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # DEAP GA Setup
    # Define parameters and their ranges
    PARAM_NAMES = ["short_sma", "long_sma"]
    
    # Define fitness function for the GA
    def evaluate_individual(individual):
        """
        Evaluate an individual (a set of SMA parameters).
        Returns only the Sharpe ratio for optimization.
        """
        global evaluation_counter, cache_hit_counter
        
        short_sma, long_sma = individual
        key = (short_sma, long_sma)
        
        # Quick return if we've seen this combination before
        if key in evaluated_combinations:
            cache_hit_counter += 1
            return evaluated_combinations[key]
        
        evaluation_counter += 1
        
        # Skip invalid combinations where short SMA >= long SMA
        if short_sma >= long_sma:
            evaluated_combinations[key] = (-999999.0,)
            return -999999.0,  # Very low fitness for invalid combinations
        
        # Copy strategy instance to avoid modifying the original
        ga_strategy = SMAStrategy(
            short_sma=short_sma,
            long_sma=long_sma,
            big_point_value=big_point_value,
            slippage=slippage,
            capital=TRADING_CAPITAL,
            atr_period=ATR_PERIOD
        )
        
        # Apply the strategy with these parameters
        temp_data = data.copy()
        temp_data = ga_strategy.apply_strategy(temp_data)
        
        # Create evaluation slice (without warm-up period)
        if original_start_idx is not None:
            eval_data = temp_data.iloc[original_start_idx:].copy()
        else:
            eval_data = temp_data.copy()
        
        # Calculate training set metrics only (for optimization)
        split_idx = int(len(eval_data) * TRAIN_TEST_SPLIT)
        train_data = eval_data.iloc[:split_idx]
        
        # Calculate Sharpe ratio on training data
        if 'Daily_PnL_Strategy' in train_data.columns and len(train_data) > 0:
            daily_returns = train_data['Daily_PnL_Strategy']
            if daily_returns.sum() != 0 and daily_returns.std() != 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
                trade_count = train_data['Position_Change_Strategy'].sum()
                
                # Only consider strategies with a minimum number of trades
                if trade_count < MIN_TRADES:
                    result = (-999999.0,)
                elif trade_count > MAX_TRADES:
                    result = (-999999.0,)
                else:
                    result = (sharpe,)
            else:
                result = (-999999.0,)
        else:
            result = (-999999.0,)
            
        # Cache the result before returning
        evaluated_combinations[key] = result
        return result
            
    # Custom mutation that ensures short_sma < long_sma and uses step size
    def custom_mutation(individual, indpb):
        # Get valid values based on step size
        valid_values = list(range(SMA_MIN, SMA_MAX + 1, 2))#FIXME
        
        for i in range(len(individual)):
            if random.random() < indpb:
                if i == 0:  # short_sma
                    # Find current index in valid_values
                    current_idx = valid_values.index(individual[0]) if individual[0] in valid_values else 0
                    # Find the index of long_sma
                    long_idx = valid_values.index(individual[1]) if individual[1] in valid_values else len(valid_values) - 1
                    
                    # Choose a new index that's less than long_idx
                    new_idx = random.randint(0, long_idx - 1)
                    individual[i] = valid_values[new_idx]
                else:  # long_sma
                    # Find current index in valid_values
                    current_idx = valid_values.index(individual[0]) if individual[0] in valid_values else 0
                    
                    # Choose a new index that's greater than short_sma index
                    new_idx = random.randint(current_idx + 1, len(valid_values) - 1)
                    individual[i] = valid_values[new_idx]
        return individual,
        
    # Custom crossover that ensures short_sma < long_sma and uses step size
    def custom_crossover(ind1, ind2):
        valid_values = list(range(SMA_MIN, SMA_MAX + 1, 2))
        
        # Make copies to avoid modifying originals
        c1, c2 = list(ind1), list(ind2)
        
        # Perform crossover on copies
        if random.random() < CROSSOVER_PROB:
            # Choose which parameters to swap
            swap_short = random.random() < 0.5
            swap_long = random.random() < 0.5
            
            # Swap parameters
            if swap_short:
                c1[0], c2[0] = c2[0], c1[0]
            if swap_long:
                c1[1], c2[1] = c2[1], c1[1]
                
            # Fix constraints if violated
            if c1[0] >= c1[1]:
                # Find indices in valid_values
                short_idx = valid_values.index(c1[0]) if c1[0] in valid_values else 0
                
                # Choose new long_sma that's greater than short_sma
                new_long_idx = random.randint(short_idx + 1, len(valid_values) - 1)
                c1[1] = valid_values[new_long_idx]
                    
            if c2[0] >= c2[1]:
                # Find indices in valid_values
                short_idx = valid_values.index(c2[0]) if c2[0] in valid_values else 0
                
                # Choose new long_sma that's greater than short_sma
                new_long_idx = random.randint(short_idx + 1, len(valid_values) - 1)
                c2[1] = valid_values[new_long_idx]
        
        # Create new individuals
        return creator.Individual(c1), creator.Individual(c2)
    
    # Clear any existing DEAP types (in case of re-running)
    if 'FitnessMax' in dir(creator):
        del creator.FitnessMax
    if 'Individual' in dir(creator):
        del creator.Individual
    
    # Create fitness and individual types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Single objective: maximize Sharpe
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Initialize toolbox
    toolbox = base.Toolbox()
    
    # Custom individual creation that ensures short_sma < long_sma and uses step size
    def create_valid_individual():
        """
        Create an individual with parameters that are multiples of the step size (2).
        Ensures short_sma < long_sma.
        """
        # Calculate valid values based on step size
        valid_values = list(range(SMA_MIN, SMA_MAX + 1, 2))
        
        # Get a random value for short_sma from the valid values (excluding the last one)
        short_idx = random.randint(0, len(valid_values) - 2)
        short_sma = valid_values[short_idx]
        
        # Get a random value for long_sma that's greater than short_sma
        long_idx = random.randint(short_idx + 1, len(valid_values) - 1)
        long_sma = valid_values[long_idx]
        
        return creator.Individual([short_sma, long_sma])
    
    # Register operators
    toolbox.register("individual", create_valid_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, indpb=MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create hall of fame to store best individuals
    hall_of_fame = tools.HallOfFame(maxsize=HALL_OF_FAME_SIZE)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Initialize population
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Debug: Check the unique values in the initial population
    short_sma_values = set([ind[0] for ind in pop])
    long_sma_values = set([ind[1] for ind in pop])
    print("\nDEBUG: Unique short_sma values in initial population:")
    print(sorted(short_sma_values))
    print("\nDEBUG: Unique long_sma values in initial population:")
    print(sorted(long_sma_values))
    print(f"\nVerifying step size = 2: All values should be of form {SMA_MIN} + n*2")
    all_values = short_sma_values.union(long_sma_values)
    for val in sorted(all_values):
        remainder = (val - SMA_MIN) % 2
        if remainder != 0:
            print(f"WARNING: Value {val} does not follow step size (remainder = {remainder})")
    
    # Run the genetic algorithm
    pop, logbook = algorithms.eaSimple(
        pop, 
        toolbox, 
        cxpb=CROSSOVER_PROB,  # Crossover probability 
        mutpb=MUTATION_PROB,   # Mutation probability
        ngen=NUM_GENERATIONS,  # Number of generations
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True
    )
    
    # Get the best individual from the hall of fame
    best_individual = hall_of_fame[0]
    best_short_sma, best_long_sma = best_individual
    
    # Record optimization time
    optimization_end_time = time.time()
    optimization_time = optimization_end_time - optimization_start_time
    print(f"\nGenetic algorithm optimization completed in {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    
    print(f"Best parameters found: Short SMA = {best_short_sma}, Long SMA = {best_long_sma}")
    print(f"Best fitness (Sharpe ratio): {best_individual.fitness.values[0]:.6f}")
    
    # Print top results without saving to CSV
    print("\n--- TOP GENETIC ALGORITHM RESULTS ---")
    for idx, individual in enumerate(hall_of_fame):
        if idx < 20 and individual[0] < individual[1]:  # Only print valid combinations where short < long
            print(f"Top {idx+1}: Short SMA = {individual[0]}, Long SMA = {individual[1]}, Sharpe = {individual.fitness.values[0]:.6f}")
        if idx >= 20:
            break
    
    # Save results to text file with proper header and format, sorted by short_SMA and deduped
    # Save results to text file with proper header and format, sorted by short_SMA and deduped
    with open(RESULTS_FILE, 'w') as f:
        f.write("short_SMA,long_SMA,trades,sharpe_ratio\n")
        
        # Create a list to hold all results
        all_results = []
        
        # Process all hall of fame individuals
        for individual in hall_of_fame:
            short_sma, long_sma = individual
            
            # Skip invalid combinations where short_sma >= long_sma
            if short_sma >= long_sma:
                continue
                
            # Skip individuals with penalty fitness value (-999999.0)
            if abs(individual.fitness.values[0] + 999999.0) < 0.1:  # Using approximate equality check
                continue
                
            # Get trade count by reapplying strategy
            ga_strategy = SMAStrategy(
                short_sma=short_sma,
                long_sma=long_sma,
                big_point_value=big_point_value,
                slippage=slippage,
                capital=TRADING_CAPITAL,
                atr_period=ATR_PERIOD
            )
            
            # Apply the strategy with these parameters
            temp_data = data.copy()
            temp_data = ga_strategy.apply_strategy(temp_data)
            
            # Create evaluation slice (without warm-up period)
            if original_start_idx is not None:
                eval_data = temp_data.iloc[original_start_idx:].copy()
            else:
                eval_data = temp_data.copy()
            
            # Count total trades
            trade_count = eval_data['Position_Change_Strategy'].sum()
            
            # Store this result
            all_results.append((short_sma, long_sma, trade_count, individual.fitness.values[0]))
        
        # Remove duplicates (if any)
        unique_results = []
        seen_params = set()
        
        for result in all_results:
            param_key = (result[0], result[1])  # short_sma, long_sma as a tuple
            
            # Double-check the constraint again
            if result[0] >= result[1]:
                continue
                
            # Skip results with penalty fitness value (-999999.0)
            if abs(result[3] + 999999.0) < 0.1:  # Using approximate equality check
                continue
                
            if param_key not in seen_params:
                unique_results.append(result)
                seen_params.add(param_key)
        
        # Sort by short_SMA
        sorted_results = sorted(unique_results, key=lambda x: (x[0], x[1]))
        
        # Write sorted unique results to file
        for short_sma, long_sma, trade_count, sharpe in sorted_results:
            # Final sanity check
            if short_sma < long_sma and abs(sharpe + 999999.0) >= 0.1:  # Not a penalty value
                f.write(f"{short_sma},{long_sma},{int(trade_count)},{sharpe:.6f}\n")

    print(f"Saved GA optimization results to {RESULTS_FILE} (sorted by short_SMA, {len(sorted_results)} unique strategies)")
    
    # Apply the best parameters found to the full dataset
    strategy.short_sma = best_short_sma
    strategy.long_sma = best_long_sma
    
    print("\nApplying best strategy parameters...")
    apply_start_time = time.time()
    data = strategy.apply_strategy(data.copy())
    apply_end_time = time.time()
    apply_time = apply_end_time - apply_start_time
    print(f"Strategy application completed in {apply_time:.2f} seconds")
    
    # Trim data back to the original date range for evaluation
    if original_start_idx is not None:
        print("Trimming warm-up period for final evaluation and visualization...")
        data_for_evaluation = data.iloc[original_start_idx:].copy()
        print(f"Original data length: {len(data)}, Evaluation data length: {len(data_for_evaluation)}")
    else:
        data_for_evaluation = data.copy()
    
    # Calculate split index for in-sample/out-of-sample
    split_index = int(len(data_for_evaluation) * TRAIN_TEST_SPLIT)
    
    # Start timing the visualization process
    viz_start_time = time.time()
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(14, 16))
    
    # Plot price and SMAs
    plt.subplot(3, 1, 1)
    plt.plot(data_for_evaluation.index, data_for_evaluation['Close'], label=f'{data_obj.symbol} Price', color='blue')
    plt.plot(data_for_evaluation.index, data_for_evaluation['SMA_Short_Strategy'], label=f'{best_short_sma}-day SMA', color='orange')
    plt.plot(data_for_evaluation.index, data_for_evaluation['SMA_Long_Strategy'], label=f'{best_long_sma}-day SMA', color='red')
    
    # Plot position changes (using vectorized identification of changes)
    long_entries = (data_for_evaluation['Position_Dir_Strategy'] == 1) & data_for_evaluation['Position_Change_Strategy']
    short_entries = (data_for_evaluation['Position_Dir_Strategy'] == -1) & data_for_evaluation['Position_Change_Strategy']
    
    # Plot the entries
    plt.scatter(data_for_evaluation.index[long_entries], data_for_evaluation.loc[long_entries, 'Close'], 
                color='green', marker='^', s=50, label='Long Entry')
    plt.scatter(data_for_evaluation.index[short_entries], data_for_evaluation.loc[short_entries, 'Close'], 
                color='red', marker='v', s=50, label='Short Entry')
    
    plt.legend()
    plt.title(f'{data_obj.symbol} with GA-Optimized SMA Strategy ({best_short_sma}, {best_long_sma})')
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
    save_plot(f'{SYMBOL}_Optimized_Strategy_Plot.png')
    
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
    print("\n--- PERFORMANCE SUMMARY OF GA-OPTIMIZED SMA STRATEGY ---")
    print(f"Symbol: {data_obj.symbol}")
    print(f"Big Point Value (from data): {big_point_value}")
    print(f"ATR Period for Position Sizing: {ATR_PERIOD} days")
    print(f"Capital Allocation: ${TRADING_CAPITAL:,}")
    print(f"Average Position Size: {metrics['avg_position_size']:.2f} contracts")
    print(f"Maximum Position Size: {metrics['max_position_size']:.0f} contracts")
    print(f"Strategy Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"Market Buy & Hold P&L: ${market_cumulative_pnl:,.2f}")
    print(f"Outperformance: ${(metrics['total_pnl'] - market_cumulative_pnl):,.2f}")
    print(f"Sharpe ratio (entire period, annualized): {metrics['sharpe_full']:.6f}")
    print(f"Sharpe ratio (in-sample, annualized): {metrics['sharpe_in_sample']:.6f}")
    print(f"Sharpe ratio (out-of-sample, annualized): {metrics['sharpe_out_sample']:.6f}")
    print(f"Maximum Drawdown: ${abs(metrics['max_drawdown_dollars']):,.2f}")
    print("\n--- TRADE COUNT SUMMARY ---")
    print(f"In-sample period trades: {metrics['in_sample_trades']}")
    print(f"Out-of-sample period trades: {metrics['out_sample_trades']}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"In-sample P&L: ${metrics['in_sample_pnl']:,.2f}")
    print(f"Out-of-sample P&L: ${metrics['out_sample_pnl']:,.2f}")
    
    print(f"\nBest parameters from GA: Short SMA = {best_short_sma}, Long SMA = {best_long_sma}, Sharpe = {best_individual.fitness.values[0]:.6f}")
    
    # Calculate overall execution time
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    # Print timing summary
    print("\n--- EXECUTION TIME SUMMARY (Genetic Algorithm Implementation) ---")
    print(f"Data loading time: {load_time:.2f} seconds")
    print(f"Data preparation time: {prep_time:.2f} seconds")
    print(f"Genetic Algorithm optimization time: {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    print(f"Strategy application time: {apply_time:.2f} seconds")
    print(f"Visualization time: {viz_time:.2f} seconds")
    print(f"Metrics calculation time: {metrics_time:.2f} seconds")
    print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    
    # Print cache statistics
    print(f"\nEvaluation Statistics:")
    print(f"Unique evaluations: {evaluation_counter}")
    print(f"Cache hits (repeated combinations): {cache_hit_counter}")
    print(f"Total checks: {evaluation_counter + cache_hit_counter}")
    print(f"Cache hit rate: {cache_hit_counter/(evaluation_counter + cache_hit_counter)*100:.2f}%")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()