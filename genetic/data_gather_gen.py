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
import read_ts
from input_gen import *
from SMA_Strategy import SMAStrategy
import logging
import sys
import pickle

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",   #no timestamp
    handlers=[
        logging.FileHandler("execution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)



# Global variables for caching and statistics
evaluated_combinations = {}
evaluation_counter = 0
cache_hit_counter = 0

# -----------------------------------------------------
# Utility functions
# -----------------------------------------------------

def save_plot(plot_name, output_dir) -> None:
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.close()


def save_parameters(
    big_point_value: float,
    slippage: float,
    capital: float,
    atr_period: int,
    filepath: str = "parameters.json",
) -> None:
    """Persist key runtime parameters to *filepath* in JSON format."""
    parameters = {
        "big_point_value": big_point_value,
        "slippage": slippage,
        "capital": capital,
        "atr_period": atr_period,
    }
    with open(filepath, "w") as file:
        json.dump(parameters, file)


def get_slippage_from_excel(symbol: str, data_dir: str) -> float:
    """Strictly retrieve slippage for *symbol* from the Excel sheet stored in
    *data_dir*. Raises if the file or symbol is missing or malformed."""
    excel_path = os.path.join(data_dir, "sessions_slippages.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Slippage Excel file not found at {excel_path}")

    lookup_symbol = symbol.replace("=F", "")
    df = pd.read_excel(excel_path)

    # Debug prints
    logging.info("\nContents of sessions_slippages.xlsx:")
    logging.info(df.head())

    if df.shape[1] < 4:
        raise ValueError(
            f"Excel file has fewer than 4 columns: {df.columns.tolist()}"
        )

    logging.info(f"Columns: {df.columns.tolist()}")

    df["SymbolUpper"] = df.iloc[:, 1].astype(str).str.upper()
    matching_rows = df[df["SymbolUpper"] == lookup_symbol.upper()]
    if matching_rows.empty:
        raise ValueError(
            f"Symbol '{lookup_symbol}' not found in column B of Excel file"
        )

    slippage_value = matching_rows.iloc[0, 3]
    if pd.isna(slippage_value) or not isinstance(slippage_value, (int, float)):
        raise ValueError(
            f"Invalid slippage value for symbol '{lookup_symbol}': {slippage_value}"
        )

    logging.info(f"Found slippage for {lookup_symbol} in column D: {slippage_value}")
    return slippage_value


def find_futures_file(symbol: str, data_dir: str) -> str:
    """Return the .DAT file matching the symbol in the data directory."""
    # Files may have additional identifiers after the symbol (e.g. WBS=107XC)
    pattern = f"A_OHLCV_@{symbol}*_minutes_1440_*.dat"
    files = glob.glob(os.path.join(data_dir, pattern))
    
    if not files:
        raise FileNotFoundError(f"No data file found for {symbol} in {data_dir}")
        
    return files[0]  # Return the first matching file


# -------------------------------------------------------------------------
# Genetic algorithm helpers
# -------------------------------------------------------------------------

def evaluate_individual(individual):
    """Evaluate a (short_sma, long_sma) individual and return a Sharpe-ratio
    fitness tuple. Uses several globals that *main()* sets up."""
    global evaluation_counter, cache_hit_counter, data, big_point_value, slippage, original_start_idx

    short_sma, long_sma = individual
    key = (short_sma, long_sma)

    # Memoisation shortcut
    if key in evaluated_combinations:
        cache_hit_counter += 1
        return evaluated_combinations[key]

    evaluation_counter += 1

    # Invalid SMA ordering penalty
    if short_sma >= long_sma:
        evaluated_combinations[key] = (-999999.0, 0)
        return -999999.0, 0

    ga_strategy = SMAStrategy(
        short_sma=short_sma,
        long_sma=long_sma,
        big_point_value=big_point_value,
        slippage=slippage,
        capital=TRADING_CAPITAL,
        atr_period=ATR_PERIOD,
    )

    temp_data = data.copy()
    temp_data = ga_strategy.apply_strategy(temp_data)

    eval_data = (
        temp_data.iloc[original_start_idx:].copy()
        if original_start_idx is not None
        else (_ for _ in ()).throw(ValueError("original_start_idx is None"))
    )

    split_idx = int(len(eval_data) * TRAIN_TEST_SPLIT)
    train_data = eval_data.iloc[:split_idx]

    if "Daily_PnL_Strategy" in train_data.columns and len(train_data) > 0:
        daily_returns = train_data["Daily_PnL_Strategy"]
        if daily_returns.sum() != 0 and daily_returns.std() != 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            trade_count = int(train_data["Position_Change_Strategy"].sum())
            if trade_count < MIN_TRADES or trade_count > MAX_TRADES:
                result = (-999999.0, trade_count)
            else:
                result = (sharpe, trade_count)
        else:
            result = (-999999.0, 0)
    else:
        result = (-999999.0, 0)

    evaluated_combinations[key] = result
    return result


def custom_mutation(individual, indpb):
    """Mutation operator that respects short_sma < long_sma and step size 2."""
    valid_values = list(range(SMA_MIN, SMA_MAX + 1, 2))
    for i in range(len(individual)):
        if random.random() < indpb:
            if i == 0:  # short_sma
                long_idx = valid_values.index(individual[1]) if individual[1] in valid_values else len(valid_values) - 1
                new_idx = random.randint(0, long_idx - 1)
                individual[i] = valid_values[new_idx]
            else:  # long_sma
                short_idx = valid_values.index(individual[0]) if individual[0] in valid_values else 0
                new_idx = random.randint(short_idx + 1, len(valid_values) - 1)
                individual[i] = valid_values[new_idx]
    return individual,


def custom_crossover(ind1, ind2):
    """Crossover operator that swaps SMA values while maintaining constraints."""
    valid_values = list(range(SMA_MIN, SMA_MAX + 1, 2))
    c1, c2 = list(ind1), list(ind2)

    if random.random() < CROSSOVER_PROB:
        if random.random() < 0.5:
            c1[0], c2[0] = c2[0], c1[0]
        if random.random() < 0.5:
            c1[1], c2[1] = c2[1], c1[1]

        if c1[0] >= c1[1]:
            short_idx = valid_values.index(c1[0]) if c1[0] in valid_values else 0
            c1[1] = valid_values[random.randint(short_idx + 1, len(valid_values) - 1)]
        if c2[0] >= c2[1]:
            short_idx = valid_values.index(c2[0]) if c2[0] in valid_values else 0
            c2[1] = valid_values[random.randint(short_idx + 1, len(valid_values) - 1)]

    return creator.Individual(c1), creator.Individual(c2)


def create_valid_individual():
    """Return a random individual satisfying short_sma < long_sma."""
    valid_values = list(range(SMA_MIN, SMA_MAX + 1, 2))
    short_idx = random.randint(0, len(valid_values) - 2)
    long_idx = random.randint(short_idx + 1, len(valid_values) - 1)
    return creator.Individual([valid_values[short_idx], valid_values[long_idx]])

# -------------------------------------------------------------------------
# Date-range / warm-up helper
# -------------------------------------------------------------------------

def apply_warmup_and_date_filter(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    warm_up_days: int,
):
    """Return (df_with_warmup, warm_up_idx).

    If *start_date* and *end_date* are provided, the function adds *warm_up_days*
    before *start_date* so that indicators (e.g. SMA & ATR) have enough history
    to 'warm-up'.  The integer *warm_up_idx* marks the first row that belongs to
    the real evaluation window (closest index to *start_date*).  When no date
    range is given, the original DataFrame is returned and *warm_up_idx* is
    None.
    """

    if not (start_date and end_date):
        return df.copy(), None

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    adjusted_start = start_dt - pd.Timedelta(days=warm_up_days)

    df_with_warmup = df[(df.index >= adjusted_start) & (df.index <= end_dt)].copy()

    if df_with_warmup.empty:
        raise ValueError(
            f"No data available between {adjusted_start.date()} and {end_dt.date()}"
        )

    warm_up_idx = df_with_warmup.index.get_indexer([start_dt], method="nearest")[0]

    # Compact informational logging.info
    logging.info(
        f"Warm-up: loaded {adjusted_start.date()} -> {end_dt.date()}  "
        f"(+{warm_up_days} d); analysis starts at idx {warm_up_idx}"
    )

    return df_with_warmup, warm_up_idx

# -------------------------------------------------------------------------
# Visualization helper -----------------------------------------------------
# -------------------------------------------------------------------------

def visualize_results(
    data_for_evaluation: pd.DataFrame,
    symbol_label: str,
    file_symbol: str,
    best_short_sma: int,
    best_long_sma: int,
    atr_period: int,
    train_test_split: float,
    output_dir: str,
) -> None:
    """Create and save strategy visualizations (price with SMAs, position sizing, and P&L)."""

    # Calculate split index for in-sample/out-of-sample
    split_index = int(len(data_for_evaluation) * train_test_split)

    # Create a figure with multiple subplots
    plt.figure(figsize=(14, 16))

    # ------------------------------------------------------------------
    # 1) Price chart with SMA overlays & trade markers
    # ------------------------------------------------------------------
    plt.subplot(3, 1, 1)
    plt.plot(
        data_for_evaluation.index,
        data_for_evaluation["Close"],
        label=f"{symbol_label} Price",
        color="blue",
    )
    plt.plot(
        data_for_evaluation.index,
        data_for_evaluation["SMA_Short_Strategy"],
        label=f"{best_short_sma}-day SMA",
        color="orange",
    )
    plt.plot(
        data_for_evaluation.index,
        data_for_evaluation["SMA_Long_Strategy"],
        label=f"{best_long_sma}-day SMA",
        color="red",
    )

    # Trade entry markers
    long_entries = (
        (data_for_evaluation["Position_Dir_Strategy"] == 1)
        & data_for_evaluation["Position_Change_Strategy"]
    )
    short_entries = (
        (data_for_evaluation["Position_Dir_Strategy"] == -1)
        & data_for_evaluation["Position_Change_Strategy"]
    )
    plt.scatter(
        data_for_evaluation.index[long_entries],
        data_for_evaluation.loc[long_entries, "Close"],
        color="green",
        marker="^",
        s=50,
        label="Long Entry",
    )
    plt.scatter(
        data_for_evaluation.index[short_entries],
        data_for_evaluation.loc[short_entries, "Close"],
        color="red",
        marker="v",
        s=50,
        label="Short Entry",
    )
    plt.legend()
    plt.title(
        f"{symbol_label} with GA-Optimized SMA Strategy ({best_short_sma}, {best_long_sma})"
    )
    plt.grid(True)

    # ------------------------------------------------------------------
    # 2) Position sizing vs ATR (dual-axis plot)
    # ------------------------------------------------------------------
    ax1 = plt.subplot(3, 1, 2)
    ax2 = ax1.twinx()

    ax1.plot(
        data_for_evaluation.index,
        data_for_evaluation["Position_Size_Strategy"],
        label="Position Size (# Contracts)",
        color="purple",
    )
    ax1.set_ylabel("Position Size (# Contracts)", color="purple")
    ax1.tick_params(axis="y", colors="purple")

    ax2.plot(
        data_for_evaluation.index,
        data_for_evaluation["ATR_Strategy"],
        label=f"ATR ({atr_period}-day)",
        color="orange",
    )
    ax2.set_ylabel(f"ATR ({atr_period}-day)", color="orange")
    ax2.tick_params(axis="y", colors="orange")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.title(f"Position Sizing Based on {atr_period}-day ATR")
    ax1.grid(True)

    # ------------------------------------------------------------------
    # 3) Strategy performance (P&L)
    # ------------------------------------------------------------------
    plt.subplot(3, 1, 3)
    strategy_pnl_cumulative = (
        data_for_evaluation["Cumulative_PnL_Strategy"]
        - data_for_evaluation["Cumulative_PnL_Strategy"].iloc[0]
    )

    plt.plot(
        data_for_evaluation.index,
        strategy_pnl_cumulative,
        label="Strategy P&L (full period)",
        color="green",
    )

    # Out-of-sample highlight
    plt.plot(
        data_for_evaluation.index[split_index:],
        strategy_pnl_cumulative.iloc[split_index:],
        label=f"Strategy P&L (last {int((1 - train_test_split) * 100)}% out-of-sample)",
        color="purple",
    )

    # Split and zero reference lines
    plt.axvline(
        x=data_for_evaluation.index[split_index],
        color="black",
        linestyle="--",
        label=f"Train/Test Split ({int(train_test_split * 100)}%/{int((1 - train_test_split) * 100)}%)",
    )
    plt.axhline(y=0.0, color="gray", linestyle="-", label="Break-even")

    # ------------------------------------------------------------------
    # Calculate Sharpe ratios for legend annotation
    # ------------------------------------------------------------------
    daily_pnl = data_for_evaluation["Daily_PnL_Strategy"]
    returns_in_sample = daily_pnl.iloc[:split_index]
    returns_out_sample = daily_pnl.iloc[split_index:]

    sharpe_in_sample = (
        (returns_in_sample.mean() / returns_in_sample.std()) * np.sqrt(252)
        if returns_in_sample.std() not in (0, np.nan) and not np.isnan(returns_in_sample.std())
        else 0
    )
    sharpe_out_sample = (
        (returns_out_sample.mean() / returns_out_sample.std()) * np.sqrt(252)
        if returns_out_sample.std() not in (0, np.nan) and not np.isnan(returns_out_sample.std())
        else 0
    )

    # Add Sharpe ratio values to legend as invisible handles
    plt.plot([], [], " ", label=f"In-sample Sharpe: {sharpe_in_sample:.2f}")
    plt.plot([], [], " ", label=f"Out-of-sample Sharpe: {sharpe_out_sample:.2f}")

    plt.legend()
    plt.title("Strategy Performance (Dollar P&L)")
    plt.ylabel("P&L ($)")
    plt.grid(True)

    # ------------------------------------------------------------------
    # Finalize & persist
    # ------------------------------------------------------------------
    plt.tight_layout()
    save_plot(f"{file_symbol}_Optimized_Strategy_Plot.png", output_dir)

    logging.info("Visualization completed.")

# -------------------------------------------------------------------------
# Printing helper ----------------------------------------------------------
# -------------------------------------------------------------------------

def print_performance_metrics(
    *,
    symbol: str,
    big_point_value: float,
    atr_period: int,
    trading_capital: float,
    metrics: dict,
    market_cumulative_pnl: float,
    best_short_sma: int,
    best_long_sma: int,
    best_sharpe: float,
    evaluation_counter: int,
    cache_hit_counter: int,
) -> None:
    """Pretty-logging.info the key performance results and evaluation statistics."""

    logging.info("\n--- PERFORMANCE SUMMARY OF GA-OPTIMIZED SMA STRATEGY ---")
    logging.info(f"Symbol: {symbol}")
    logging.info(f"Big Point Value (from data): {big_point_value}")
    logging.info(f"ATR Period for Position Sizing: {atr_period} days")
    logging.info(f"Capital Allocation: ${trading_capital:,}")

    # --- Core metrics
    logging.info(f"Average Position Size: {metrics['avg_position_size']:.2f} contracts")
    logging.info(f"Maximum Position Size: {metrics['max_position_size']:.0f} contracts")
    logging.info(f"Strategy Total P&L: ${metrics['total_pnl']:,.2f}")
    logging.info(f"Market Buy & Hold P&L: ${market_cumulative_pnl:,.2f}")
    logging.info(f"Outperformance: ${(metrics['total_pnl'] - market_cumulative_pnl):,.2f}")
    logging.info(f"Sharpe ratio (entire period, annualized): {metrics['sharpe_full']:.6f}")
    logging.info(f"Sharpe ratio (in-sample, annualized): {metrics['sharpe_in_sample']:.6f}")
    logging.info(f"Sharpe ratio (out-of-sample, annualized): {metrics['sharpe_out_sample']:.6f}")
    logging.info(f"Maximum Drawdown: ${abs(metrics['max_drawdown_dollars']):,.2f}")

    # --- Trade counts
    logging.info("\n--- TRADE COUNT SUMMARY ---")
    logging.info(f"In-sample period trades: {metrics['in_sample_trades']}")
    logging.info(f"Out-of-sample period trades: {metrics['out_sample_trades']}")
    logging.info(f"Total trades: {metrics['total_trades']}")
    logging.info(f"In-sample P&L: ${metrics['in_sample_pnl']:,.2f}")
    logging.info(f"Out-of-sample P&L: ${metrics['out_sample_pnl']:,.2f}")

    # --- GA best parameters
    logging.info(
        f"\nBest parameters from GA: Short SMA = {best_short_sma}, Long SMA = {best_long_sma}, "
        f"Sharpe = {best_sharpe:.6f}"
    )

    # --- Evaluation statistics
    logging.info("\nEvaluation Statistics:")
    logging.info(f"Unique evaluations: {evaluation_counter}")
    logging.info(f"Cache hits (repeated combinations): {cache_hit_counter}")
    total_checks = evaluation_counter + cache_hit_counter
    logging.info(f"Total checks: {total_checks}")
    if total_checks:
        logging.info(f"Cache hit rate: {cache_hit_counter/total_checks*100:.2f}%")

# -------------------------------------------------------------------------
# Main execution flow ------------------------------------------------------
# -------------------------------------------------------------------------

def main():
    # Setup paths using relative directories
    WORKING_DIR = "."  # Current directory
    DATA_DIR = os.path.join(WORKING_DIR, "data")

    SYMBOL = TICKER

    # Create the output directory for each symbol
    output_dir = os.path.join(WORKING_DIR, 'output2', SYMBOL)  # Symbol-specific folder
    os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

    # Expose commonly-used variables to the helper functions defined at module scope
    global data, big_point_value, slippage, original_start_idx

    logging.info(f"Loading {TICKER} data from local files...")
    data_file = find_futures_file(SYMBOL, DATA_DIR)
    logging.info(f"Found data file: {os.path.basename(data_file)}")

    # Load the futures data file
    all_data = read_ts.read_ts_ohlcv_dat(data_file)
    
    # Extract metadata and OHLCV data from the first data object
    data_obj = all_data[0]
    tick_size = data_obj.big_point_value * data_obj.tick_size
    
    # Get big point value from data
    big_point_value = data_obj.big_point_value

    # Fetch slippage value from Excel
    slippage = get_slippage_from_excel(TICKER, DATA_DIR)
    logging.info(f"Using slippage from Excel column D: {slippage}")

    # Save the parameters to a JSON file
    save_parameters(big_point_value, slippage, TRADING_CAPITAL, ATR_PERIOD)

    ohlc_data = data_obj.data.copy()  # Make a copy to avoid modifying original data
    
    # Print information about the data
    logging.info(f"\nSymbol: {data_obj.symbol}")
    logging.info(f"Description: {data_obj.description}")
    logging.info(f"Exchange: {data_obj.exchange}")
    logging.info(f"Interval: {data_obj.interval_type} {data_obj.interval_span}")
    logging.info(f"Tick size: {tick_size}")
    logging.info(f"Big point value: {big_point_value}")
    logging.info(f"Data shape: {ohlc_data.shape}")
    logging.info(f"Date range: {ohlc_data['datetime'].min()} to {ohlc_data['datetime'].max()}")
    
    # Display the first few rows of data
    logging.info("\nFirst few rows of OHLCV data:")
    logging.info(ohlc_data.head())
    
    # Convert the OHLCV data to the format expected by the SMA strategy
    # First, rename columns to match
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
    
    # ------------------------------------------------------------------
    # Apply warm-up / date filtering in a single, tidy helper call
    # ------------------------------------------------------------------
    WARM_UP_DAYS = SMA_MAX + ATR_PERIOD + 50  # small safety buffer
    data, original_start_idx = apply_warmup_and_date_filter(
        data,
        start_date=START_DATE,
        end_date=END_DATE,
        warm_up_days=WARM_UP_DAYS,
    )

    
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
    logging.info("\nStarting genetic algorithm optimization...")
    logging.info(f"Population size: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
    logging.info(f"Using genetic algorithm parameters from input.py")
    optimization_start_time = time.time()
    
    # Fix random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # DEAP GA Setup
    # Define parameters and their ranges
    PARAM_NAMES = ["short_sma", "long_sma"]
    
    # Ensure DEAP creator/classes are fresh for each run
    if "FitnessMax" in dir(creator):
        del creator.FitnessMax
    if "Individual" in dir(creator):
        del creator.Individual

    # Use a very small non-zero weight for trade count to prevent division-by-zero
    # in DEAP while keeping its influence on selection negligible.
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1e-9))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Register operators
    toolbox.register("individual", create_valid_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, indpb=MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create hall of fame to store best individuals
    hall_of_fame = tools.HallOfFame(maxsize=HALL_OF_FAME_SIZE)
    
    # Statistics setup – only track the Sharpe ratio (first fitness element)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Initialize population
    pop = toolbox.population(n=POPULATION_SIZE)
    
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
    logging.info(f"\nGenetic algorithm optimization completed in {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    
    logging.info(f"Best parameters found: Short SMA = {best_short_sma}, Long SMA = {best_long_sma}")
    logging.info(f"Best fitness (Sharpe ratio): {best_individual.fitness.values[0]:.6f}")
    
    logging.info("\n--- TOP GENETIC ALGORITHM RESULTS ---")
    for idx, individual in enumerate(hall_of_fame):
        if idx < 5 and individual[0] < individual[1]:  # Only logging.info valid combinations where short < long
            logging.info(f"Top {idx+1}: Short SMA = {individual[0]}, Long SMA = {individual[1]}, Sharpe = {individual.fitness.values[0]:.6f}")
        if idx >= 20:
            break
    
    # Save results to a pickle file, sorted by short_SMA and deduped
    
    # Create a list to hold all results
    all_results = []
    
    # Debug counters
    invalid_sma_order = 0
    penalty_fitness = 0
    
    logging.info(f"Hall of Fame size: {len(hall_of_fame)}")
    
    # Process all hall of fame individuals – trade count already stored in the
    # second fitness dimension, so no need to re-run the strategy.
    for individual in hall_of_fame:
        short_sma, long_sma = individual

        # Skip invalid combinations where short_sma >= long_sma
        if short_sma >= long_sma:
            invalid_sma_order += 1
            continue

        sharpe = individual.fitness.values[0]

        # Skip individuals with penalty fitness value (-999999.0)
        if abs(sharpe + 999999.0) < 0.1:  # Using approximate equality check
            penalty_fitness += 1
            continue

        trade_count = int(individual.fitness.values[1])

        # Store this result (order: short, long, sharpe, trades)
        all_results.append((short_sma, long_sma, sharpe, trade_count))
    
    # Print the number of elements in all_results
    logging.info(f"Number of elements in all_results: {len(all_results)}")
    logging.info(f"Hall of Fame filtering: {invalid_sma_order} had invalid SMA order, {penalty_fitness} had penalty fitness")
    logging.info(f"Total filtered from Hall of Fame: {invalid_sma_order + penalty_fitness}")
    logging.info(f"Hall of Fame -> all_results: {len(hall_of_fame)} -> {len(all_results)}")
    
    # Remove duplicates (if any)
    unique_results = []
    seen_params = set()
    
    for result in all_results:
        param_key = (result[0], result[1])  # short_sma, long_sma as a tuple
        
        # Double-check the constraint again
        if result[0] >= result[1]:
            continue
            
        # Skip results with penalty fitness value (-999999.0)
        if abs(result[2] + 999999.0) < 0.1:  # Using approximate equality check
            continue
            
        if param_key not in seen_params:
            unique_results.append(result)
            seen_params.add(param_key)
    
    # Sort by short_SMA for a clean presentation order
    sorted_results = sorted(unique_results, key=lambda x: (x[0], x[1]))

    # ------------------------------------------------------------------
    # Save the *unique* strategy list
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(
        sorted_results,
        columns=["short_SMA", "long_SMA", "sharpe_ratio", "trades"],
    )
    results_df["trades"] = results_df["trades"].astype(int)

    results_df.to_pickle(RESULTS_FILE)

    logging.info(
        f"Saved GA optimization results to {RESULTS_FILE} (sorted by short_SMA, {len(sorted_results)} unique strategies)"
    )
    
    # Apply the best parameters found to the full dataset
    strategy.short_sma = best_short_sma
    strategy.long_sma = best_long_sma
    
    logging.info("\nApplying best strategy parameters...")
    data = strategy.apply_strategy(data.copy())
    logging.info("Strategy application completed.")
    
    # Trim data back to the original date range for evaluation
    if original_start_idx is not None:
        logging.info("Trimming warm-up period for final evaluation and visualization...")
        data_for_evaluation = data.iloc[original_start_idx:].copy()
        logging.info(f"Original data length: {len(data)}, Evaluation data length: {len(data_for_evaluation)}")
    else:
        raise ValueError("original_start_idx is None, cannot proceed with evaluation and visualization.")

        # Save best strategy PnL
    col_name = f"SMA_{SYMBOL}_best_{best_short_sma}/{best_long_sma}"
    trimmed = data_for_evaluation.copy()
    pnl_df = pd.DataFrame({col_name: trimmed["Daily_PnL_Strategy"]})

    pnl_df.index = pnl_df.index.normalize()

    if os.path.exists("pnl_temp.pkl"):
        with open("pnl_temp.pkl", "rb") as f:
            existing = pickle.load(f)
            if isinstance(existing, pd.DataFrame):
                for col in pnl_df.columns:
                    # ✅ Always overwrite best Sharpe column
                    existing[col] = pnl_df[col]
                pnl_df = existing

    with open("pnl_temp.pkl", "wb") as f:
        pickle.dump(pnl_df, f)
    
    # Generate and save strategy visualization using the helper
    visualize_results(
        data_for_evaluation=data_for_evaluation,
        symbol_label=data_obj.symbol,
        file_symbol=SYMBOL,
        best_short_sma=best_short_sma,
        best_long_sma=best_long_sma,
        atr_period=ATR_PERIOD,
        train_test_split=TRAIN_TEST_SPLIT,
        output_dir=output_dir,
    )

    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(
        data_for_evaluation,  # Use the trimmed data for metrics
        strategy_name="Strategy",
        train_test_split=TRAIN_TEST_SPLIT
    )
    logging.info("Performance metrics calculation completed.")
    
    # Calculate market performance for comparison (for reporting only, not plotting)
    market_cumulative_pnl = data_for_evaluation['Market_PnL_Strategy'].cumsum().iloc[-1]
    
    # Pretty-logging.info performance summary using helper
    print_performance_metrics(
        symbol=data_obj.symbol,
        big_point_value=big_point_value,
        atr_period=ATR_PERIOD,
        trading_capital=TRADING_CAPITAL,
        metrics=metrics,
        market_cumulative_pnl=market_cumulative_pnl,
        best_short_sma=best_short_sma,
        best_long_sma=best_long_sma,
        best_sharpe=best_individual.fitness.values[0],
        evaluation_counter=evaluation_counter,
        cache_hit_counter=cache_hit_counter,
    )
    
    logging.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()