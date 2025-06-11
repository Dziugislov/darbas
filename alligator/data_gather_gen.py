import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import time
import random
import datetime
import concurrent.futures

from deap import base, creator, tools, algorithms

# Import the read_ts module for data loading
import read_ts

# Import configuration (defines: TICKER, DATA_DIR, SMA_MIN, SMA_MAX, STEP, POPULATION_SIZE, NUM_GENERATIONS,
# CROSSOVER_PROB, MUTATION_PROB, HALL_OF_FAME_SIZE, TRAIN_TEST_SPLIT, MIN_TRADES, MAX_TRADES, 
# RESULTS_FILE, TRADING_CAPITAL, ATR_PERIOD, RANDOM_SEED, START_DATE, END_DATE)
from input_gen import *

# Replace two‐SMA strategy with your new three‐SMA AlligatorSMAStrategy
from Alligator_SMA_Strategy import AlligatorSMAStrategy

# Global caches
evaluated_combinations = {}
evaluation_counter = 0
cache_hit_counter = 0

def main():
    overall_start_time = time.time()
    
    WORKING_DIR = "."
    DATA_DIR = os.path.join(WORKING_DIR, "data")
    
    SYMBOL = TICKER.replace('=F', '')
    output_dir = os.path.join(WORKING_DIR, 'output2', SYMBOL)
    os.makedirs(output_dir, exist_ok=True)
    
    def save_plot(plot_name):
        plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()
    
    def save_parameters():
        parameters = {
            "big_point_value": big_point_value,
            "slippage": slippage,
            "capital": TRADING_CAPITAL,
            "atr_period": ATR_PERIOD
        }
        with open("parameters.json", "w") as file:
            json.dump(parameters, file)
    
    def get_slippage_from_excel(symbol, data_dir):
        excel_path = os.path.join(data_dir, "sessions_slippages.xlsx")
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Slippage Excel file not found at {excel_path}")
        lookup_symbol = symbol.replace('=F', '').upper()
        df = pd.read_excel(excel_path)
        if df.shape[1] < 4:
            raise ValueError(f"Excel file has fewer than 4 columns: {df.columns.tolist()}")
        df['SymbolUpper'] = df.iloc[:, 1].astype(str).str.upper()
        matching_rows = df[df['SymbolUpper'] == lookup_symbol]
        if matching_rows.empty:
            raise ValueError(f"Symbol '{lookup_symbol}' not found in column B of Excel file")
        slippage_value = matching_rows.iloc[0, 3]
        if pd.isna(slippage_value) or not isinstance(slippage_value, (int, float)):
            raise ValueError(f"Invalid slippage value for symbol '{lookup_symbol}': {slippage_value}")
        return slippage_value
    
    def find_futures_file(symbol, data_dir):
        patterns = [
            f"*@{symbol}_*.dat",
            f"*_@{symbol}_*.dat",
            f"*_{symbol}_*.dat",
            f"*@{symbol}*.dat"
        ]
        for pat in patterns:
            files = glob.glob(os.path.join(data_dir, pat))
            if files:
                return files[0]
        raise FileNotFoundError(f"No data file found for {symbol} in {data_dir}")
    
    # === Load local data ===
    print(f"Loading {TICKER} data from local files...")
    data_file = find_futures_file(SYMBOL, DATA_DIR)
    print(f"Found data file: {os.path.basename(data_file)} (size {os.path.getsize(data_file)} bytes)")
    
    print("Loading data file...")
    load_start_time = time.time()
    all_data = read_ts.read_ts_ohlcv_dat(data_file)
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    print(f"Data loaded in {load_time:.2f}s, items: {len(all_data)}")
    
    data_obj = all_data[0]
    tick_size = data_obj.big_point_value * data_obj.tick_size
    big_point_value = data_obj.big_point_value
    
    slippage = get_slippage_from_excel(TICKER, DATA_DIR)
    print(f"Using slippage from Excel (col D): {slippage}")
    save_parameters()
    
    prep_start_time = time.time()
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
    
    original_start_idx = None
    if START_DATE and END_DATE:
        warm_up_days = SMA_MAX + ATR_PERIOD + 50
        start_date = pd.to_datetime(START_DATE)
        end_date = pd.to_datetime(END_DATE)
        adjusted_start = start_date - pd.Timedelta(days=warm_up_days)
        data = data[(data.index >= adjusted_start) & (data.index <= end_date)]
        if data.empty:
            raise ValueError(f"No data for date range {START_DATE} to {END_DATE}")
        original_start_idx = data.index.get_indexer([start_date], method='nearest')[0]
        print(f"Extended data with {warm_up_days}d warm-up, adjusted range: {adjusted_start.date()} to {END_DATE}")
    
    prep_end_time = time.time()
    prep_time = prep_end_time - prep_start_time
    print(f"Data prep done in {prep_time:.2f}s, shape: {data.shape}")
    
    # Initialize a placeholder strategy (periods will be set in GA)
    strategy = AlligatorSMAStrategy(
        short_sma=0,
        med_sma=0,
        long_sma=0,
        big_point_value=big_point_value,
        slippage=slippage,
        capital=TRADING_CAPITAL,
        atr_period=ATR_PERIOD
    )
    
    # === Genetic Algorithm Setup ===
    print("\nStarting GA optimization for three‐SMA AlligatorSMAStrategy...")
    print(f"Population size: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
    random.seed(RANDOM_SEED)
    
    # Fitness function
    def evaluate_individual(individual):
        """
        individual = [short_sma, med_sma, long_sma]
        Returns a tuple (sharpe,)
        """
        global evaluation_counter, cache_hit_counter
        
        short_sma, med_sma, long_sma = individual
        key = (short_sma, med_sma, long_sma)
        
        if key in evaluated_combinations:
            cache_hit_counter += 1
            return evaluated_combinations[key]
        
        evaluation_counter += 1
        
        # Enforce short < med < long
        if not (short_sma < med_sma < long_sma):
            evaluated_combinations[key] = (-999999.0,)
            return (-999999.0,)
        
        ga_strategy = AlligatorSMAStrategy(
            short_sma=short_sma,
            med_sma=med_sma,
            long_sma=long_sma,
            big_point_value=big_point_value,
            slippage=slippage,
            capital=TRADING_CAPITAL,
            atr_period=ATR_PERIOD
        )
        
        temp = data.copy()
        temp = ga_strategy.apply_strategy(temp, strategy_name="Strategy")
        
        if original_start_idx is not None:
            eval_data = temp.iloc[original_start_idx:].copy()
        else:
            eval_data = temp.copy()
        
        split_idx = int(len(eval_data) * TRAIN_TEST_SPLIT)
        train_data = eval_data.iloc[:split_idx]
        
        if 'Daily_PnL_Strategy' in train_data.columns and len(train_data) > 0:
            dr = train_data['Daily_PnL_Strategy']
            if dr.std() > 0 and dr.sum() != 0:
                sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
                trade_count = train_data['Position_Change_Strategy'].sum()
                if trade_count < MIN_TRADES or trade_count > MAX_TRADES:
                    result = (-999999.0,)
                else:
                    result = (sharpe,)
            else:
                result = (-999999.0,)
        else:
            result = (-999999.0,)
        
        evaluated_combinations[key] = result
        return result
    
    # Custom mutation: ensure short < med < long
    def custom_mutation(individual, indpb):
        valid_values = list(range(SMA_MIN, SMA_MAX + 1, STEP))
        # individual = [short, med, long]
        # With probability indpb, mutate each gene but keep ordering
        for i in range(3):
            if random.random() < indpb:
                if i == 0:  # mutate short
                    # possible indices for short: [0 .. idx(med)-1]
                    med_idx = valid_values.index(individual[1])
                    new_idx = random.randint(0, med_idx - 1)
                    individual[0] = valid_values[new_idx]
                elif i == 1:  # mutate med
                    short_idx = valid_values.index(individual[0])
                    long_idx = valid_values.index(individual[2])
                    # Check if there's a valid range for mutation
                    if long_idx - short_idx > 1:
                        new_idx = random.randint(short_idx + 1, long_idx - 1)
                        individual[1] = valid_values[new_idx]
                    else:
                        # If no valid range, skip mutation for this gene
                        continue
                else:  # mutate long
                    med_idx = valid_values.index(individual[1])
                    new_idx = random.randint(med_idx + 1, len(valid_values) - 1)
                    individual[2] = valid_values[new_idx]
        return (individual,)
    
    # Custom crossover: ensure ordering after swap
    def custom_crossover(ind1, ind2):
        valid_values = list(range(SMA_MIN, SMA_MAX + 1, STEP))
        c1, c2 = list(ind1), list(ind2)
        
        if random.random() < CROSSOVER_PROB:
            # Swap each gene with 50% chance, then fix ordering
            for i in range(3):
                if random.random() < 0.5:
                    c1[i], c2[i] = c2[i], c1[i]
            
            # Fix c1
            c1_sorted = sorted(c1)
            c1[:] = c1_sorted
            # Fix c2
            c2_sorted = sorted(c2)
            c2[:] = c2_sorted
        
        return creator.Individual(c1), creator.Individual(c2)
    
    # Build DEAP types
    if 'FitnessMax' in dir(creator):
        del creator.FitnessMax
    if 'Individual' in dir(creator):
        del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Create a valid 3‐gene individual:
    def create_valid_individual():
        valid_values = list(range(SMA_MIN, SMA_MAX + 1, STEP))
        # pick short from [0 .. n-3], med from [short+1 .. n-2], long from [med+1 .. n-1]
        short_idx = random.randint(0, len(valid_values) - 3)
        med_idx = random.randint(short_idx + 1, len(valid_values) - 2)
        long_idx = random.randint(med_idx + 1, len(valid_values) - 1)
        return creator.Individual([
            valid_values[short_idx],
            valid_values[med_idx],
            valid_values[long_idx]
        ])
    
    toolbox.register("individual", create_valid_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, indpb=MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    hall_of_fame = tools.HallOfFame(maxsize=HALL_OF_FAME_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Add progress tracking
    start_time = time.time()
    last_update = start_time
    print("\nStarting optimization...")
    
    # Initialize logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + (stats.fields if stats else [])
    
    def update_progress(gen):
        nonlocal last_update
        current_time = time.time()
        elapsed = current_time - start_time
        if gen > 0:  # Only estimate after first generation
            time_per_gen = elapsed / gen
            remaining_gens = NUM_GENERATIONS - gen
            est_remaining = time_per_gen * remaining_gens
            print(f"\rGeneration {gen}/{NUM_GENERATIONS} - Elapsed: {elapsed:.1f}s - Est. remaining: {est_remaining:.1f}s", end="")
        last_update = current_time
    
    # Modify the eaSimple algorithm to include progress updates
    for gen in range(NUM_GENERATIONS):
        print(f"Starting generation {gen}...")  # Debugging print
        # Select the next generation individuals
        offspring = tools.selTournament(pop, len(pop), tournsize=3)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for i in range(1, len(offspring), 2):
            if i < len(offspring) - 1:
                if random.random() < CROSSOVER_PROB:
                    offspring[i], offspring[i+1] = custom_crossover(offspring[i], offspring[i+1])
            if i % 100 == 0:
                print(f"Crossover progress: {i}/{len(offspring)}")  # Crossover progress
        
        for i in range(len(offspring)):
            if random.random() < MUTATION_PROB:
                offspring[i], = custom_mutation(offspring[i], MUTATION_PROB)
            if i % 100 == 0:
                print(f"Mutation progress: {i}/{len(offspring)}")  # Mutation progress
        
        # Evaluate the individuals with an invalid fitness using parallel processing
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fitnesses = list(executor.map(toolbox.evaluate, invalid_ind))
        for idx, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
            ind.fitness.values = fit
            if idx % 100 == 0:
                print(f"Fitness evaluation progress: {idx}/{len(invalid_ind)}")  # Fitness evaluation progress
        
        # Replace the old population by the new one
        pop[:] = offspring
        
        # Update hall of fame
        hall_of_fame.update(pop)
        
        # Update statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        
        # Update progress
        update_progress(gen)
        print(f"Completed generation {gen}.")  # Debugging print
    
    print("\nOptimization complete!")
    
    # Debugging print to check post-optimization flow
    print("Post-optimization: Starting final operations...")
    
    best = hall_of_fame[0]
    best_short, best_med, best_long = best
    optimization_time = time.time() - overall_start_time
    print(f"\nGA completed in {optimization_time:.2f}s ({optimization_time/60:.2f} min)")
    print(f"Best params: short={best_short}, med={best_med}, long={best_long}, Sharpe={best.fitness.values[0]:.6f}")
    
    # Save top 20 from hall_of_fame (that satisfy short<med<long and not penalized)
    print("Saving top results...")  # Debugging print
    with open(RESULTS_FILE, 'w') as f:
        f.write("short_SMA,med_SMA,long_SMA,trades,sharpe_ratio\n")
        unique_results = []
        seen = set()
        for ind in hall_of_fame:
            s, m, l = ind
            if not (s < m < l):
                continue
            sharpe = ind.fitness.values[0]
            if abs(sharpe + 999999.0) < 0.1:
                continue
            ga_strat = AlligatorSMAStrategy(
                short_sma=s,
                med_sma=m,
                long_sma=l,
                big_point_value=big_point_value,
                slippage=slippage,
                capital=TRADING_CAPITAL,
                atr_period=ATR_PERIOD
            )
            temp = data.copy()
            temp = ga_strat.apply_strategy(temp, strategy_name="Strategy")
            if original_start_idx is not None:
                eval_data = temp.iloc[original_start_idx:].copy()
            else:
                eval_data = temp.copy()
            trades = eval_data['Position_Change_Strategy'].sum()
            key = (s, m, l)
            if key not in seen:
                seen.add(key)
                unique_results.append((s, m, l, trades, sharpe))
        
        unique_results.sort(key=lambda x: (x[0], x[1], x[2]))
        for s, m, l, trades, sharpe in unique_results:
            f.write(f"{s},{m},{l},{int(trades)},{sharpe:.6f}\n")
    print(f"Saved GA results to {RESULTS_FILE} ({len(unique_results)} entries)")  # Debugging print
    
    # Apply the best to full dataset
    print("Applying best parameters to full data…")  # Debugging print
    strategy.short_sma = best_short
    strategy.med_sma = best_med
    strategy.long_sma = best_long
    apply_start = time.time()
    full = strategy.apply_strategy(data.copy(), strategy_name="Strategy")
    apply_time = time.time() - apply_start
    print(f"Application done in {apply_time:.2f}s")  # Debugging print
    
    if original_start_idx is not None:
        final_eval = full.iloc[original_start_idx:].copy()
    else:
        final_eval = full.copy()
    
    split_index = int(len(final_eval) * TRAIN_TEST_SPLIT)
    
    # === Plotting ===
    print("Starting visualization...")  # Debugging print
    viz_start = time.time()
    plt.figure(figsize=(14, 16))
    
    # 1) Price + three SMAs + entry markers
    plt.subplot(3, 1, 1)
    plt.plot(final_eval.index, final_eval['Close'], label=f'{data_obj.symbol} Price', color='blue')
    plt.plot(final_eval.index, final_eval['SMA_Short_Strategy'], label=f'{best_short}-day SMA', color='orange')
    plt.plot(final_eval.index, final_eval['SMA_Med_Strategy'], label=f'{best_med}-day SMA', color='green')
    plt.plot(final_eval.index, final_eval['SMA_Long_Strategy'], label=f'{best_long}-day SMA', color='red')
    
    long_entries = (final_eval['Position_Dir_Strategy'] == 1) & final_eval['Position_Change_Strategy']
    short_entries = (final_eval['Position_Dir_Strategy'] == -1) & final_eval['Position_Change_Strategy']
    plt.scatter(final_eval.index[long_entries], final_eval.loc[long_entries, 'Close'],
                color='green', marker='^', s=50, label='Long Entry')
    plt.scatter(final_eval.index[short_entries], final_eval.loc[short_entries, 'Close'],
                color='red', marker='v', s=50, label='Short Entry')
    
    plt.legend()
    plt.title(f'{data_obj.symbol} GA‐Optimized Alligator SMA ({best_short}, {best_med}, {best_long})')
    plt.grid(True)
    
    # 2) Position size & ATR
    ax1 = plt.subplot(3, 1, 2)
    ax2 = ax1.twinx()
    ax1.plot(final_eval.index, final_eval['Position_Size_Strategy'],
             label='Position Size (# Contracts)', color='purple')
    ax1.set_ylabel('Position Size', color='purple')
    ax1.tick_params(axis='y', colors='purple')
    
    ax2.plot(final_eval.index, final_eval['ATR_Strategy'],
             label=f'ATR ({ATR_PERIOD}-day)', color='orange')
    ax2.set_ylabel(f'ATR ({ATR_PERIOD}-day)', color='orange')
    ax2.tick_params(axis='y', colors='orange')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title(f'Position Sizing Based on {ATR_PERIOD}-day ATR')
    ax1.grid(True)
    
    # 3) Cumulative P&L
    plt.subplot(3, 1, 3)
    strat_pnl = final_eval['Cumulative_PnL_Strategy'] - final_eval['Cumulative_PnL_Strategy'].iloc[0]
    plt.plot(final_eval.index, strat_pnl, label='Strategy P&L (full)', color='green')
    plt.plot(final_eval.index[split_index:], strat_pnl.iloc[split_index:],
             label=f'Out‐of‐sample P&L', color='purple')
    plt.axvline(final_eval.index[split_index], color='black', linestyle='--',
                label=f'Split {int(TRAIN_TEST_SPLIT*100)}%/{int((1-TRAIN_TEST_SPLIT)*100)}%')
    plt.axhline(0.0, color='gray', linestyle='-', label='Zero Line')
    
    plt.legend()
    plt.title('Strategy Performance (Dollar P&L)')
    plt.ylabel('P&L ($)')
    plt.grid(True)
    
    plt.tight_layout()
    save_plot(f'{SYMBOL}_GA_Optimized_AlligatorSMA.png')
    viz_time = time.time() - viz_start
    print(f"Visualization done in {viz_time:.2f}s")  # Debugging print
    
    # === Performance Metrics ===
    print("Calculating performance metrics...")  # Debugging print
    metrics_start = time.time()
    metrics = strategy.calculate_performance_metrics(
        final_eval,
        strategy_name="Strategy",
        train_test_split=TRAIN_TEST_SPLIT
    )
    metrics_time = time.time() - metrics_start
    
    market_cum = final_eval['Market_PnL_Strategy'].cumsum().iloc[-1]
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Symbol: {data_obj.symbol}")
    print(f"Big Point Value: {big_point_value}")
    print(f"ATR Period: {ATR_PERIOD} days")
    print(f"Capital: ${TRADING_CAPITAL:,}")
    print(f"Avg Position Size: {metrics['avg_position_size']:.2f}")
    print(f"Max Position Size: {metrics['max_position_size']:.0f}")
    print(f"Strategy Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"Market B&H P&L: ${market_cum:,.2f}")
    print(f"Outperformance: ${(metrics['total_pnl'] - market_cum):,.2f}")
    print(f"Sharpe (full): {metrics['sharpe_full']:.6f}")
    print(f"Sharpe (in‐sample): {metrics['sharpe_in_sample']:.6f}")
    print(f"Sharpe (out‐of‐sample): {metrics['sharpe_out_sample']:.6f}")
    print(f"Max Drawdown: ${abs(metrics['max_drawdown_dollars']):,.2f}")
    print("\n--- TRADE COUNTS ---")
    print(f"In‐sample trades: {metrics['in_sample_trades']}")
    print(f"Out‐of‐sample trades: {metrics['out_sample_trades']}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"In‐sample P&L: ${metrics['in_sample_pnl']:,.2f}")
    print(f"Out‐of‐sample P&L: ${metrics['out_sample_pnl']:,.2f}")
    
    print(f"\nBest GA params: short={best_short}, med={best_med}, long={best_long}, Sharpe={best.fitness.values[0]:.6f}")
    
    print("\n--- TIMING SUMMARY ---")
    total_time = time.time() - overall_start_time
    print(f"Load time: {load_time:.2f}s")
    print(f"Prep time: {prep_time:.2f}s")
    print(f"GA optimization time: {optimization_time:.2f}s ({optimization_time/60:.2f}m)")
    print(f"Strategy apply time: {apply_time:.2f}s")
    print(f"Visualization time: {viz_time:.2f}s")
    print(f"Metrics time: {metrics_time:.2f}s")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    
    print("\nEvaluation stats:")
    print(f"Unique evals: {evaluation_counter}")
    print(f"Cache hits: {cache_hit_counter}")
    print(f"Total checks: {evaluation_counter + cache_hit_counter}")
    print(f"Cache hit rate: {100 * cache_hit_counter/(evaluation_counter + cache_hit_counter):.2f}%")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
