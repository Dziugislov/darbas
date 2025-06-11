import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import numpy as np

# Import configuration
from input import *
from Alligator_Strategy import AlligatorStrategy

# Adjust start date to account for JAW_MAX days lag
start_date_obj = datetime.strptime(SIMULATION_START_DATE, '%Y-%m-%d')
adjusted_start_date_obj = start_date_obj - relativedelta(days=JAW_MAX*1.5)  # Extra buffer for weekends/holidays
adjusted_start_date = adjusted_start_date_obj.strftime('%Y-%m-%d')

print(f"Adjusted start date to {adjusted_start_date} to account for {JAW_MAX} days SMA calculation period")
print(f"Downloading {TICKER} data from {adjusted_start_date} to {END_DATE}...")
data = yf.download(TICKER, start=adjusted_start_date, end=END_DATE)

# Simplify the data structure by removing multi-level columns
data.columns = data.columns.get_level_values(0)

# Create lists of parameters to test
jaw_range = range(JAW_MIN, JAW_MAX + 1, JAW_STEP)
lips_ratio_range = np.arange(LIPS_RATIO_MIN, LIPS_RATIO_MAX + 0.001, LIPS_RATIO_STEP)
teeth_range = range(TEETH_MIN, TEETH_MAX + 1, TEETH_STEP)

print(f"Optimizing Classic Alligator parameters with standard shifts...")
print(f"Jaw range: {JAW_MIN} to {JAW_MAX} with step {JAW_STEP}, shift = 8 periods")
print(f"Lips ratio range: {LIPS_RATIO_MIN} to {LIPS_RATIO_MAX} with step {LIPS_RATIO_STEP}, shift = 3 periods")
print(f"Teeth range: {TEETH_MIN} to {TEETH_MAX} with step {TEETH_STEP}, shift = 5 periods")
print(f"Using fixed contract size: {CONTRACT_SIZE}")

# Initialize the strategy with placeholder values
strategy = AlligatorStrategy(
    jaw_period=0,      # Will be set during optimization
    teeth_period=INITIAL_TEETH,    # Initial teeth period for first optimization
    lips_period=0,     # Will be set during optimization
    contract_size=CONTRACT_SIZE,
    slippage=SLIPPAGE
)

# Filter to the actual trading period for optimization
trading_start_idx = data.index.searchsorted(SIMULATION_START_DATE)
trading_data = data.iloc[trading_start_idx:].copy()

#----------------------------------------------------------
# STEP 1: Optimize Jaw and Lips (keeping Teeth fixed)
#----------------------------------------------------------
print("\n--- STEP 1: Optimizing Jaw and Lips periods ---")

# Run the first optimization to find the best Jaw and Lips parameters
best_jaw_lips_params, best_jaw_lips_sharpe, best_jaw_lips_trades, jaw_lips_results = strategy.optimize_jaw_lips(
    trading_data.copy(),
    jaw_range,
    lips_ratio_range,
    INITIAL_TEETH,  # Initial teeth period
    train_test_split=TRAIN_TEST_SPLIT,
    initial_capital=INITIAL_CAPITAL,
    results_file=JAW_LIPS_RESULTS_FILE
)

print(f"Optimal Jaw-Lips parameters: Jaw = {best_jaw_lips_params[0]} days, Lips = {best_jaw_lips_params[2]} days")
print(f"In-sample Sharpe ratio = {best_jaw_lips_sharpe:.4f}")
print(f"Number of trades with optimal Jaw-Lips parameters = {best_jaw_lips_trades}")

#----------------------------------------------------------
# STEP 2: Optimize Teeth (using best Jaw and Lips from step 1)
#----------------------------------------------------------
print("\n--- STEP 2: Optimizing Teeth period with fixed Jaw and Lips ---")

# Set the best jaw and lips periods from the previous optimization
best_jaw = best_jaw_lips_params[0]
best_lips = best_jaw_lips_params[2]

# Run the second optimization to find the best Teeth parameter
best_params, best_sharpe, best_trades, teeth_results = strategy.optimize_teeth(
    trading_data.copy(),
    best_jaw,
    teeth_range,
    best_lips,
    train_test_split=TRAIN_TEST_SPLIT,
    initial_capital=INITIAL_CAPITAL,
    results_file=TEETH_RESULTS_FILE
)

print(f"Optimal Alligator parameters after both optimizations:")
print(f"Jaw = {best_params[0]} days (shift = 8), Teeth = {best_params[1]} days (shift = 5), Lips = {best_params[2]} days (shift = 3)")
print(f"In-sample Sharpe ratio = {best_sharpe:.4f}")
print(f"Number of trades with optimal parameters = {best_trades}")

# Update strategy with the best parameters
strategy.jaw_period = best_params[0]
strategy.teeth_period = best_params[1]
strategy.lips_period = best_params[2]

# Apply the best parameters found from optimization to the entire dataset
full_results = strategy.apply_strategy(data.copy(), initial_capital=INITIAL_CAPITAL)

# Filter to the actual trading period for visualization and evaluation
results = full_results.iloc[trading_start_idx:].copy()

# Calculate split index for in-sample/out-of-sample
split_index = int(len(results) * TRAIN_TEST_SPLIT)

# Plot the price and Alligator lines
plt.figure(figsize=(14, 10))

# Plot price and Alligator components
plt.subplot(2, 1, 1)
plt.plot(results.index, results['Close'], label=f'{TICKER} Price', color='black')
plt.plot(results.index, results['Jaw_Strategy'], label=f'Jaw ({best_params[0]}-day SMA, shift=8)', color='blue')
plt.plot(results.index, results['Teeth_Strategy'], label=f'Teeth ({best_params[1]}-day SMA, shift=5)', color='red')
plt.plot(results.index, results['Lips_Strategy'], label=f'Lips ({best_params[2]}-day SMA, shift=3)', color='green')

# Plot long position entries
long_entries = (results['Position_Strategy'] == 1) & (results['Position_Strategy'].shift(1) != 1)
if long_entries.any():
    plt.scatter(results.index[long_entries], results['Entry_Price_Strategy'][long_entries],
                color='green', marker='^', s=100, label='Long Entry')

# Plot short position entries
short_entries = (results['Position_Strategy'] == -1) & (results['Position_Strategy'].shift(1) != -1)
if short_entries.any():
    plt.scatter(results.index[short_entries], results['Entry_Price_Strategy'][short_entries],
                color='red', marker='v', s=100, label='Short Entry')

# Plot exits
exits = (results['Position_Strategy'] == 0) & (results['Position_Strategy'].shift(1) != 0)
if exits.any():
    plt.scatter(results.index[exits], results['Close'][exits],
                color='black', marker='x', s=100, label='Position Exit')

plt.legend()
plt.title(f'{TICKER} with Classic Alligator Strategy (Jaw={best_params[0]}/Teeth={best_params[1]}/Lips={best_params[2]}) - Shifted')

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Plot the performance
plt.subplot(2, 1, 2)

# Plot the cumulative returns
plt.plot(results.index, results['Cumulative_Returns_Strategy'], label='Cumulative Returns (full period)', color='green')
plt.plot(results.index[split_index:], results['Cumulative_Returns_Strategy'].iloc[split_index:],
         label=f'Cumulative Returns (last {int((1 - TRAIN_TEST_SPLIT) * 100)}% out-of-sample)', color='purple')
plt.axvline(x=results.index[split_index], color='black', linestyle='--',
            label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
plt.axhline(y=1.0, color='gray', linestyle='-', label=f'Initial Capital (${INITIAL_CAPITAL:,})')
plt.legend()
plt.title('Classic Alligator Strategy Performance (Multiple of Initial Capital)')
plt.ylabel('Cumulative Return (x initial)')

# Format x-axis dates for performance chart
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('classic_alligator_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate performance metrics
metrics = strategy.calculate_performance_metrics(
    results,
    strategy_name="Strategy",
    train_test_split=TRAIN_TEST_SPLIT,
    initial_capital=INITIAL_CAPITAL
)

# Print summary statistics
print("\n--- PERFORMANCE SUMMARY (CLASSIC ALLIGATOR STRATEGY) ---")
print(f"Simulation period: {SIMULATION_START_DATE} to {END_DATE}")
print(f"Alligator Parameters: Jaw={best_params[0]} (shift=8), Teeth={best_params[1]} (shift=5), Lips={best_params[2]} (shift=3)")
print(f"Fixed Contract Size: {CONTRACT_SIZE}")
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

print(f"\nJaw-Lips optimization results saved to {JAW_LIPS_RESULTS_FILE}")
print(f"Teeth optimization results saved to {TEETH_RESULTS_FILE}")
print(f"Best parameters: Jaw={best_params[0]}, Teeth={best_params[1]}, Lips={best_params[2]}, Sharpe={best_sharpe:.6f}, Trades={best_trades}")
print("\nAnalysis complete!")