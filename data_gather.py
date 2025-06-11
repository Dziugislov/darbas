import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
from datetime import datetime

# Import configuration
from input import *
from SMA_Strategy import SMAStrategy

# Download historical exchange rate data
print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Simplify the data structure by removing multi-level columns
data.columns = data.columns.get_level_values(0)

# Define the range of SMA periods to test
sma_range = range(SMA_MIN, SMA_MAX + 1, SMA_STEP)

print(f"Optimizing SMA parameters using range from {SMA_MIN} to {SMA_MAX} with step {SMA_STEP}...")
print(f"Using fixed stop-loss: {STOP_LOSS}, and dynamic trailing stops")
print(f"Trading with fixed contract size: {CONTRACT_SIZE}")

# Initialize the strategy
strategy = SMAStrategy(
    short_sma=0,  # Will be set during optimization
    long_sma=0,  # Will be set during optimization
    contract_size=CONTRACT_SIZE,
    fixed_stop_loss=STOP_LOSS,
    slippage=SLIPPAGE
)

# Run the optimization function to find the best SMA parameters
best_sma, best_sharpe, best_trades, all_results = strategy.optimize(
    data.copy(),
    sma_range,
    train_test_split=TRAIN_TEST_SPLIT,
    initial_capital=INITIAL_CAPITAL,
    results_file=RESULTS_FILE
)

print(f"Optimal SMA parameters: Short = {best_sma[0]} days, Long = {best_sma[1]} days")
print(f"In-sample Sharpe ratio = {best_sharpe:.4f}")
print(f"Number of trades with optimal parameters = {best_trades}")

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
plt.plot(data.index, data['Close'], label=f'{TICKER} Price', color='blue')
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

# Fixed stop loss triggers
fixed_stop_triggers = (data['Position_Strategy'].shift(1) != 0) & (data['Position_Strategy'] == 0) & \
                      ((data['Position_Strategy'].shift(1) == 1) & (
                                  data['Close'] <= data['Stop_Loss_Level_Strategy'].shift(1)) | \
                       (data['Position_Strategy'].shift(1) == -1) & (
                                   data['Close'] >= data['Stop_Loss_Level_Strategy'].shift(1)))

# Trailing stop triggers
trailing_stop_triggers = (data['Position_Strategy'].shift(1) != 0) & (data['Position_Strategy'] == 0) & \
                         (data['Trailing_Stop_Level_Strategy'].shift(1).notna()) & \
                         ((data['Position_Strategy'].shift(1) == 1) & (
                                     data['Close'] <= data['Trailing_Stop_Level_Strategy'].shift(1)) | \
                          (data['Position_Strategy'].shift(1) == -1) & (
                                      data['Close'] >= data['Trailing_Stop_Level_Strategy'].shift(1)))

# Plot the stop loss triggers
plt.scatter(data.index[fixed_stop_triggers], data['Close'][fixed_stop_triggers],
            color='red', marker='x', s=100, label='Fixed Stop Loss')

# Plot the trailing stop triggers
plt.scatter(data.index[trailing_stop_triggers], data['Close'][trailing_stop_triggers],
            color='purple', marker='d', s=80, label='Trailing Stop')

plt.legend()
plt.title(f'{TICKER} with Optimized SMA Strategy ({best_sma[0]}, {best_sma[1]}) and Dynamic Trailing Stops')

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
plt.savefig('sma_performance_with_dynamic_stops.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate performance metrics
metrics = strategy.calculate_performance_metrics(
    data,
    strategy_name="Strategy",
    train_test_split=TRAIN_TEST_SPLIT,
    initial_capital=INITIAL_CAPITAL
)

# Print summary statistics
print("\n--- PERFORMANCE SUMMARY WITH DYNAMIC TRAILING STOPS ---")
print(f"Fixed Contract Size: {CONTRACT_SIZE}")
print(f"Fixed Stop Loss: {STOP_LOSS:.2f} ({(1 - STOP_LOSS) * 100:.1f}%)")
print(f"Dynamic Trailing Stops: 2% at 5% profit, 5% at 10% profit, 10% at 20% profit")
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

print(f"\nAll simulation results saved to {RESULTS_FILE}")
print(
    f"Best parameters: Short SMA = {best_sma[0]}, Long SMA = {best_sma[1]}, Sharpe = {best_sharpe:.6f}, Trades = {best_trades}")
print("\nAnalysis complete!")