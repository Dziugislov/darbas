import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Import from existing files
from input import TICKER, START_DATE, END_DATE, TRAIN_TEST_SPLIT, INITIAL_CAPITAL, CONTRACT_SIZE, STOP_LOSS, SLIPPAGE
from SMA_Strategy import SMAStrategy
from data_analysis import analyze_sma_results, hierarchical_cluster_analysis


def split_into_monthly_periods(data):
    """
    Split a DataFrame with DatetimeIndex into monthly chunks

    Parameters:
    data: DataFrame with DatetimeIndex

    Returns:
    list: List of DataFrames, each covering approximately 1 month
    """
    periods = []

    # Get start and end dates
    start_date = data.index[0]
    end_date = data.index[-1]

    # Initialize current period start
    current_start = start_date

    while current_start < end_date:
        # Calculate end of current 1-month period
        # Add 1 month (approximately 30 days)
        current_end = current_start + pd.DateOffset(days=30)

        # Ensure we don't go beyond the available data
        if current_end > end_date:
            # If less than 1 month remains, disregard this period as per requirements
            break

        # Get data for this period
        period_data = data.loc[current_start:current_end]

        # Only add if we have data in this period
        if len(period_data) > 0:
            periods.append(period_data)

        # Update start date for next period
        current_start = current_end + pd.DateOffset(days=1)

    return periods


def calculate_sharpe_for_period(returns):
    """
    Calculate annualized Sharpe ratio for a period

    Parameters:
    returns: Series - Daily returns

    Returns:
    float: Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0

    # Scale by sqrt(252) to annualize (252 trading days per year)
    return returns.mean() / returns.std() * np.sqrt(252)


def run_periodic_comparison():
    """
    Main function to run the bi-monthly out-of-sample analysis for best Sharpe parameters and best medoid
    """
    print(f"Running periodic comparison analysis...")

    # First, get the best Sharpe parameters from the sma_all_results.txt file
    print("Identifying best Sharpe parameters...")
    _, best_short_sma, best_long_sma, _, _ = analyze_sma_results()

    # Then get the best medoid (medoid 1) from hierarchical clustering
    print("Identifying best medoid parameters...")
    _, _, top_medoids, _, _ = hierarchical_cluster_analysis()

    # The first medoid in top_medoids is medoid 1 (the best one)
    medoid1_short_sma = int(top_medoids[0][0])  # Extract short SMA from medoid 1
    medoid1_long_sma = int(top_medoids[0][1])  # Extract long SMA from medoid 1

    print(f"Best Sharpe parameters: Short SMA = {best_short_sma}, Long SMA = {best_long_sma}")
    print(f"Medoid 1 parameters: Short SMA = {medoid1_short_sma}, Long SMA = {medoid1_long_sma}")

    # Download the full historical data
    print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
    full_data = yf.download(TICKER, start=START_DATE, end=END_DATE)

    # Simplify the data structure
    full_data.columns = full_data.columns.get_level_values(0)

    # Initialize SMA strategies for both parameter sets - using the Always In version
    best_sharpe_strategy = SMAStrategy(
        short_sma=best_short_sma,
        long_sma=best_long_sma,
        contract_size=CONTRACT_SIZE,
        slippage=SLIPPAGE
    )

    medoid1_strategy = SMAStrategy(
        short_sma=medoid1_short_sma,
        long_sma=medoid1_long_sma,
        contract_size=CONTRACT_SIZE,
        slippage=SLIPPAGE
    )

    # Apply the strategies to the entire dataset first
    # This ensures we have properly calculated SMAs across the entire history
    print("Applying strategies to full dataset (always in market)...")
    full_data_best = best_sharpe_strategy.apply_strategy(
        full_data.copy(),
        strategy_name="BestSharpe",
        initial_capital=INITIAL_CAPITAL
    )

    full_data_medoid = medoid1_strategy.apply_strategy(
        full_data.copy(),
        strategy_name="Medoid1",
        initial_capital=INITIAL_CAPITAL
    )

    # Calculate the split index for the out-of-sample portion
    split_index = int(len(full_data) * TRAIN_TEST_SPLIT)

    # Get out-of-sample portion for both datasets
    out_sample_best = full_data_best.iloc[split_index:].copy()
    out_sample_medoid = full_data_medoid.iloc[split_index:].copy()

    print(f"Full dataset: {len(full_data)} data points")
    print(f"Out-of-sample dataset: {len(out_sample_best)} data points")

    # Split the out-of-sample data into monthly periods based on dates
    out_sample_dates = out_sample_best.index
    periods = split_into_monthly_periods(out_sample_best)
    print(f"Split out-of-sample data into {len(periods)} monthly periods")

    # Initialize lists to store results
    period_labels = []
    best_sharpe_values = []
    medoid1_sharpe_values = []
    best_trades_count = []
    medoid1_trades_count = []

    # Process each period
    for i, period_data in enumerate(periods):
        period_start = period_data.index[0].strftime('%Y-%m-%d')
        period_end = period_data.index[-1].strftime('%Y-%m-%d')
        period_label = f"{period_start}\nto\n{period_end}"
        period_labels.append(period_label)

        print(f"\nAnalyzing period {i + 1}: {period_start} to {period_end} ({len(period_data)} data points)")

        # Get corresponding medoid period data
        period_medoid = out_sample_medoid.loc[period_start:period_end]

        # Count trades in this period for best Sharpe
        best_trades = (period_data['Signal_BestSharpe'] != period_data['Signal_BestSharpe'].shift(1)).sum()
        if pd.notna(period_data['Signal_BestSharpe'].iloc[0]):
            best_trades -= 1
        best_trades_count.append(best_trades)

        # Count trades in this period for medoid 1
        medoid_trades = (period_medoid['Signal_Medoid1'] != period_medoid['Signal_Medoid1'].shift(1)).sum()
        if pd.notna(period_medoid['Signal_Medoid1'].iloc[0]):
            medoid_trades -= 1
        medoid1_trades_count.append(medoid_trades)

        # Calculate Sharpe ratio for best parameters in this period
        best_sharpe_returns = period_data['Returns_BestSharpe']

        # Only calculate Sharpe if we have non-zero std dev of returns
        if best_sharpe_returns.std() > 0:
            best_sharpe_value = best_sharpe_returns.mean() / best_sharpe_returns.std() * np.sqrt(252)
        else:
            # If we have trades but std=0, assign a small value
            if best_trades > 0:
                best_sharpe_value = 0.0001 if best_sharpe_returns.mean() >= 0 else -0.0001
            else:
                best_sharpe_value = 0

        best_sharpe_values.append(best_sharpe_value)

        # Calculate Sharpe ratio for medoid 1 in this period
        medoid1_returns = period_medoid['Returns_Medoid1']

        # Only calculate Sharpe if we have non-zero std dev of returns
        if medoid1_returns.std() > 0:
            medoid1_sharpe_value = medoid1_returns.mean() / medoid1_returns.std() * np.sqrt(252)
        else:
            # If we have trades but std=0, assign a small value
            if medoid_trades > 0:
                medoid1_sharpe_value = 0.0001 if medoid1_returns.mean() >= 0 else -0.0001
            else:
                medoid1_sharpe_value = 0

        medoid1_sharpe_values.append(medoid1_sharpe_value)

        print(f"Best Sharpe ({best_short_sma}/{best_long_sma}): Trades={best_trades}, Sharpe={best_sharpe_value:.4f}")
        print(
            f"Medoid 1 ({medoid1_short_sma}/{medoid1_long_sma}): Trades={medoid_trades}, Sharpe={medoid1_sharpe_value:.4f}")

        # Calculate PnL for this period
        best_pnl = period_data['Daily_PnL_BestSharpe'].sum()
        medoid1_pnl = period_medoid['Daily_PnL_Medoid1'].sum()

        print(f"Best Sharpe period PnL: ${best_pnl:.2f}")
        print(f"Medoid 1 period PnL: ${medoid1_pnl:.2f}")

    # Create the bar chart visualization
    create_sharpe_comparison_chart(
        period_labels,
        best_sharpe_values,
        medoid1_sharpe_values,
        best_short_sma,
        best_long_sma,
        medoid1_short_sma,
        medoid1_long_sma
    )

    # Create the bar chart visualization
    create_sharpe_comparison_chart(
        period_labels,
        best_sharpe_values,
        medoid1_sharpe_values,
        best_short_sma,
        best_long_sma,
        medoid1_short_sma,
        medoid1_long_sma,
        best_trades_count,
        medoid1_trades_count,
        save_only=False
    )

    # Return the results for further analysis if needed
    return {
        'period_labels': period_labels,
        'best_sharpe_values': best_sharpe_values,
        'medoid1_sharpe_values': medoid1_sharpe_values,
        'best_short_sma': best_short_sma,
        'best_long_sma': best_long_sma,
        'medoid1_short_sma': medoid1_short_sma,
        'medoid1_long_sma': medoid1_long_sma,
        'best_trades_count': best_trades_count,
        'medoid1_trades_count': medoid1_trades_count,
        'medoid_better_percentage': medoid_better_percentage
    }


def create_sharpe_comparison_chart(period_labels, best_sharpe_values, medoid1_sharpe_values,
                                   best_short_sma, best_long_sma, medoid1_short_sma, medoid1_long_sma,
                                   best_trades_count=None, medoid1_trades_count=None, save_only=False):
    """
    Create a bar chart comparing Sharpe ratios for each period

    Parameters:
    period_labels: list - Labels for each period
    best_sharpe_values: list - Sharpe ratios for best parameters
    medoid1_sharpe_values: list - Sharpe ratios for medoid 1
    best_short_sma, best_long_sma: Best Sharpe parameters
    medoid1_short_sma, medoid1_long_sma: Medoid 1 parameters
    best_trades_count: list - Number of trades for best parameters in each period
    medoid1_trades_count: list - Number of trades for medoid 1 in each period
    save_only: bool - Whether to only save the chart without displaying it
    """
    # Set up the figure
    plt.figure(figsize=(16, 10))

    # Set the width of bars
    width = 0.35

    # Create simplified labels with just the end date
    simplified_labels = []
    for period in period_labels:
        # Split by "to" and take the second part which is the end date
        if "to" in period:
            end_date = period.split("to")[1].strip()
            simplified_labels.append(end_date)
        else:
            simplified_labels.append(period)

    # Set up x positions for the bars
    x = np.arange(len(simplified_labels))

    # Create the bars
    best_bars = plt.bar(x - width / 2, best_sharpe_values, width,
                        label=f'Best Sharpe ({best_short_sma}/{best_long_sma})', color='blue')
    medoid_bars = plt.bar(x + width / 2, medoid1_sharpe_values, width,
                          label=f'Medoid 1 ({medoid1_short_sma}/{medoid1_long_sma})', color='orange')

    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Calculate average Sharpe values (for table only, not displayed on graph)
    avg_best_sharpe = np.mean(best_sharpe_values)
    avg_medoid1_sharpe = np.mean(medoid1_sharpe_values)

    # Add trade count annotations if provided
    if best_trades_count and medoid1_trades_count:
        for i, (count1, count2) in enumerate(zip(best_trades_count, medoid1_trades_count)):
            # Only add annotation if there are trades
            if count1 > 0:
                plt.annotate(f"{count1}",
                             xy=(x[i] - width / 2, best_sharpe_values[i]),
                             xytext=(0, 5),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=8)

            if count2 > 0:
                plt.annotate(f"{count2}",
                             xy=(x[i] + width / 2, medoid1_sharpe_values[i]),
                             xytext=(0, 5),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=8)

    # Add labels, title, and legend
    plt.xlabel('Out-of-Sample Periods (1 Month Each)', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.title('Comparison of Sharpe Ratios (Always In Market): Best Sharpe Parameters vs Medoid 1', fontsize=14)
    plt.xticks(x, simplified_labels, rotation=45, ha='right')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Add a grid for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Save and display the figure
    plt.savefig('monthly_sharpe_comparison.png', dpi=300, bbox_inches='tight')
    print("\nMonthly Sharpe ratio comparison chart saved as 'monthly_sharpe_comparison.png'")
    plt.show()

    # Also create a table with the values for reference
    print("\nMonthly Sharpe Ratio Comparison:")
    print("-" * 100)
    header = f"{'Period':<25} | {'Best Sharpe':<15} | {'Medoid 1':<15} | {'Difference':<15}"
    if best_trades_count and medoid1_trades_count:
        header += f" | {'Best Trades':<10} | {'Medoid Trades':<10}"
    print(header)
    print("-" * 100)

    for i, period in enumerate(period_labels):
        # Replace newlines with spaces for clean table printing
        period_clean = period.replace('\n', ' ')
        diff = best_sharpe_values[i] - medoid1_sharpe_values[i]

        row = f"{period_clean:<25} | {best_sharpe_values[i]:>15.4f} | {medoid1_sharpe_values[i]:>15.4f} | {diff:>15.4f}"

        if best_trades_count and medoid1_trades_count:
            row += f" | {best_trades_count[i]:>10} | {medoid1_trades_count[i]:>12}"

        print(row)

    print("-" * 100)

    avg_row = f"{'Average':<25} | {avg_best_sharpe:>15.4f} | {avg_medoid1_sharpe:>15.4f} | {avg_best_sharpe - avg_medoid1_sharpe:>15.4f}"

    if best_trades_count and medoid1_trades_count:
        avg_best_trades = sum(best_trades_count) / len(best_trades_count) if best_trades_count else 0
        avg_medoid_trades = sum(medoid1_trades_count) / len(medoid1_trades_count) if medoid1_trades_count else 0
        avg_row += f" | {avg_best_trades:>10.1f} | {avg_medoid_trades:>12.1f}"

    print(avg_row)
    print("-" * 100)


if __name__ == "__main__":
    results = run_periodic_comparison()

    # Calculate the percentage of time medoid1 has better Sharpe ratio
    medoid_better_count = sum(
        1 for m, b in zip(results['medoid1_sharpe_values'], results['best_sharpe_values']) if m > b)
    total_periods = len(results['medoid1_sharpe_values'])
    medoid_better_percentage = (medoid_better_count / total_periods) * 100 if total_periods > 0 else 0

    print(
        f"\nMedoid 1 had a higher Sharpe ratio than Best Sharpe in {medoid_better_count} out of {total_periods} periods.")
    print(f"Percentage of time Medoid 1 outperformed Best Sharpe: {medoid_better_percentage:.2f}%")