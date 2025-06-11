import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime

from input import MIN_TRADES, MAX_TRADES, MIN_ELEMENTS_PER_CLUSTER, DEFAULT_NUM_CLUSTERS
from input import TICKER, START_DATE, END_DATE, TRAIN_TEST_SPLIT, INITIAL_CAPITAL, CONTRACT_SIZE, STOP_LOSS, SLIPPAGE
from SMA_Strategy import SMAStrategy
from cluster_visualization import compute_medoids, create_cluster_visualization


# Load the SMA simulation results
def analyze_sma_results(file_path='sma_all_results.txt'):
    print(f"Loading simulation results from {file_path}...")

    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Print basic information about the data
    print(f"Loaded {len(data)} simulation results")
    print(f"Short SMA range: {data['short_SMA'].min()} to {data['short_SMA'].max()}")
    print(f"Long SMA range: {data['long_SMA'].min()} to {data['long_SMA'].max()}")
    print(f"Sharpe ratio range: {data['sharpe_ratio'].min():.4f} to {data['sharpe_ratio'].max():.4f}")

    # Find the best Sharpe ratio
    best_idx = data['sharpe_ratio'].idxmax()
    best_short_sma = data.loc[best_idx, 'short_SMA']
    best_long_sma = data.loc[best_idx, 'long_SMA']
    best_sharpe = data.loc[best_idx, 'sharpe_ratio']
    best_trades = data.loc[best_idx, 'trades']

    print(f"\nBest parameters:")
    print(f"Short SMA: {best_short_sma}")
    print(f"Long SMA: {best_long_sma}")
    print(f"Sharpe Ratio: {best_sharpe:.6f}")
    print(f"Number of Trades: {best_trades}")

    # Create a pivot table for the heatmap
    heatmap_data = data.pivot_table(
        index='long_SMA',
        columns='short_SMA',
        values='sharpe_ratio'
    )

    # Create the heatmap visualization
    plt.figure(figsize=(12, 10))

    # Create a mask for invalid combinations (where short_SMA >= long_SMA)
    mask = np.zeros_like(heatmap_data, dtype=bool)
    for i, long_sma in enumerate(heatmap_data.index):
        for j, short_sma in enumerate(heatmap_data.columns):
            if short_sma >= long_sma:
                mask[i, j] = True

    # Plot the heatmap with the mask
    ax = sns.heatmap(
        heatmap_data,
        mask=mask,
        cmap='coolwarm',  # Blue to red colormap
        annot=False,  # Don't annotate each cell with its value
        fmt='.4f',
        linewidths=0,
        cbar_kws={'label': 'Sharpe Ratio'}
    )

    # Invert the y-axis so smaller long_SMA values are at the top
    ax.invert_yaxis()

    # Find the position of the best Sharpe ratio in the heatmap
    best_y = heatmap_data.index.get_loc(best_long_sma)
    best_x = heatmap_data.columns.get_loc(best_short_sma)

    # Add a star to mark the best Sharpe ratio
    # We need to add 0.5 to center the marker in the cell
    ax.add_patch(Circle((best_x + 0.5, best_y + 0.5), 0.4, facecolor='none',
                        edgecolor='white', lw=2))
    plt.plot(best_x + 0.5, best_y + 0.5, 'w*', markersize=10)

    # Set labels and title
    plt.title(f'SMA Optimization Heatmap (Best Sharpe: {best_sharpe:.4f} at {best_short_sma}/{best_long_sma})',
              fontsize=14)
    plt.xlabel('Short SMA (days)', fontsize=12)
    plt.ylabel('Long SMA (days)', fontsize=12)

    # Rotate tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add a text annotation for the best parameters
    plt.annotate(
        f'Best: Short={best_short_sma}, Long={best_long_sma}\nSharpe={best_sharpe:.4f}, Trades={best_trades}',
        xy=(best_x + 0.5, best_y + 0.5),
        xytext=(best_x + 5, best_y + 5),
        arrowprops=dict(arrowstyle="->", color='white'),
        color='white',
        backgroundcolor='black',
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7)
    )

    # Display and save the plot
    plt.tight_layout()
    plt.savefig('sma_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nHeatmap saved as 'sma_heatmap.png'")
    plt.show()

    # Return the data and best parameters
    return data, best_short_sma, best_long_sma, best_sharpe, best_trades


def plot_strategy_performance(short_sma, long_sma, top_medoids=None, contract_size=CONTRACT_SIZE,
                              fixed_stop_loss=STOP_LOSS, slippage=SLIPPAGE):
    """
    Plot the strategy performance using the best SMA parameters and include top medoids
    Uses the SMAStrategy class for consistent logic across the codebase

    Parameters:
    short_sma: int - The short SMA period
    long_sma: int - The long SMA period
    top_medoids: list - List of top medoids, each as (short_sma, long_sma, sharpe, trades)
    contract_size: int - Fixed number of contracts to trade
    fixed_stop_loss: float - Stop loss multiplier
    slippage: float - Slippage in price units
    """
    print(f"\n----- PLOTTING STRATEGY PERFORMANCE -----")
    print(f"Using Short SMA: {short_sma}, Long SMA: {long_sma}")
    print(f"Trading with fixed position size of {contract_size} contracts")
    if top_medoids:
        print(f"Including top {len(top_medoids)} medoids")

    # Download historical data
    print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
    data = yf.download(TICKER, start=START_DATE, end=END_DATE)

    # Simplify the data structure
    data.columns = data.columns.get_level_values(0)

    # Create a dictionary to store results for each strategy
    strategies = {
        'Best': {'short_sma': short_sma, 'long_sma': long_sma}
    }

    # Add medoids if provided
    if top_medoids:
        for i, medoid in enumerate(top_medoids, 1):
            strategies[f'Medoid {i}'] = {'short_sma': int(medoid[0]), 'long_sma': int(medoid[1])}

    # Apply the proper strategy for each parameter set
    for name, params in strategies.items():
        # Create a strategy instance for each parameter set
        sma_strategy = SMAStrategy(
            short_sma=params['short_sma'],
            long_sma=params['long_sma'],
            contract_size=contract_size,
            fixed_stop_loss=fixed_stop_loss,
            slippage=slippage
        )

        # Apply the strategy
        data = sma_strategy.apply_strategy(
            data,
            strategy_name=name,
            initial_capital=INITIAL_CAPITAL
        )

    # Calculate split index for in-sample/out-of-sample
    split_index = int(len(data) * TRAIN_TEST_SPLIT)
    split_date = data.index[split_index]

    # Create color palette for strategies
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink']

    # Create the performance visualization (removed the price/SMA chart)
    plt.figure(figsize=(14, 10))

    # Plot cumulative returns (now as the first subplot)
    plt.subplot(2, 1, 1)

    for i, (name, params) in enumerate(strategies.items()):
        color = colors[i % len(colors)]

        # Plot full period
        plt.plot(data.index, data[f'Cumulative_Returns_{name}'],
                 label=f'{name} ({params["short_sma"]}/{params["long_sma"]})', color=color)

        # Plot out-of-sample portion with thicker line
        plt.plot(data.index[split_index:], data[f'Cumulative_Returns_{name}'].iloc[split_index:],
                 color=color, linewidth=2.5, alpha=0.7)

    plt.axvline(x=split_date, color='black', linestyle='--',
                label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
    plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5,
                label=f'Initial Capital (${INITIAL_CAPITAL:,})')
    plt.legend(loc='upper left')
    plt.title('Strategy Performance (Multiple of Initial Capital)')
    plt.ylabel('Cumulative Return (x initial)')

    # Plot account balance (now as the second subplot)
    plt.subplot(2, 1, 2)

    for i, (name, params) in enumerate(strategies.items()):
        color = colors[i % len(colors)]

        # Plot full period
        plt.plot(data.index, data[f'Capital_{name}'],
                 label=f'{name} ({params["short_sma"]}/{params["long_sma"]})', color=color)

        # Plot out-of-sample portion with thicker line
        plt.plot(data.index[split_index:], data[f'Capital_{name}'].iloc[split_index:],
                 color=color, linewidth=2.5, alpha=0.7)

    plt.axvline(x=split_date, color='black', linestyle='--')
    plt.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='-',
                label=f'Initial Capital (${INITIAL_CAPITAL:,})')
    plt.legend(loc='upper left')
    plt.title(f'Account Balance (Starting with ${INITIAL_CAPITAL:,})')
    plt.ylabel('Capital (USD)')

    plt.tight_layout()
    plt.savefig('strategy_performance_with_medoids.png', dpi=300, bbox_inches='tight')
    print("\nStrategy performance chart with medoids saved as 'strategy_performance_with_medoids.png'")
    plt.show()

    # Create a list to store performance data for saving to file
    performance_data = []

    # Print detailed performance metrics for all strategies with in-sample and out-of-sample breakdown
    print("\n----- PERFORMANCE SUMMARY -----")

    # Open a file to save the performance summary
    with open('performance_summary_log.txt', 'w') as f:
        f.write("----- PERFORMANCE SUMMARY -----\n\n")

        # IN-SAMPLE PERFORMANCE
        print("\nIN-SAMPLE PERFORMANCE:")
        f.write("IN-SAMPLE PERFORMANCE:\n")
        header = f"{'Strategy':<10} | {'Short/Long':<10} | {'P&L':>12} | {'Return %':>10} | {'Sharpe':>7} | {'Trades':>6}"
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)
        f.write(separator + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")

        for name, params in strategies.items():
            short = params['short_sma']
            long = params['long_sma']

            # Get in-sample data
            in_sample = data.iloc[:split_index]

            # Calculate in-sample metrics
            in_sample_daily_pnl = in_sample[f'Daily_PnL_{name}']
            in_sample_cumulative_pnl = in_sample[f'Cumulative_PnL_{name}'].iloc[-1]
            in_sample_return_percent = (in_sample_cumulative_pnl / INITIAL_CAPITAL) * 100

            # Calculate Sharpe ratio (annualized)
            daily_returns = in_sample[f'Returns_{name}']  # Use log returns if available
            in_sample_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(
                252) if daily_returns.std() > 0 else 0

            # Count in-sample trades
            in_sample_trades = (in_sample[f'Signal_{name}'] != in_sample[f'Signal_{name}'].shift(1)).sum()
            if not pd.isna(in_sample[f'Signal_{name}'].iloc[0]):
                in_sample_trades -= 1

            # Create row text
            row = f"{name:<10} | {short:>4}/{long:<5} | ${in_sample_cumulative_pnl:>10,.2f} | {in_sample_return_percent:>9.2f}% | {in_sample_sharpe:>6.3f} | {in_sample_trades:>6}"

            # Print and save row
            print(row)
            f.write(row + "\n")

            # Store the data for later export
            performance_data.append({
                'Period': 'In-Sample',
                'Strategy': name,
                'Short_SMA': short,
                'Long_SMA': long,
                'PnL': in_sample_cumulative_pnl,
                'Return_Percent': in_sample_return_percent,
                'Sharpe': in_sample_sharpe,
                'Trades': in_sample_trades
            })

        print(separator)
        f.write(separator + "\n")

        # OUT-OF-SAMPLE PERFORMANCE
        print("\nOUT-OF-SAMPLE PERFORMANCE:")
        f.write("\nOUT-OF-SAMPLE PERFORMANCE:\n")
        print(separator)
        print(header)
        print(separator)
        f.write(separator + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")

        for name, params in strategies.items():
            short = params['short_sma']
            long = params['long_sma']

            # Get out-of-sample data
            out_sample = data.iloc[split_index:]

            # Calculate out-of-sample metrics
            out_sample_daily_pnl = out_sample[f'Daily_PnL_{name}']

            # Cumulative P&L just for the out-of-sample period
            out_sample_cumulative_pnl = out_sample[f'Daily_PnL_{name}'].sum()
            out_sample_return_percent = (out_sample_cumulative_pnl / INITIAL_CAPITAL) * 100

            # Calculate Sharpe ratio (annualized)
            daily_returns = out_sample[f'Returns_{name}']  # Use log returns if available
            out_sample_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(
                252) if daily_returns.std() > 0 else 0

            # Count out-of-sample trades
            out_sample_trades = (out_sample[f'Signal_{name}'] != out_sample[f'Signal_{name}'].shift(1)).sum()
            # If the first signal in out-sample is the same as the last signal in in-sample, it's not a new trade
            if out_sample[f'Signal_{name}'].iloc[0] == data[f'Signal_{name}'].iloc[split_index - 1]:
                out_sample_trades -= 1

            # Create row text
            row = f"{name:<10} | {short:>4}/{long:<5} | ${out_sample_cumulative_pnl:>10,.2f} | {out_sample_return_percent:>9.2f}% | {out_sample_sharpe:>6.3f} | {out_sample_trades:>6}"

            # Print and save row
            print(row)
            f.write(row + "\n")

            # Store the data for later export
            performance_data.append({
                'Period': 'Out-of-Sample',
                'Strategy': name,
                'Short_SMA': short,
                'Long_SMA': long,
                'PnL': out_sample_cumulative_pnl,
                'Return_Percent': out_sample_return_percent,
                'Sharpe': out_sample_sharpe,
                'Trades': out_sample_trades
            })

        print(separator)
        f.write(separator + "\n")

        # FULL PERIOD PERFORMANCE
        print("\nFULL PERIOD PERFORMANCE:")
        f.write("\nFULL PERIOD PERFORMANCE:\n")
        print(separator)
        print(header)
        print(separator)
        f.write(separator + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")

        for name, params in strategies.items():
            short = params['short_sma']
            long = params['long_sma']

            # Calculate full period metrics
            full_daily_pnl = data[f'Daily_PnL_{name}']
            full_cumulative_pnl = data[f'Cumulative_PnL_{name}'].iloc[-1]
            full_return_percent = (full_cumulative_pnl / INITIAL_CAPITAL) * 100

            # Calculate Sharpe ratio (annualized)
            daily_returns = data[f'Returns_{name}']  # Use log returns if available
            full_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

            # Count full period trades
            full_trades = (data[f'Signal_{name}'] != data[f'Signal_{name}'].shift(1)).sum()
            if not pd.isna(data[f'Signal_{name}'].iloc[0]):
                full_trades -= 1

            # Create row text
            row = f"{name:<10} | {short:>4}/{long:<5} | ${full_cumulative_pnl:>10,.2f} | {full_return_percent:>9.2f}% | {full_sharpe:>6.3f} | {full_trades:>6}"

            # Print and save row
            print(row)
            f.write(row + "\n")

            # Store the data for later export
            performance_data.append({
                'Period': 'Full',
                'Strategy': name,
                'Short_SMA': short,
                'Long_SMA': long,
                'PnL': full_cumulative_pnl,
                'Return_Percent': full_return_percent,
                'Sharpe': full_sharpe,
                'Trades': full_trades
            })

        print(separator)
        f.write(separator + "\n")

        # Add additional information to the performance summary
        f.write("\nADDITIONAL INFORMATION:\n")
        f.write(f"Ticker: {TICKER}\n")
        f.write(f"Date Range: {START_DATE} to {END_DATE}\n")
        f.write(f"Train/Test Split: {TRAIN_TEST_SPLIT * 100:.0f}%/{(1 - TRAIN_TEST_SPLIT) * 100:.0f}%\n")
        f.write(f"Split Date: {split_date}\n")
        f.write(f"Initial Capital: ${INITIAL_CAPITAL:,}\n")
        f.write(f"Contract Size: {contract_size}\n")
        f.write(f"Stop Loss: {(1 - fixed_stop_loss) * 100:.1f}%\n")
        f.write(f"Slippage: {slippage}\n")

    print(f"\nPerformance summary saved to 'performance_summary.txt'")

    # Save performance data to CSV for further analysis
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('performance_data.csv', index=False)
    print(f"Performance data saved to 'performance_data.csv'")

    # Save only the best strategy data if requested
    best_strategy_cols = [col for col in data.columns if 'Best' in col]
    best_strategy_data = data[['Open', 'High', 'Low', 'Close', 'Volume'] + best_strategy_cols]
    best_strategy_data.to_csv('strategy_data_best.csv')
    print(f"Complete best strategy data saved to 'strategy_data_best.csv'")

    return data


def cluster_analysis(file_path='sma_all_results.txt', min_trades=MIN_TRADES, max_trades=MAX_TRADES,
                     min_elements_per_cluster=MIN_ELEMENTS_PER_CLUSTER):
    """
    Perform clustering analysis on SMA optimization results to find robust parameter regions
    """
    print(f"\n----- CLUSTER ANALYSIS -----")
    print(f"Loading data from {file_path}...")

    # Load the data
    df = pd.read_csv(file_path)

    # Convert data to numpy array for easier processing
    X = df[['short_SMA', 'long_SMA', 'sharpe_ratio', 'trades']].values

    # Filter data by number of trades and ensure short_SMA < long_SMA
    X_filtered = X[(X[:, 0] < X[:, 1]) &  # short_SMA < long_SMA
                   (X[:, 3] >= min_trades) &  # trades >= min_trades
                   (X[:, 3] <= max_trades)]  # trades <= max_trades

    if len(X_filtered) == 0:
        print(
            f"No data points meet the criteria after filtering! Adjust min_trades ({min_trades}) and max_trades ({max_trades}).")
        return

    print(f"Filtered data to {len(X_filtered)} points with {min_trades}-{max_trades} trades")

    # Extract the fields for better scaling visibility
    short_sma_values = X_filtered[:, 0]
    long_sma_values = X_filtered[:, 1]
    sharpe_values = X_filtered[:, 2]
    trades_values = X_filtered[:, 3]

    print(f"Short SMA range: {short_sma_values.min()} to {short_sma_values.max()}")
    print(f"Long SMA range: {long_sma_values.min()} to {long_sma_values.max()}")
    print(f"Sharpe ratio range: {sharpe_values.min():.4f} to {sharpe_values.max():.4f}")
    print(f"Trades range: {trades_values.min()} to {trades_values.max()}")

    # Scale the data for clustering - using StandardScaler for each dimension
    # This addresses the issue where SMA values have much larger ranges than Sharpe ratio
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Print scaling info for verification
    print("\nScaled data information:")
    scaled_short = X_scaled[:, 0]
    scaled_long = X_scaled[:, 1]
    scaled_sharpe = X_scaled[:, 2]
    scaled_trades = X_scaled[:, 3]

    print(f"Scaled Short SMA range: {scaled_short.min():.4f} to {scaled_short.max():.4f}")
    print(f"Scaled Long SMA range: {scaled_long.min():.4f} to {scaled_long.max():.4f}")
    print(f"Scaled Sharpe ratio range: {scaled_sharpe.min():.4f} to {scaled_sharpe.max():.4f}")
    print(f"Scaled Trades range: {scaled_trades.min():.4f} to {scaled_trades.max():.4f}")

    # Determine number of clusters
    try:
        k = int(input(f"Enter number of clusters (default={DEFAULT_NUM_CLUSTERS}): ") or DEFAULT_NUM_CLUSTERS)
    except ValueError:
        print(f"Invalid input, using default value of {DEFAULT_NUM_CLUSTERS}")
        k = DEFAULT_NUM_CLUSTERS

    # Apply KMeans clustering
    print(f"Performing KMeans clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Get cluster labels
    labels = kmeans.labels_

    # Count elements per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))

    print("\nCluster sizes:")
    for cluster_id, size in cluster_sizes.items():
        print(f"Cluster {cluster_id}: {size} elements")

    # Filter clusters with enough elements
    valid_clusters = {i for i, count in cluster_sizes.items() if count >= min_elements_per_cluster}
    filtered_indices = np.array([i in valid_clusters for i in labels])

    # Filter data to only include points in valid clusters
    X_valid = X_filtered[filtered_indices]
    labels_valid = labels[filtered_indices]

    # Compute medoids
    print("Computing medoids...")
    medoids = compute_medoids(X_valid, labels_valid, valid_clusters)

    # Compute centroids
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)

    # Print raw centroids for debugging
    print("\nCluster Centroids (in original space):")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i}: Short SMA={centroid[0]:.2f}, Long SMA={centroid[1]:.2f}, "
              f"Sharpe={centroid[2]:.4f}, Trades={centroid[3]:.2f}")

    # Sort medoids by Sharpe ratio
    medoids_sorted = sorted(medoids, key=lambda x: x[2], reverse=True)
    top_medoids = medoids_sorted[:5]  # Get top 5 medoids by Sharpe ratio

    # Find max Sharpe ratio point overall
    max_sharpe_idx = np.argmax(df['sharpe_ratio'].values)
    max_sharpe_point = df.iloc[max_sharpe_idx][['short_SMA', 'long_SMA', 'sharpe_ratio', 'trades']].values

    # Print results
    print("\n----- CLUSTERING RESULTS -----")
    print(f"Max Sharpe point: Short SMA={int(max_sharpe_point[0])}, Long SMA={int(max_sharpe_point[1])}, "
          f"Sharpe={max_sharpe_point[2]:.4f}, Trades={int(max_sharpe_point[3])}")

    print("\nTop 5 Medoids (by Sharpe ratio):")
    for idx, medoid in enumerate(top_medoids, 1):
        print(f"Top {idx}: Short SMA={int(medoid[0])}, Long SMA={int(medoid[1])}, "
              f"Sharpe={medoid[2]:.4f}, Trades={int(medoid[3])}")

    # Create visualization with clustering results
    create_cluster_visualization(X_filtered, medoids, top_medoids, centroids, max_sharpe_point)

    return X_filtered, medoids, top_medoids, centroids, max_sharpe_point


if __name__ == "__main__":
    # Run the basic analysis first
    data, best_short, best_long, best_sharpe, best_trades = analyze_sma_results()

    # Run the cluster analysis to get medoids
    X_filtered, medoids, top_medoids, centroids, max_sharpe_point = cluster_analysis()

    # Plot strategy performance with the best parameters AND top medoids using fixed contract size
    plot_strategy_performance(best_short, best_long, top_medoids, contract_size=CONTRACT_SIZE)