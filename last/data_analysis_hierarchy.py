import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist, squareform
import calendar
import os
import glob
import read_ts
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D
import matplotlib

from input import MIN_TRADES, MAX_TRADES, MIN_ELEMENTS_PER_CLUSTER, DEFAULT_NUM_CLUSTERS
from input import TICKER, START_DATE, END_DATE, TRAIN_TEST_SPLIT, INITIAL_CAPITAL
from SMA_Strategy import SMAStrategy

import json

def main():

    def load_parameters():
        try:
            with open("parameters.json", "r") as file:
                parameters = json.load(file)
                contract_multiplier = parameters["contract_multiplier"]
                dynamic_slippage = parameters["dynamic_slippage"]
                return contract_multiplier, dynamic_slippage
        except FileNotFoundError:
            print("Parameters file not found. Ensure it was saved correctly in data_gather.py.")
            return None, None

    # Load the parameters
    contract_multiplier, dynamic_slippage = load_parameters()

    print(f"Contract Multiplier: {contract_multiplier}")
    print(f"Dynamic Slippage: {dynamic_slippage}")

    # Setup paths
    WORKING_DIR = r"D:\dziug\Documents\darbas\last"
    DATA_DIR = os.path.join(WORKING_DIR, "data")
    SYMBOL = TICKER.replace('=F', '')


    # Assuming SYMBOL is defined in data_gather.py and passed or imported to data_analysis.py
    # If SYMBOL is not already defined, do this:
    SYMBOL = TICKER.replace('=F', '')  # Replace '=F' from ticker if necessary

    # Define the output folder where the plots will be saved
    output_dir = os.path.join("D:/dziug/Documents/darbas/last/output", SYMBOL)
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Function to save plots in the created folder
    def save_plot(plot_name):
        plt.savefig(os.path.join(output_dir, plot_name))  # Save plot to the symbol-specific folder
        plt.close()  # Close the plot to free up memory


    def find_futures_file(symbol, data_dir):
        """Find a data file for the specified futures symbol"""
        # Try common file pattern
        pattern = f"{symbol}_OHLCV_@*_minutes_1440_*.dat"
        files = glob.glob(os.path.join(data_dir, pattern))
        
        if not files:
            # Try alternative pattern if the first one doesn't match
            pattern = f"{symbol}*_OHLCV_*.dat"
            files = glob.glob(os.path.join(data_dir, pattern))
        
        if not files:
            # Last resort: check for any files containing the symbol
            pattern = f"*{symbol}*"
            files = glob.glob(os.path.join(data_dir, f"*{symbol}*.dat"))
        
        return files[0] if files else None


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
        print("\nHeatmap saved as 'sma_heatmap.png'")
        save_plot('Heatmap.png')

        # Return the data and best parameters
        return data, best_short_sma, best_long_sma, best_sharpe, best_trades


    def plot_strategy_performance(short_sma, long_sma, top_medoids=None, contract_size=contract_multiplier,
                                slippage=dynamic_slippage):
        """
        Plot the strategy performance using the best SMA parameters and include top medoids
        Uses the SMAStrategy class for consistent logic across the codebase

        Parameters:
        short_sma: int - The short SMA period
        long_sma: int - The long SMA period
        top_medoids: list - List of top medoids, each as (short_sma, long_sma, sharpe, trades)
        contract_size: int - Fixed number of contracts to trade
        slippage: float - Slippage in price units
        """
        print(f"\n----- PLOTTING STRATEGY PERFORMANCE -----")
        print(f"Using Short SMA: {short_sma}, Long SMA: {long_sma}")
        print(f"Trading with fixed position size of {contract_multiplier} contracts")
        if top_medoids:
            print(f"Including top {len(top_medoids)} medoids")

        # Load data from local files instead of Yahoo Finance
        print(f"Loading {TICKER} data from local files...")
        data_file = find_futures_file(SYMBOL, DATA_DIR)
        if not data_file:
            print(f"Error: No data file found for {TICKER} in {DATA_DIR}")
            exit(1)
        
        print(f"Found data file: {os.path.basename(data_file)}")
        print(f"File size: {os.path.getsize(data_file)} bytes")

        # Load the data from local file
        all_data = read_ts.read_ts_ohlcv_dat(data_file)
        data_obj = all_data[0]
        ohlc_data = data_obj.data.copy()

        # Convert the OHLCV data to the format expected by the strategy
        data = ohlc_data.rename(columns={
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        data.set_index('Date', inplace=True)

        # Filter data to match the date range if specified in input.py
        if START_DATE and END_DATE:
            data = data[(data.index >= pd.to_datetime(START_DATE)) & (data.index <= pd.to_datetime(END_DATE))]
            print(f"Filtered data to date range: {START_DATE} to {END_DATE}")

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
                contract_size=contract_multiplier,
                slippage=dynamic_slippage
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
        plt.title('Hierarchical, Strategy Performance (Multiple of Initial Capital)')
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
        print("\nStrategy performance chart with medoids saved as 'strategy_performance_with_medoids.png'")
        save_plot('strategy_performance_with_medoids.png')

        # Create a list to store performance data for saving to file
        performance_data = []

        # Print detailed performance metrics for all strategies with in-sample and out-of-sample breakdown
        print("\n----- PERFORMANCE SUMMARY -----")

        # Open a file to save the performance summary
        with open('performance_summary_hierarchical.txt', 'w') as f:
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
            f.write(f"Contract Size: {contract_multiplier}\n")
            f.write(f"Slippage: {dynamic_slippage}\n")

        print(f"\nPerformance summary saved to 'performance_summary_hierarchical.txt'")

        # Save performance data to CSV for further analysis
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv('performance_data_hierarchical.csv', index=False)
        print(f"Performance data saved to 'performance_data_hierarchical.csv'")

        # Save only the best strategy data if requested
        best_strategy_cols = [col for col in data.columns if 'Best' in col]
        best_strategy_data = data[['Open', 'High', 'Low', 'Close', 'Volume'] + best_strategy_cols]
        best_strategy_data.to_csv('strategy_data_best_hierarchical.csv')
        print(f"Complete best strategy data saved to 'strategy_data_best_hierarchical.csv'")

        return data


    def compute_hierarchical_medoids(X, labels, valid_clusters):
        """
        Compute medoids for each valid cluster from hierarchical clustering
        A medoid is the data point in a cluster that has the minimum average distance to all other points in the cluster

        Parameters:
        X: numpy array of shape (n_samples, n_features) - Original data points (not scaled)
        labels: numpy array of shape (n_samples,) - Cluster labels for each data point
        valid_clusters: set - Set of valid cluster IDs

        Returns:
        list of tuples - Each tuple contains (short_SMA, long_SMA, sharpe_ratio, trades) for each medoid
        """

        # Initialize list to store medoids for each cluster
        medoids = []

        # Process each valid cluster
        for cluster_id in valid_clusters:
            # Extract points belonging to this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_points = X[cluster_indices]

            if len(cluster_points) == 0:
                continue

            if len(cluster_points) == 1:
                # If only one point in cluster, it's the medoid
                medoids.append(tuple(cluster_points[0]))
                continue

            # Compute pairwise distances between all points in this cluster
            distances = cdist(cluster_points, cluster_points, metric='euclidean')

            # For each point, compute sum of distances to all other points
            total_distances = np.sum(distances, axis=1)

            # Find index of point with minimum total distance
            medoid_idx = np.argmin(total_distances)

            # Get the actual medoid point
            medoid = cluster_points[medoid_idx]

            # Store the medoid as a tuple
            medoids.append(tuple(medoid))

            # Print medoid info
            print(f"Cluster {cluster_id} medoid: Short SMA={int(medoid[0])}, Long SMA={int(medoid[1])}, "
                f"Sharpe={medoid[2]:.4f}, Trades={int(medoid[3])}")

        return medoids


    def create_dendrogram(X_scaled, method='ward', figsize=(12, 8), color_threshold=None, truncate_mode=None,
                        p=10, save_path=f'dendrogram.png'):
        """
        Create and display a dendrogram for hierarchical clustering

        Parameters:
        X_scaled: numpy array - Scaled data points
        method: str - Linkage method ('ward', 'complete', 'average', 'single')
        figsize: tuple - Figure size
        color_threshold: float - The threshold to apply for coloring the dendrogram
        truncate_mode: str - 'lastp' for last p branches, 'level' for no more than p levels
        p: int - Used with truncate_mode
        save_path: str - Path to save the dendrogram image

        Returns:
        Z: numpy array - The hierarchical clustering linkage matrix
        """


        # Compute the distance matrix
        dist_matrix = pdist(X_scaled, metric='euclidean')

        # Compute the linkage matrix
        Z = shc.linkage(dist_matrix, method=method)

        # Print some statistics about the linkage
        print(f"\nDendrogram using {method} linkage method:")
        print(f"Number of data points: {len(X_scaled)}")
        print(f"Cophenetic correlation: {shc.cophenet(Z, dist_matrix)[0]:.4f}")

        # Create a new figure
        plt.figure(figsize=figsize)

        # Set the background color
        plt.gca().set_facecolor('white')

        # Set a title
        plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)', fontsize=14)

        # Generate the dendrogram
        dendrogram = shc.dendrogram(
            Z,
            truncate_mode=truncate_mode,
            p=p,
            leaf_rotation=90.,
            leaf_font_size=10.,
            show_contracted=True,
            color_threshold=color_threshold
        )

        # Add labels and axes
        plt.xlabel('Sample Index or Cluster Size', fontsize=12)
        plt.ylabel('Distance', fontsize=12)

        # Add a horizontal line to indicate a distance threshold if specified
        if color_threshold is not None:
            plt.axhline(y=color_threshold, color='crimson', linestyle='--',
                        label=f'Threshold: {color_threshold:.2f}')
            plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dendrogram saved as '{save_path}'")

        # Show the plot
        save_plot('Dendogram.png')


    def hierarchical_cluster_analysis(file_path='sma_all_results.txt', min_trades=MIN_TRADES, max_trades=MAX_TRADES,
                                    min_elements_per_cluster=MIN_ELEMENTS_PER_CLUSTER):
        """
        Perform hierarchical clustering analysis on SMA optimization results to find robust parameter regions

        Parameters:
        file_path: str - Path to the file with SMA results
        min_trades: int - Minimum number of trades to consider
        max_trades: int - Maximum number of trades to consider
        min_elements_per_cluster: int - Minimum number of elements per cluster

        Returns:
        tuple - (X_filtered, medoids, top_medoids, max_sharpe_point, labels)
        """
        print(f"\n----- HIERARCHICAL CLUSTER ANALYSIS -----")
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
            return None, None, None, None, None

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

        # Create a dendrogram to help choose the number of clusters
        print("\nCreating dendrogram to visualize hierarchical structure...")
        linkage_method = 'ward'  # Ward minimizes the variance within clusters
        Z = create_dendrogram(X_scaled, method=linkage_method,
                            figsize=(12, 8),
                            save_path='hierarchical_dendrogram.png')

        # Determine optimal number of clusters or use default
        print(f"Using default number of clusters based on dendrogram: {DEFAULT_NUM_CLUSTERS}")
        k = DEFAULT_NUM_CLUSTERS

        # Apply hierarchical clustering
        print(f"Performing hierarchical clustering with {k} clusters using {linkage_method} linkage...")
        if linkage_method == 'ward':
            # Ward linkage requires euclidean distance
            hierarchical = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage_method
            )
        else:
            # For other linkage methods, we can specify affinity
            hierarchical = AgglomerativeClustering(
                n_clusters=k,
                affinity='euclidean',
                linkage=linkage_method
            )

        # Fit the model and get cluster labels
        labels = hierarchical.fit_predict(X_scaled)

        # Count elements per cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))

        print("\nCluster sizes:")
        for cluster_id, size in cluster_sizes.items():
            print(f"Cluster {cluster_id}: {size} elements")

        # Filter clusters with enough elements
        valid_clusters = {i for i, count in cluster_sizes.items() if count >= min_elements_per_cluster}
        if not valid_clusters:
            print(f"No clusters have at least {min_elements_per_cluster} elements! Reducing threshold to 1...")
            valid_clusters = set(unique_labels)
        else:
            print(f"Using {len(valid_clusters)} clusters with at least {min_elements_per_cluster} elements")

        filtered_indices = np.array([i in valid_clusters for i in labels])

        # Filter data to only include points in valid clusters
        X_valid = X_filtered[filtered_indices]
        labels_valid = labels[filtered_indices]

        # Compute medoids of hierarchical clusters
        print("Computing medoids for each cluster...")
        medoids = compute_hierarchical_medoids(X_valid, labels_valid, valid_clusters)

        # Find max Sharpe ratio point overall
        max_sharpe_idx = np.argmax(df['sharpe_ratio'].values)
        max_sharpe_point = df.iloc[max_sharpe_idx][['short_SMA', 'long_SMA', 'sharpe_ratio', 'trades']].values

        # Sort medoids by Sharpe ratio
        medoids_sorted = sorted(medoids, key=lambda x: x[2], reverse=True)
        top_medoids = medoids_sorted[:5]  # Get top 5 medoids by Sharpe ratio

        # Print results
        print("\n----- HIERARCHICAL CLUSTERING RESULTS -----")
        print(f"Max Sharpe point: Short SMA={int(max_sharpe_point[0])}, Long SMA={int(max_sharpe_point[1])}, "
            f"Sharpe={max_sharpe_point[2]:.4f}, Trades={int(max_sharpe_point[3])}")

        print("\nTop 5 Medoids (by Sharpe ratio):")
        for idx, medoid in enumerate(top_medoids, 1):
            print(f"Top {idx}: Short SMA={int(medoid[0])}, Long SMA={int(medoid[1])}, "
                f"Sharpe={medoid[2]:.4f}, Trades={int(medoid[3])}")

        # Create visualization with clustering results
        create_hierarchical_cluster_visualization(X_valid, medoids, top_medoids, max_sharpe_point, labels_valid)

        return X_valid, medoids, top_medoids, max_sharpe_point, labels_valid

    def bimonthly_out_of_sample_comparison(data, best_short_sma, best_long_sma, top_medoids, min_sharpe=0.2, 
                                        contract_size=contract_multiplier, slippage=dynamic_slippage):
        """
        Compare bimonthly (2-month) performance between the best Sharpe strategy and a portfolio of top medoids
        
        Parameters:
        data: DataFrame with market data
        best_short_sma: int - The short SMA period for the best Sharpe strategy
        best_long_sma: int - The long SMA period for the best Sharpe strategy
        top_medoids: list - List of top medoids, each as (short_sma, long_sma, sharpe, trades)
        min_sharpe: float - Minimum Sharpe ratio threshold for medoids to be included
        contract_size: int - Fixed number of contracts to trade
        slippage: float - Slippage in price units
        
        The function creates a portfolio of up to 3 top medoids (with Sharpe >= min_sharpe)
        and compares its performance to the best Sharpe strategy. The medoid portfolio results
        are normalized by dividing by the number of medoids used.
        """
        print(f"\n----- BIMONTHLY OUT-OF-SAMPLE COMPARISON -----")
        print(f"Best Sharpe: ({best_short_sma}/{best_long_sma})")
        
        # Handle the case where top_medoids is None
        if top_medoids is None:
            print("No medoids provided. Comparison cannot be performed.")
            return None
        
        # Check if top_medoids is a single tuple (rather than a list of tuples)
        if isinstance(top_medoids, tuple) and len(top_medoids) >= 3:
            # Convert single tuple to a list with one tuple
            medoids_list = [top_medoids]
        else:
            # Assume it's already a list
            medoids_list = top_medoids
        
        # Take at most 3 medoids and filter by minimum Sharpe
        filtered_medoids = []
        for m in medoids_list[:3]:
            # Check if we can access the required elements (support numpy arrays, tuples, and lists)
            try:
                # Extract Sharpe ratio and check if it meets the threshold
                short_sma = m[0]
                long_sma = m[1]
                sharpe = float(m[2])  # Convert to float to handle numpy types
                trades = m[3]
                
                if sharpe >= min_sharpe:
                    filtered_medoids.append(m)
            except (IndexError, TypeError) as e:
                print(f"Error processing medoid: {e}")
        
        if not filtered_medoids:
            print(f"No medoids have a Sharpe ratio >= {min_sharpe}. Comparison cannot be performed.")
            return None
        
        print(f"Creating portfolio of {len(filtered_medoids)} medoids with Sharpe ratio >= {min_sharpe}:")
        for i, medoid in enumerate(filtered_medoids, 1):
            print(f"Medoid {i}: ({int(medoid[0])}/{int(medoid[1])}) - Sharpe: {float(medoid[2]):.4f}")
        
        # Load data from local file if not already provided
        if data is None:
            print(f"Loading {TICKER} data from local files...")
            data_file = find_futures_file(SYMBOL, DATA_DIR)
            if not data_file:
                print(f"Error: No data file found for {TICKER} in {DATA_DIR}")
                exit(1)
            
            print(f"Found data file: {os.path.basename(data_file)}")
            print(f"File size: {os.path.getsize(data_file)} bytes")

            # Load the data from local file
            all_data = read_ts.read_ts_ohlcv_dat(data_file)
            data_obj = all_data[0]
            ohlc_data = data_obj.data.copy()

            # Convert the OHLCV data to the format expected by the strategy
            data = ohlc_data.rename(columns={
                'datetime': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            data.set_index('Date', inplace=True)
            # Filter data to match the date range if specified in input.py
            if START_DATE and END_DATE:
                data = data[(data.index >= pd.to_datetime(START_DATE)) & (data.index <= pd.to_datetime(END_DATE))]
                print(f"Filtered data to date range: {START_DATE} to {END_DATE}")
            # Simplify the data structure
            data.columns = data.columns.get_level_values(0)
        
        # Create strategies
        strategies = {
            'Best': {'short_sma': best_short_sma, 'long_sma': best_long_sma}
        }
        
        # Add filtered medoids
        for i, medoid in enumerate(filtered_medoids, 1):
            strategies[f'Medoid_{i}'] = {'short_sma': int(medoid[0]), 'long_sma': int(medoid[1])}
        
        # Apply each strategy to the data
        for name, params in strategies.items():
            strategy = SMAStrategy(
                short_sma=params['short_sma'],
                long_sma=params['long_sma'],
                contract_size=contract_multiplier,
                slippage=dynamic_slippage
            )
            
            # Apply the strategy
            data = strategy.apply_strategy(
                data.copy(),
                strategy_name=name,
                initial_capital=INITIAL_CAPITAL
            )
        
        # Get the out-of-sample split date
        split_index = int(len(data) * TRAIN_TEST_SPLIT)
        split_date = data.index[split_index]
        print(f"Out-of-sample period starts on: {split_date.strftime('%Y-%m-%d')}")
        
        # Get out-of-sample data
        oos_data = data.iloc[split_index:].copy()
        
        # Add a year and bimonthly period columns for grouping (each year has 6 bimonthly periods)
        oos_data['year'] = oos_data.index.year.astype(int)
        oos_data['bimonthly'] = ((oos_data.index.month - 1) // 2 + 1).astype(int)
        
        # Create simplified period labels with just the start month (YYYY-MM)
        oos_data['period_label'] = oos_data.apply(
            lambda row: f"{int(row['year'])}-{int((row['bimonthly'] - 1) * 2 + 1):02d}",
            axis=1
        )
        
        # Create a DataFrame to store bimonthly Sharpe ratios
        bimonthly_sharpe = []
        
        # Group by year and bimonthly period, calculate Sharpe ratio for each period
        for period_label, group in oos_data.groupby('period_label'):
            # Skip periods with too few trading days
            if len(group) < 10:
                continue
                
            # Create a bimonthly result entry
            year, start_month = period_label.split('-')
            year = int(year)
            start_month = int(start_month)
            
            bimonthly_result = {
                'period_label': period_label,
                'date': pd.Timestamp(year=year, month=start_month, day=15),  # Middle of first month in period
                'trading_days': len(group),
            }
            
            # Calculate Sharpe ratio for each strategy in this period
            for name in strategies.keys():
                # Get returns for this strategy in this period
                returns = group[f'Returns_{name}']
                
                # Calculate Sharpe ratio (annualized)
                if len(returns) > 1 and returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    sharpe = 0
                    
                bimonthly_result[f'{name}_sharpe'] = sharpe
                bimonthly_result[f'{name}_return'] = returns.sum() * 100  # Percentage return for the period
            
            # Calculate the normalized average of medoid Sharpe ratios
            medoid_sharpes = [bimonthly_result[f'Medoid_{i}_sharpe'] for i in range(1, len(filtered_medoids) + 1)]
            bimonthly_result['Avg_Medoid_sharpe'] = sum(medoid_sharpes) / len(filtered_medoids)
            
            # Calculate the normalized average of medoid returns
            medoid_returns = [bimonthly_result[f'Medoid_{i}_return'] for i in range(1, len(filtered_medoids) + 1)]
            bimonthly_result['Avg_Medoid_return'] = sum(medoid_returns) / len(filtered_medoids)
            
            bimonthly_sharpe.append(bimonthly_result)
        
        # Convert to DataFrame
        bimonthly_sharpe_df = pd.DataFrame(bimonthly_sharpe)
        
        # Sort the DataFrame by date for proper chronological display
        if not bimonthly_sharpe_df.empty:
            bimonthly_sharpe_df = bimonthly_sharpe_df.sort_values('date')
        else:
            print(f"WARNING: No bimonthly periods found for {SYMBOL}. Cannot create chart.")
            return None
        
        # Check if there are enough bimonthly periods for a meaningful chart
        if len(bimonthly_sharpe_df) < 2:
            print(f"WARNING: Only {len(bimonthly_sharpe_df)} bimonthly periods found for {SYMBOL}. "
                f"This may be insufficient for creating a meaningful chart.")
        
        # Add rounded values to dataframe for calculations
        bimonthly_sharpe_df['Best_sharpe_rounded'] = np.round(bimonthly_sharpe_df['Best_sharpe'], 2)
        bimonthly_sharpe_df['Avg_Medoid_sharpe_rounded'] = np.round(bimonthly_sharpe_df['Avg_Medoid_sharpe'], 2)
        
        # Print detailed comparison of Sharpe ratios
        print("\nDetailed Sharpe ratio comparison by period:")
        print(f"{'Period':<12} | {'Best Sharpe':>12} | {'Medoid Portfolio':>16} | {'Difference':>12} | {'Portfolio Wins':<14}")
        print("-" * 80)
        
        for idx, row in bimonthly_sharpe_df.iterrows():
            period = row['period_label']
            best_sharpe = row['Best_sharpe']
            avg_medoid_sharpe = row['Avg_Medoid_sharpe']
            best_rounded = row['Best_sharpe_rounded']
            avg_medoid_rounded = row['Avg_Medoid_sharpe_rounded']
            
            diff = avg_medoid_sharpe - best_sharpe
            portfolio_wins = avg_medoid_sharpe > best_sharpe
            
            print(f"{period:<12} | {best_sharpe:12.6f} | {avg_medoid_sharpe:16.6f} | {diff:12.6f} | {portfolio_wins!s:<14}")
        
        # Calculate win rate using raw values
        portfolio_wins = sum(bimonthly_sharpe_df['Avg_Medoid_sharpe'] > bimonthly_sharpe_df['Best_sharpe'])
        total_periods = len(bimonthly_sharpe_df)
        win_percentage = (portfolio_wins / total_periods) * 100 if total_periods > 0 else 0
        
        # Calculate win rate using rounded values (for alternative comparison)
        rounded_wins = sum(bimonthly_sharpe_df['Avg_Medoid_sharpe_rounded'] > bimonthly_sharpe_df['Best_sharpe_rounded'])
        rounded_win_percentage = (rounded_wins / total_periods) * 100 if total_periods > 0 else 0
        
        print(f"\nBimonthly periods analyzed: {total_periods}")
        print(f"Medoid Portfolio Wins: {portfolio_wins} of {total_periods} periods ({win_percentage:.2f}%)")
        print(f"Using rounded values (2 decimal places): {rounded_wins} of {total_periods} periods ({rounded_win_percentage:.2f}%)")
        
        try:
            # Create a bar plot to compare bimonthly Sharpe ratios
            plt.figure(figsize=(14, 8))
            
            # Set up x-axis dates
            x = np.arange(len(bimonthly_sharpe_df))
            width = 0.35  # Width of the bars
            
            # Create bars
            plt.bar(x - width/2, bimonthly_sharpe_df['Best_sharpe'], width, 
                label=f'Best Sharpe ({best_short_sma}/{best_long_sma})', color='blue')
            plt.bar(x + width/2, bimonthly_sharpe_df['Avg_Medoid_sharpe'], width, 
                label=f'Medoid Portfolio ({len(filtered_medoids)} strategies)', color='green')
            
            # Add a horizontal line at Sharpe = 0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Create medoid description for the title
            medoid_desc = ", ".join([f"({int(m[0])}/{int(m[1])})" for m in filtered_medoids])
            
            # Customize the plot - USING ROUNDED PERCENTAGES
            plt.title(f'Hierarchy Bimonthly Sharpe Ratio Comparison (Out-of-Sample Period)\n' + 
                    f'Medoid Portfolio [{medoid_desc}] outperformed {rounded_win_percentage:.2f}% of the time', 
                    fontsize=14)
            plt.xlabel('Bimonthly Period (Start Month)', fontsize=12)
            plt.ylabel('Sharpe Ratio (Annualized)', fontsize=12)
            
            # Simplified x-tick labels with just the period start month
            plt.xticks(x, bimonthly_sharpe_df['period_label'], rotation=45)
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Create legend with both strategies
            best_sharpe_patch = plt.Rectangle((0, 0), 1, 1, fc="blue")
            medoid_patch = plt.Rectangle((0, 0), 1, 1, fc="green")
            
            plt.legend(
                [best_sharpe_patch, medoid_patch],
                [f'Best Sharpe ({best_short_sma}/{best_long_sma})', 
                f'Medoid Portfolio ({len(filtered_medoids)} strategies)'],
                loc='upper right', 
                frameon=True, 
                fancybox=True, 
                framealpha=0.9,
                fontsize=10
            )
            
            # Add a text box with ROUNDED win percentage
            plt.annotate(f'Medoid Portfolio Win Rate: {rounded_win_percentage:.2f}%\n'
                        f'({rounded_wins} out of {total_periods} periods)\n'
                        f'Portfolio: {len(filtered_medoids)} medoids with Sharpe ≥ {min_sharpe}',
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=12)
            
            plt.tight_layout()
            
            # Use the save_plot function consistently
            chart_filename = f'Bimonthly_Charts_hierarchy_{SYMBOL}.png'
            save_plot(chart_filename)
            print(f"\nBimonthly comparison chart saved to output directory as '{chart_filename}'")
            
            # Save the bimonthly data to CSV with both raw and rounded values
            csv_filename = f'bimonthly_sharpe_portfolio_{len(filtered_medoids)}_medoids_{SYMBOL}.csv'
            bimonthly_sharpe_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
            print(f"Bimonthly comparison data saved to output directory as '{csv_filename}'")
            
        except Exception as e:
            print(f"ERROR saving hierarchical bimonthly chart for {SYMBOL}: {str(e)}")
        
        # Return the bimonthly Sharpe ratio data
        return bimonthly_sharpe_df

    def create_hierarchical_cluster_visualization(X, medoids, top_medoids, max_sharpe_point, labels):
        """
        Create visualization of hierarchical clustering results with scatter plots

        Parameters:
        X: numpy array - Original data points (not scaled)
        medoids: list of tuples - Cluster medoids
        top_medoids: list of tuples - Top medoids by Sharpe ratio
        max_sharpe_point: tuple - Point with maximum Sharpe ratio
        labels: numpy array - Cluster labels

        Returns:
        None
        """


        # Create a new figure with 2 subplots (1x2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Get the number of unique clusters
        n_clusters = len(np.unique(labels))

        # Create a colormap with distinct colors for each cluster
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        # Plot 1: Short SMA vs Long SMA (colored by cluster)
        for i, color in enumerate(colors):
            # Get indices of points in cluster i
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue

            # Plot the points without labels to avoid cluttering the legend
            ax1.scatter(
                cluster_points[:, 0],  # Short SMA
                cluster_points[:, 1],  # Long SMA
                s=50, c=[color], alpha=0.7
                # No label parameter here to keep them out of the legend
            )

        # Plot the medoids with black edges
        for i, medoid in enumerate(medoids):
            ax1.scatter(
                medoid[0],  # Short SMA
                medoid[1],  # Long SMA
                s=150, c='white', alpha=1,
                edgecolors='black', linewidths=2,
                marker='o'
            )

        # Plot the top medoids with star markers
        for i, medoid in enumerate(top_medoids):
            ax1.scatter(
                medoid[0],  # Short SMA
                medoid[1],  # Long SMA
                s=200, c='gold', alpha=1,
                edgecolors='black', linewidths=1.5,
                marker='*'
            )
            
        # Plot the max Sharpe point with a diamond marker
        ax1.scatter(
            max_sharpe_point[0],  # Short SMA
            max_sharpe_point[1],  # Long SMA
            s=250, c='red', alpha=1,
            edgecolors='black', linewidths=2,
            marker='D'
        )

        # Add labels and title
        ax1.set_xlabel('Short SMA (days)', fontsize=12)
        ax1.set_ylabel('Long SMA (days)', fontsize=12)
        ax1.set_title('Hierarchical Clusters: Short SMA vs Long SMA', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Add a diagonal line where short_SMA = long_SMA
        max_val = max(X[:, 0].max(), X[:, 1].max()) * 1.1
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

        # Create a simplified legend with just the key elements
        custom_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
                markeredgecolor='black', markeredgewidth=2, markersize=10, label='Cluster Medoid'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                markeredgecolor='black', markersize=10, label='Top Medoid'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                markeredgecolor='black', markersize=10, label='Best Sharpe Point')
        ]
        ax1.legend(handles=custom_handles, loc='upper right')
        
        # Plot 2: Short SMA vs Long SMA (colored by Sharpe ratio)
        scatter = ax2.scatter(
            X[:, 0],  # Short SMA
            X[:, 1],  # Long SMA
            s=50,
            c=X[:, 2],  # Sharpe ratio for color
            cmap='coolwarm',
            alpha=0.7
        )

        # Plot the top medoids with star markers
        for i, medoid in enumerate(top_medoids):
            ax2.scatter(
                medoid[0],  # Short SMA
                medoid[1],  # Long SMA
                s=200, c='gold', alpha=1,
                edgecolors='black', linewidths=1.5,
                marker='*'
            )

        # Plot the max Sharpe point with a diamond marker
        ax2.scatter(
            max_sharpe_point[0],  # Short SMA
            max_sharpe_point[1],  # Long SMA
            s=250, c='red', alpha=1,
            edgecolors='black', linewidths=2,
            marker='D'
        )

        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Sharpe Ratio', fontsize=12)

        # Add labels and title
        ax2.set_xlabel('Short SMA (days)', fontsize=12)
        ax2.set_ylabel('Long SMA (days)', fontsize=12)
        ax2.set_title('Sharpe Ratio Heatmap with Medoids', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Add a diagonal line where short_SMA = long_SMA
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

        # Create a custom legend for the second plot
        custom_handles = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                markeredgecolor='black', markersize=10, label='Top Medoid'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                markeredgecolor='black', markersize=10, label='Best Sharpe Point')
        ]

        ax2.legend(handles=custom_handles, loc='upper right')

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(f'hierarchical_clusters.png', dpi=300, bbox_inches='tight')
        print("\nHierarchical cluster visualization saved as 'hierarchical_clusters.png'")

        # Show the plot
        save_plot('hierarchical_clusters.png')


    if __name__ == "__main__":
        # Set matplotlib backend explicitly
        matplotlib.use('TkAgg')  # Or try 'Qt5Agg' if this doesn't work
        
        print("Starting SMA strategy analysis with hierarchical clustering...")
        
        # Run the basic analysis first
        data, best_short, best_long, best_sharpe, best_trades = analyze_sma_results()

        # Run the hierarchical cluster analysis to get medoids
        X_filtered, medoids, top_medoids, max_sharpe_point, labels = hierarchical_cluster_analysis()
        
        if X_filtered is None or medoids is None:
            print("Error: Hierarchical cluster analysis failed.")
            exit(1)

        # Plot strategy performance with the best parameters AND top medoids using fixed contract size
        market_data = plot_strategy_performance(best_short, best_long, top_medoids, contract_size=contract_multiplier)
        
        # Run the bimonthly out-of-sample comparison between best Sharpe and top medoids
        if top_medoids and len(top_medoids) > 0:
            print("\nPerforming bimonthly out-of-sample comparison with hierarchical clustering...")
            monthly_sharpe_df = bimonthly_out_of_sample_comparison(
                market_data, 
                best_short, 
                best_long, 
                top_medoids,  # Pass the entire list of top medoids, not just the first one
                contract_size=contract_multiplier
            )
        else:
            print("No top medoids found. Cannot run bimonthly comparison.")
            
        print("\nAnalysis complete! All plots have been displayed and saved.")

if __name__ == "__main__":
    main()
