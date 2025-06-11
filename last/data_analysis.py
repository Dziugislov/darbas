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
import calendar
import read_ts

import os
import matplotlib.pyplot as plt
import matplotlib



from input import MIN_TRADES, MAX_TRADES, MIN_ELEMENTS_PER_CLUSTER, DEFAULT_NUM_CLUSTERS
from input import TICKER, START_DATE, END_DATE, TRAIN_TEST_SPLIT, INITIAL_CAPITAL
from SMA_Strategy import SMAStrategy

import os
import glob

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


    # Added functions from cluster_visualization.py
    def compute_medoids(X, labels, valid_clusters):
        """Compute medoids for each cluster (point with minimum distance to all other points in cluster)"""
        medoids = []

        for cluster_id in valid_clusters:
            # Get points in this cluster
            cluster_points = X[labels == cluster_id]

            if len(cluster_points) == 0:
                continue

            # Calculate pairwise distances within cluster
            min_total_distance = float('inf')
            medoid = None

            for i, point1 in enumerate(cluster_points):
                total_distance = 0
                for j, point2 in enumerate(cluster_points):
                    # Calculate Euclidean distance between points
                    distance = np.sqrt(np.sum((point1 - point2) ** 2))
                    total_distance += distance

                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    medoid = point1

            if medoid is not None:
                medoids.append(medoid)

        return medoids

    def cluster_analysis(file_path='sma_all_results.txt', min_trades=MIN_TRADES, max_trades=MAX_TRADES,
                        min_elements_per_cluster=MIN_ELEMENTS_PER_CLUSTER):
        """
        Perform clustering analysis on SMA optimization results to find robust parameter regions
        Now clusters based on short_SMA, long_SMA, and sharpe_ratio only (not trades)
        """
        print(f"\n----- CLUSTER ANALYSIS -----")
        print(f"Loading data from {file_path}...")

        # Load the data
        df = pd.read_csv(file_path)

        # Convert data to numpy array for easier processing
        X_full = df[['short_SMA', 'long_SMA', 'sharpe_ratio', 'trades']].values

        # Filter data by number of trades and ensure short_SMA < long_SMA
        X_filtered_full = X_full[(X_full[:, 0] < X_full[:, 1]) &  # short_SMA < long_SMA
                    (X_full[:, 3] >= min_trades) &  # trades >= min_trades
                    (X_full[:, 3] <= max_trades)]  # trades <= max_trades

        if len(X_filtered_full) == 0:
            print(
                f"No data points meet the criteria after filtering! Adjust min_trades ({min_trades}) and max_trades ({max_trades}).")
            return None, None, None, None, None

        # Create a version with only the 3 dimensions for clustering
        X_filtered = X_filtered_full[:, 0:3]  # Only short_SMA, long_SMA, and sharpe_ratio

        print(f"Filtered data to {len(X_filtered)} points with {min_trades}-{max_trades} trades")

        # Extract the fields for better scaling visibility
        short_sma_values = X_filtered[:, 0]
        long_sma_values = X_filtered[:, 1]
        sharpe_values = X_filtered[:, 2]
        trades_values = X_filtered_full[:, 3]

        print(f"Short SMA range: {short_sma_values.min()} to {short_sma_values.max()}")
        print(f"Long SMA range: {long_sma_values.min()} to {long_sma_values.max()}")
        print(f"Sharpe ratio range: {sharpe_values.min():.4f} to {sharpe_values.max():.4f}")
        print(f"Trades range: {trades_values.min()} to {trades_values.max()}")

        # Scale the data for clustering - using StandardScaler for each dimension
        # This addresses the issue where SMA values have much larger ranges than Sharpe ratio
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)  # Only scale the 3 dimensions we use for clustering

        # Print scaling info for verification
        print("\nScaled data information:")
        scaled_short = X_scaled[:, 0]
        scaled_long = X_scaled[:, 1]
        scaled_sharpe = X_scaled[:, 2]

        print(f"Scaled Short SMA range: {scaled_short.min():.4f} to {scaled_short.max():.4f}")
        print(f"Scaled Long SMA range: {scaled_long.min():.4f} to {scaled_long.max():.4f}")
        print(f"Scaled Sharpe ratio range: {scaled_sharpe.min():.4f} to {scaled_sharpe.max():.4f}")

        # Determine number of clusters
        print(f"Using default number of clusters: {DEFAULT_NUM_CLUSTERS}")
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
        X_valid = X_filtered_full[filtered_indices]  # Use full data to get trades too
        labels_valid = labels[filtered_indices]

        # Compute medoids using the existing method that expects 4D data
        print("Computing medoids...")
        medoids = compute_medoids(X_valid, labels_valid, valid_clusters)

        # Compute centroids (only for the 3 dimensions we clustered on)
        centroids_scaled = kmeans.cluster_centers_
        # Create a version that can be inverse-transformed (matches the original dimension count)
        centroids = np.zeros((centroids_scaled.shape[0], 3))
        centroids[:, 0:3] = scaler.inverse_transform(centroids_scaled)

        # Print raw centroids for debugging
        print("\nCluster Centroids (in original space):")
        for i, centroid in enumerate(centroids):
            if i in valid_clusters:  # Only show valid clusters
                print(f"Centroid {i}: Short SMA={centroid[0]:.2f}, Long SMA={centroid[1]:.2f}, "
                    f"Sharpe={centroid[2]:.4f}")

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
        create_cluster_visualization(X_filtered_full, medoids, top_medoids, centroids, max_sharpe_point)

        return X_filtered_full, medoids, top_medoids, centroids, max_sharpe_point

    def create_cluster_visualization(X_filtered, medoids, top_medoids, centroids, max_sharpe_point):
        """Create a continuous heatmap visualization with cluster centers overlaid"""

        print("Creating cluster visualization...")

        # Load the full dataset to ensure we have complete data for the heatmap
        data = pd.read_csv('sma_all_results.txt')

        # Create a pivot table for the heatmap using ALL data points
        heatmap_data = data.pivot_table(
            index='long_SMA',
            columns='short_SMA',
            values='sharpe_ratio'
        )

        # Create the heatmap visualization
        plt.figure(figsize=(12, 10))

        # Create mask for invalid combinations (where short_SMA >= long_SMA)
        mask = np.zeros_like(heatmap_data, dtype=bool)
        for i, long_sma in enumerate(heatmap_data.index):
            for j, short_sma in enumerate(heatmap_data.columns):
                if short_sma >= long_sma:
                    mask[i, j] = True

        # Plot the base heatmap with ALL data (continuous like the original)
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

        # Plot max Sharpe point (Green Star)
        # Note: We need to convert from data values to plot coordinates
        best_x_pos = np.where(heatmap_data.columns == max_sharpe_point[0])[0][0] + 0.5
        best_y_pos = np.where(heatmap_data.index == max_sharpe_point[1])[0][0] + 0.5
        plt.scatter(best_x_pos, best_y_pos, marker='*', color='lime', s=200,
                    edgecolor='black', zorder=5)

        # Plot all medoids (Black Squares)
        for medoid in medoids:
            try:
                x_pos = np.where(heatmap_data.columns == medoid[0])[0][0] + 0.5
                y_pos = np.where(heatmap_data.index == medoid[1])[0][0] + 0.5
                plt.scatter(x_pos, y_pos, marker='s', color='black', s=75, zorder=4)
            except IndexError:
                print(f"Warning: Medoid at ({medoid[0]}, {medoid[1]}) not found in heatmap coordinates")

        # Plot top 5 medoids (Purple Diamonds)
        for medoid in top_medoids:
            try:
                x_pos = np.where(heatmap_data.columns == medoid[0])[0][0] + 0.5
                y_pos = np.where(heatmap_data.index == medoid[1])[0][0] + 0.5
                plt.scatter(x_pos, y_pos, marker='D', color='purple', s=100, zorder=5)
            except IndexError:
                print(f"Warning: Top medoid at ({medoid[0]}, {medoid[1]}) not found in heatmap coordinates")

        # Plot centroids (Blue Circles) - IMPROVED VERSION
        print(f"Plotting {len(centroids)} centroids...")
        centroids_plotted = 0

        for i, centroid in enumerate(centroids):
            # Get the actual raw centroid values
            short_sma = centroid[0]
            long_sma = centroid[1]

            print(f"Centroid {i}: raw values = ({short_sma}, {long_sma})")

            # First try exact values
            try:
                if (short_sma in heatmap_data.columns) and (long_sma in heatmap_data.index) and (short_sma < long_sma):
                    x_pos = np.where(heatmap_data.columns == short_sma)[0][0] + 0.5
                    y_pos = np.where(heatmap_data.index == long_sma)[0][0] + 0.5
                    plt.scatter(x_pos, y_pos, marker='o', color='blue', s=75, zorder=4)
                    centroids_plotted += 1
                    continue
            except (IndexError, TypeError):
                pass

            # Try rounded values
            try:
                short_sma_rounded = int(round(short_sma))
                long_sma_rounded = int(round(long_sma))

                print(f"  Rounded: ({short_sma_rounded}, {long_sma_rounded})")

                if (short_sma_rounded in heatmap_data.columns) and (long_sma_rounded in heatmap_data.index) and (
                        short_sma_rounded < long_sma_rounded):
                    x_pos = np.where(heatmap_data.columns == short_sma_rounded)[0][0] + 0.5
                    y_pos = np.where(heatmap_data.index == long_sma_rounded)[0][0] + 0.5
                    plt.scatter(x_pos, y_pos, marker='o', color='blue', s=75, zorder=4)
                    centroids_plotted += 1
                    continue
            except (IndexError, TypeError):
                pass

            # Try finding nearest valid point
            try:
                # Find nearest valid parameter values
                short_options = np.array(heatmap_data.columns)
                long_options = np.array(heatmap_data.index)

                # Find nearest short SMA
                short_idx = np.argmin(np.abs(short_options - short_sma))
                short_nearest = short_options[short_idx]

                # Find nearest long SMA
                long_idx = np.argmin(np.abs(long_options - long_sma))
                long_nearest = long_options[long_idx]

                print(f"  Nearest: ({short_nearest}, {long_nearest})")

                # Check if valid
                if short_nearest < long_nearest:
                    x_pos = np.where(heatmap_data.columns == short_nearest)[0][0] + 0.5
                    y_pos = np.where(heatmap_data.index == long_nearest)[0][0] + 0.5
                    plt.scatter(x_pos, y_pos, marker='o', color='blue', s=75, zorder=4, alpha=0.7)
                    centroids_plotted += 1
                else:
                    print(f"  Invalid nearest parameters (short >= long): {short_nearest} >= {long_nearest}")
            except (IndexError, TypeError) as e:
                print(f"  Error finding nearest point: {e}")

        print(f"Successfully plotted {centroids_plotted} out of {len(centroids)} centroids")

        # Create custom legend
        max_sharpe_handle = mlines.Line2D([], [], color='lime', marker='*', linestyle='None',
                                        markersize=15, markeredgecolor='black', label='Max Sharpe')
        medoid_handle = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                                    markersize=10, label='Medoids')
        top_medoid_handle = mlines.Line2D([], [], color='purple', marker='D', linestyle='None',
                                        markersize=10, label='Top 5 Medoids')
        centroid_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                        markersize=10, label='Centroids')

        # Add legend
        plt.legend(handles=[max_sharpe_handle, medoid_handle, top_medoid_handle, centroid_handle],
                loc='best')

        # Set labels and title
        plt.title('SMA Parameter Clustering Analysis (Sharpe Ratio)', fontsize=14)
        plt.xlabel('Short SMA (days)', fontsize=12)
        plt.ylabel('Long SMA (days)', fontsize=12)

        # Rotate tick labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Display plot
        plt.tight_layout()
        save_plot('Cluster_Analysis.png')


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

        # Display the plot
        plt.tight_layout()
        save_plot('Heatmap.png')

        # Return the data and best parameters
        return data, best_short_sma, best_long_sma, best_sharpe, best_trades


    def plot_strategy_performance(short_sma, long_sma, top_medoids=None, contract_size=contract_multiplier,
                                dynamic_slippage=dynamic_slippage):
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

        # Download historical data FIXME
        # Download historical data from Yahoo Finance (replace with your local data)
        # Original line:
        # data = yf.download(TICKER, start=START_DATE, end=END_DATE)

        # New logic to load local data:
        print(f"Loading {TICKER} data from local files...")
        data_file = find_futures_file(SYMBOL, DATA_DIR)  # Assuming `find_futures_file` is defined similarly
        if not data_file:
            print(f"Error: No data file found for {TICKER} in {DATA_DIR}")
            exit(1)

        print(f"Found data file: {os.path.basename(data_file)}")
        print(f"File size: {os.path.getsize(data_file)} bytes")

        # Load the data from local file (use the same method you used in data_gather.py)
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
        plt.title('K-means/medoid, Strategy Performance (Multiple of Initial Capital)')
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
        save_plot('Multiple_Plots')

        # Print detailed performance metrics for all strategies with in-sample and out-of-sample breakdown
        print("\n----- PERFORMANCE SUMMARY -----")

        # IN-SAMPLE PERFORMANCE
        print("\nIN-SAMPLE PERFORMANCE:")
        header = f"{'Strategy':<10} | {'Short/Long':<10} | {'P&L':>12} | {'Return %':>10} | {'Sharpe':>7} | {'Trades':>6}"
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)

        performance_data = []

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

            # Print row
            print(row)

            # Store the data for later use
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

        # OUT-OF-SAMPLE PERFORMANCE
        print("\nOUT-OF-SAMPLE PERFORMANCE:")
        print(separator)
        print(header)
        print(separator)

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

            # Print row
            print(row)

            # Store the data
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

        # FULL PERIOD PERFORMANCE
        print("\nFULL PERIOD PERFORMANCE:")
        print(separator)
        print(header)
        print(separator)

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

            # Print row
            print(row)

            # Store the data
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

        return data

    def bimonthly_out_of_sample_comparison(data, best_short_sma, best_long_sma, top_medoids, min_sharpe=0.2, 
                                        contract_size=contract_multiplier, dynamic_slippage=dynamic_slippage):
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
        
        # Download historical data if not already provided
        if data is None:
            print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
            #FIXME
            print(f"Loading {TICKER} data from local files...")
            data_file = find_futures_file(SYMBOL, DATA_DIR)  # Assuming `find_futures_file` is defined similarly
            if not data_file:
                print(f"Error: No data file found for {TICKER} in {DATA_DIR}")
                exit(1)
            
            print(f"Found data file: {os.path.basename(data_file)}")
            print(f"File size: {os.path.getsize(data_file)} bytes")

            # Load the data from local file (use the same method you used in data_gather.py)
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
        
        # Customize the plot - using rounded win percentage instead of raw
        plt.title(f'k-means/medoid Bimonthly Sharpe Ratio Comparison (Out-of-Sample Period)\n' + 
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
        
        # Add a text box with rounded win percentage instead of raw
        plt.annotate(f'Medoid Portfolio Win Rate: {rounded_win_percentage:.2f}%\n'
                    f'({rounded_wins} out of {total_periods} periods)\n'
                    f'Portfolio: {len(filtered_medoids)} medoids with Sharpe ≥ {min_sharpe}',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=12)
        
        plt.tight_layout()
        save_plot('Bimonthly_Charts.png')
        
        # Return the bimonthly Sharpe ratio data
        return bimonthly_sharpe_df

    if __name__ == "__main__":
        # Set matplotlib backend explicitly
        matplotlib.use('TkAgg')  # Or try 'Qt5Agg' if this doesn't work
        
        print("Starting SMA strategy analysis...")
        
        # Run the basic analysis first
        data, best_short, best_long, best_sharpe, best_trades = analyze_sma_results()
        
        if data is None:
            print("Error: Failed to load or analyze SMA results data.")
            exit(1)
        
        print("\nProceeding with cluster analysis...")
        
        # Run the cluster analysis to get medoids
        X_filtered, medoids, top_medoids, centroids, max_sharpe_point = cluster_analysis()
        
        if X_filtered is None or medoids is None:
            print("Error: Cluster analysis failed.")
            exit(1)
        
        print("\nPlotting strategy performance...")
        
        # Plot strategy performance with the best parameters AND top medoids using fixed contract size
        market_data = plot_strategy_performance(best_short, best_long, top_medoids, contract_size=contract_multiplier)
        
        # Run the bimonthly out-of-sample comparison between best Sharpe and top medoids
        if top_medoids and len(top_medoids) > 0:
            print("\nPerforming bimonthly out-of-sample comparison...")
            bimonthly_sharpe_df = bimonthly_out_of_sample_comparison(
                market_data, 
                best_short, 
                best_long, 
                top_medoids,  # Pass the entire top_medoids list
                contract_size=contract_multiplier
            )
        else:
            print("No top medoids found. Cannot run bimonthly comparison.")
        
        print("\nAnalysis complete! All plots should have been displayed.")

if __name__ == "__main__":
    main()
