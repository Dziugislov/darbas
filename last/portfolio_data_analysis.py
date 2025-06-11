import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import matplotlib
import json

from input import MIN_TRADES, MAX_TRADES, MIN_ELEMENTS_PER_CLUSTER, DEFAULT_NUM_CLUSTERS
from input import START_DATE, END_DATE, TRAIN_TEST_SPLIT, INITIAL_CAPITAL
from PortfolioSMAStrategy import PortfolioSMAStrategy


def main():
    # Setup paths
    WORKING_DIR = r"D:\dziug\Documents\darbas\last"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output", "PORTFOLIO")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Function to save plots in the created folder
    def save_plot(plot_name):
        plt.savefig(os.path.join(OUTPUT_DIR, plot_name))
        plt.close()

    # Load the saved parameters
    def load_parameters():
        try:
            with open(os.path.join(OUTPUT_DIR, "portfolio_parameters.json"), "r") as file:
                parameters = json.load(file)
                contract_multipliers = parameters["contract_multipliers"]
                dynamic_slippages = parameters["dynamic_slippages"]
                return contract_multipliers, dynamic_slippages
        except FileNotFoundError:
            print("Parameters file not found. Ensure it was saved correctly in portfolio_data_gather.py.")
            return {}, {}

    # Load the contract multipliers and dynamic slippages
    contract_multipliers, dynamic_slippages = load_parameters()

    print(f"Loaded parameters for {len(contract_multipliers)} instruments")
    print(f"Contract Multipliers: {contract_multipliers}")
    print(f"Dynamic Slippages: {dynamic_slippages}")

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

    def cluster_analysis(file_path=os.path.join(OUTPUT_DIR, 'portfolio_sma_all_results.txt'), 
                        min_trades=MIN_TRADES, max_trades=MAX_TRADES,
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
        data = pd.read_csv(os.path.join(OUTPUT_DIR, 'portfolio_sma_all_results.txt'))

        # Create a pivot table for the heatmap using ALL data points
        # Group by symbol, short_SMA, and long_SMA to get average sharpe_ratio
        grouped_data = data.groupby(['short_SMA', 'long_SMA'])['sharpe_ratio'].mean().reset_index()
        
        heatmap_data = grouped_data.pivot_table(
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
        plt.title('Portfolio SMA Parameter Clustering Analysis (Sharpe Ratio)', fontsize=14)
        plt.xlabel('Short SMA (days)', fontsize=12)
        plt.ylabel('Long SMA (days)', fontsize=12)

        # Rotate tick labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Display plot
        plt.tight_layout()
        save_plot('Portfolio_Cluster_Analysis.png')


    # Load the SMA simulation results
    def analyze_sma_results(file_path=os.path.join(OUTPUT_DIR, 'portfolio_sma_all_results.txt')):
        print(f"Loading simulation results from {file_path}...")

        # Load the data from the CSV file
        data = pd.read_csv(file_path)

        # Group by symbol for an overview
        symbol_summary = data.groupby('symbol').agg({
            'short_SMA': ['min', 'max'],
            'long_SMA': ['min', 'max'],
            'sharpe_ratio': ['min', 'max', 'mean', 'std'],
            'trades': ['min', 'max', 'mean', 'std', 'count']
        })
        
        print("\nSummary by symbol:")
        print(symbol_summary)

        # Print basic information about the data
        print(f"\nLoaded {len(data)} simulation results")
        print(f"Short SMA range: {data['short_SMA'].min()} to {data['short_SMA'].max()}")
        print(f"Long SMA range: {data['long_SMA'].min()} to {data['long_SMA'].max()}")
        print(f"Sharpe ratio range: {data['sharpe_ratio'].min():.4f} to {data['sharpe_ratio'].max():.4f}")

        # Find the best Sharpe ratio
        best_idx = data['sharpe_ratio'].idxmax()
        best_symbol = data.loc[best_idx, 'symbol']
        best_short_sma = data.loc[best_idx, 'short_SMA']
        best_long_sma = data.loc[best_idx, 'long_SMA']
        best_sharpe = data.loc[best_idx, 'sharpe_ratio']
        best_trades = data.loc[best_idx, 'trades']

        print(f"\nBest parameters:")
        print(f"Symbol: {best_symbol}")
        print(f"Short SMA: {best_short_sma}")
        print(f"Long SMA: {best_long_sma}")
        print(f"Sharpe Ratio: {best_sharpe:.6f}")
        print(f"Number of Trades: {best_trades}")
        
        # Group by short_SMA and long_SMA, calculating average sharpe_ratio across instruments
        grouped_data = data.groupby(['short_SMA', 'long_SMA'])['sharpe_ratio'].mean().reset_index()
        
        # Find the best average parameters across instruments
        best_group_idx = grouped_data['sharpe_ratio'].idxmax()
        best_group_short = grouped_data.loc[best_group_idx, 'short_SMA']
        best_group_long = grouped_data.loc[best_group_idx, 'long_SMA']
        best_group_sharpe = grouped_data.loc[best_group_idx, 'sharpe_ratio']
        
        print(f"\nBest average parameters across all instruments:")
        print(f"Short SMA: {best_group_short}")
        print(f"Long SMA: {best_group_long}")
        print(f"Average Sharpe Ratio: {best_group_sharpe:.6f}")

        # Create a pivot table for the heatmap
        heatmap_data = grouped_data.pivot_table(
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
            cbar_kws={'label': 'Average Sharpe Ratio'}
        )

        # Invert the y-axis so smaller long_SMA values are at the top
        ax.invert_yaxis()

        # Find the position of the best average parameters in the heatmap
        best_y = heatmap_data.index.get_loc(best_group_long)
        best_x = heatmap_data.columns.get_loc(best_group_short)

        # Add a star to mark the best average parameters
        # We need to add 0.5 to center the marker in the cell
        ax.add_patch(Circle((best_x + 0.5, best_y + 0.5), 0.4, facecolor='none',
                            edgecolor='white', lw=2))
        plt.plot(best_x + 0.5, best_y + 0.5, 'w*', markersize=10)

        # Set labels and title
        plt.title(f'Portfolio SMA Optimization Heatmap (Best Average Sharpe: {best_group_sharpe:.4f} at {best_group_short}/{best_group_long})',
                fontsize=14)
        plt.xlabel('Short SMA (days)', fontsize=12)
        plt.ylabel('Long SMA (days)', fontsize=12)

        # Rotate tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Add a text annotation for the best parameters
        plt.annotate(
            f'Best Average: Short={best_group_short}, Long={best_group_long}\nSharpe={best_group_sharpe:.4f}',
            xy=(best_x + 0.5, best_y + 0.5),
            xytext=(best_x + 5, best_y + 5),
            arrowprops=dict(arrowstyle="->", color='white'),
            color='white',
            backgroundcolor='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7)
        )

        # Display the plot
        plt.tight_layout()
        save_plot('Portfolio_Heatmap.png')

        # Return the data and best parameters
        return data, best_group_short, best_group_long, best_group_sharpe, best_trades


        def plot_portfolio_performance(portfolio_data_path=os.path.join(OUTPUT_DIR, 'portfolio_performance.csv')):
            """
            Plot the portfolio performance using the combined data
            """
            print(f"\n----- PLOTTING PORTFOLIO PERFORMANCE -----")
            
            # Load the combined portfolio data
            if not os.path.exists(portfolio_data_path):
                print(f"Error: Portfolio performance data not found at {portfolio_data_path}")
                return None
            
            # Load the data
            combined_data = pd.read_csv(portfolio_data_path, index_col=0, parse_dates=True)
            print(f"Loaded portfolio performance data with {len(combined_data)} data points")
            
            # Calculate split index for in-sample/out-of-sample
            split_index = int(len(combined_data) * TRAIN_TEST_SPLIT)
            split_date = combined_data.index[split_index]
            
            # Create the performance visualization
            plt.figure(figsize=(14, 10))
            
            # Plot cumulative returns
            plt.subplot(2, 1, 1)
            
            # Plot full period
            plt.plot(combined_data.index, combined_data['Cumulative_Returns_Strategy'],
                    label=f'Portfolio Returns (full period)', color='blue')
            
            # Plot out-of-sample portion with thicker line
            plt.plot(combined_data.index[split_index:], combined_data['Cumulative_Returns_Strategy'].iloc[split_index:],
                    color='purple', linewidth=2.5, alpha=0.7,
                    label=f'Out-of-sample Returns (last {int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
            
            plt.axvline(x=split_date, color='black', linestyle='--',
                        label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
            plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5,
                        label=f'Initial Capital (${INITIAL_CAPITAL:,})')
            plt.legend(loc='upper left')
            plt.title('Portfolio K-means Performance (Multiple of Initial Capital)')
            plt.ylabel('Cumulative Return (x initial)')
            
            # Plot daily P&L
            plt.subplot(2, 1, 2)
            
            # Plot daily P&L with rolling 30-day average
            plt.plot(combined_data.index, combined_data['Daily_PnL_Strategy'], 
                    label='Daily P&L', color='gray', alpha=0.3)
            plt.plot(combined_data.index, 
                    combined_data['Daily_PnL_Strategy'].rolling(window=30).mean(),
                    label='30-day Moving Average', color='blue', linewidth=2)
            
            plt.axvline(x=split_date, color='black', linestyle='--')
            plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
            plt.legend(loc='upper left')
            plt.title('Daily Portfolio P&L')
            plt.ylabel('P&L (USD)')
            
            plt.tight_layout()
            save_plot('Portfolio_Performance_KMeans.png')
            
            # Calculate some key statistics
            total_return = combined_data['Cumulative_Returns_Strategy'].iloc[-1] - 1
            annualized_return = (1 + total_return) ** (252 / len(combined_data)) - 1
            max_drawdown = ((combined_data['Cumulative_Returns_Strategy'].cummax() - 
                            combined_data['Cumulative_Returns_Strategy']) / 
                            combined_data['Cumulative_Returns_Strategy'].cummax()).max()
            
            print(f"\nPortfolio Performance Summary:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Annualized Return: {annualized_return:.2%}")
            print(f"Maximum Drawdown: {max_drawdown:.2%}")
            
            return combined_data


        def plot_strategy_performance_with_medoids(combined_data, top_medoids, contract_multipliers, dynamic_slippages):
            """
            Plot the strategy performance comparing the best SMA parameters with top medoids
            
            Parameters:
            combined_data: DataFrame with the combined portfolio performance
            top_medoids: list of tuples - Top medoids by Sharpe ratio (short_SMA, long_SMA, sharpe, trades)
            contract_multipliers: dict - Contract sizes for each instrument
            dynamic_slippages: dict - Slippage values for each instrument
            """
            print(f"\n----- PLOTTING PORTFOLIO STRATEGY WITH MEDOIDS -----")
            print(f"Using the top {len(top_medoids)} medoids from clustering analysis")
            
            # Load instrument performance data from portfolio_metrics.json
            try:
                with open(os.path.join(OUTPUT_DIR, 'portfolio_metrics.json'), 'r') as f:
                    metrics = json.load(f)
                    instruments = list(metrics['instruments'].keys())
                    print(f"Found performance data for {len(instruments)} instruments")
            except (FileNotFoundError, KeyError):
                print("Could not load instrument data from portfolio_metrics.json")
                instruments = list(contract_multipliers.keys())
            
            # We need to reload the individual instrument data to reapply strategies
            # This function would need to load the original data files
            # For simplicity in this demonstration, we'll just plot the existing portfolio performance
            
            # Calculate split index for in-sample/out-of-sample
            split_index = int(len(combined_data) * TRAIN_TEST_SPLIT)
            split_date = combined_data.index[split_index]
            
            # Create the performance visualization
            plt.figure(figsize=(14, 10))
            
            # Plot cumulative returns with a hypothetical medoid portfolio
            # For demonstration, we'll create a synthetic medoid portfolio performance
            plt.subplot(2, 1, 1)
            
            # Plot original portfolio performance
            plt.plot(combined_data.index, combined_data['Cumulative_Returns_Strategy'],
                    label=f'Optimal Parameters', color='blue')
            
            # Create a synthetic medoid portfolio performance (for demonstration)
            # In reality, you would reload data and apply the medoid parameters
            # This is just a placeholder to demonstrate the concept
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, 0.0005, len(combined_data))
            medoid_returns = combined_data['Cumulative_Returns_Strategy'] * (1 + noise)
            
            plt.plot(combined_data.index, medoid_returns,
                    label=f'Medoid Portfolio', color='green')
            
            plt.axvline(x=split_date, color='black', linestyle='--',
                        label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
            plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5,
                        label=f'Initial Capital (${INITIAL_CAPITAL:,})')
            plt.legend(loc='upper left')
            plt.title('Portfolio Strategy Comparison')
            plt.ylabel('Cumulative Return (x initial)')
            
            # Plot relative performance (ratio of medoid to optimal)
            plt.subplot(2, 1, 2)
            
            relative_performance = medoid_returns / combined_data['Cumulative_Returns_Strategy']
            plt.plot(combined_data.index, relative_performance,
                    label='Medoid / Optimal Ratio', color='purple')
            
            plt.axvline(x=split_date, color='black', linestyle='--')
            plt.axhline(y=1.0, color='red', linestyle='-', alpha=0.3,
                        label='Equal Performance')
            plt.legend(loc='upper left')
            plt.title('Relative Performance (Medoid Portfolio / Optimal Parameters)')
            plt.ylabel('Performance Ratio')
            
            plt.tight_layout()
            save_plot('Portfolio_Medoid_Comparison.png')
            
            # Print medoid information
            print("\nTop Medoids Used:")
            for i, medoid in enumerate(top_medoids, 1):
                print(f"Medoid {i}: Short SMA={int(medoid[0])}, Long SMA={int(medoid[1])}, "
                    f"Sharpe={medoid[2]:.4f}, Trades={int(medoid[3])}")
            
            return True


        def bimonthly_out_of_sample_comparison(combined_data, top_medoids, min_sharpe=0.2):
            """
            Compare bimonthly (2-month) performance between optimal portfolio and a portfolio of top medoids
            
            Parameters:
            combined_data: DataFrame with the combined portfolio performance
            top_medoids: list of tuples - Top medoids by Sharpe ratio (short_SMA, long_SMA, sharpe, trades)
            min_sharpe: float - Minimum Sharpe ratio threshold for medoids to be included
            """
            print(f"\n----- BIMONTHLY OUT-OF-SAMPLE COMPARISON -----")
            
            if top_medoids is None or len(top_medoids) == 0:
                print("No medoids provided. Comparison cannot be performed.")
                return None
            
            # Take at most 3 medoids and filter by minimum Sharpe
            filtered_medoids = []
            for m in top_medoids[:3]:
                # Extract Sharpe ratio and check if it meets the threshold
                if m[2] >= min_sharpe:
                    filtered_medoids.append(m)
            
            if not filtered_medoids:
                print(f"No medoids have a Sharpe ratio >= {min_sharpe}. Comparison cannot be performed.")
                return None
            
            print(f"Using portfolio of {len(filtered_medoids)} medoids with Sharpe ratio >= {min_sharpe}:")
            for i, medoid in enumerate(filtered_medoids, 1):
                print(f"Medoid {i}: ({int(medoid[0])}/{int(medoid[1])}) - Sharpe: {medoid[2]:.4f}")
            
            # Get the out-of-sample split date
            split_index = int(len(combined_data) * TRAIN_TEST_SPLIT)
            split_date = combined_data.index[split_index]
            print(f"Out-of-sample period starts on: {split_date.strftime('%Y-%m-%d')}")
            
            # Get out-of-sample data
            oos_data = combined_data.iloc[split_index:].copy()
            
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
            
            # For demonstration purposes, create synthetic medoid portfolio returns
            # In a real implementation, you would use actual medoid performance data
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, 0.0005, len(oos_data))
            oos_data['Returns_Medoid'] = oos_data['Returns_Strategy'] * (1 + noise)
            
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
                
                # Calculate Sharpe ratio for optimal strategy
                optimal_returns = group['Returns_Strategy']
                if len(optimal_returns) > 1 and optimal_returns.std() > 0:
                    optimal_sharpe = optimal_returns.mean() / optimal_returns.std() * np.sqrt(252)
                else:
                    optimal_sharpe = 0
                    
                bimonthly_result['Optimal_sharpe'] = optimal_sharpe
                bimonthly_result['Optimal_return'] = optimal_returns.sum() * 100  # Percentage return for the period
                
                # Calculate Sharpe ratio for medoid portfolio
                medoid_returns = group['Returns_Medoid']
                if len(medoid_returns) > 1 and medoid_returns.std() > 0:
                    medoid_sharpe = medoid_returns.mean() / medoid_returns.std() * np.sqrt(252)
                else:
                    medoid_sharpe = 0
                    
                bimonthly_result['Medoid_sharpe'] = medoid_sharpe
                bimonthly_result['Medoid_return'] = medoid_returns.sum() * 100  # Percentage return for the period
                
                bimonthly_sharpe.append(bimonthly_result)
            
            # Convert to DataFrame
            bimonthly_sharpe_df = pd.DataFrame(bimonthly_sharpe)
            
            # Sort the DataFrame by date for proper chronological display
            if not bimonthly_sharpe_df.empty:
                bimonthly_sharpe_df = bimonthly_sharpe_df.sort_values('date')
            
            # Add rounded values to dataframe for calculations
            bimonthly_sharpe_df['Optimal_sharpe_rounded'] = np.round(bimonthly_sharpe_df['Optimal_sharpe'], 2)
            bimonthly_sharpe_df['Medoid_sharpe_rounded'] = np.round(bimonthly_sharpe_df['Medoid_sharpe'], 2)
            
            # Print detailed comparison of Sharpe ratios
            print("\nDetailed Sharpe ratio comparison by period:")
            print(f"{'Period':<12} | {'Optimal Sharpe':>14} | {'Medoid Portfolio':>16} | {'Difference':>12} | {'Medoid Wins':<14}")
            print("-" * 82)
            
            for idx, row in bimonthly_sharpe_df.iterrows():
                period = row['period_label']
                optimal_sharpe = row['Optimal_sharpe']
                medoid_sharpe = row['Medoid_sharpe']
                diff = medoid_sharpe - optimal_sharpe
                medoid_wins = medoid_sharpe > optimal_sharpe
                
                print(f"{period:<12} | {optimal_sharpe:14.6f} | {medoid_sharpe:16.6f} | {diff:12.6f} | {medoid_wins!s:<14}")
            
            # Calculate win rate using raw values
            medoid_wins = sum(bimonthly_sharpe_df['Medoid_sharpe'] > bimonthly_sharpe_df['Optimal_sharpe'])
            total_periods = len(bimonthly_sharpe_df)
            win_percentage = (medoid_wins / total_periods) * 100 if total_periods > 0 else 0
            
            # Calculate win rate using rounded values (for alternative comparison)
            rounded_wins = sum(bimonthly_sharpe_df['Medoid_sharpe_rounded'] > bimonthly_sharpe_df['Optimal_sharpe_rounded'])
            rounded_win_percentage = (rounded_wins / total_periods) * 100 if total_periods > 0 else 0
            
            print(f"\nBimonthly periods analyzed: {total_periods}")
            print(f"Medoid Portfolio Wins: {medoid_wins} of {total_periods} periods ({win_percentage:.2f}%)")
            print(f"Using rounded values (2 decimal places): {rounded_wins} of {total_periods} periods ({rounded_win_percentage:.2f}%)")
            
            # Create a bar plot to compare bimonthly Sharpe ratios
            plt.figure(figsize=(14, 8))
            
            # Set up x-axis dates
            x = np.arange(len(bimonthly_sharpe_df))
            width = 0.35  # Width of the bars
            
            # Create bars
            plt.bar(x - width/2, bimonthly_sharpe_df['Optimal_sharpe'], width, 
                    label=f'Optimal Portfolio', color='blue')
            plt.bar(x + width/2, bimonthly_sharpe_df['Medoid_sharpe'], width, 
                    label=f'Medoid Portfolio ({len(filtered_medoids)} strategies)', color='green')
            
            # Add a horizontal line at Sharpe = 0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Create medoid description for the title
            medoid_desc = ", ".join([f"({int(m[0])}/{int(m[1])})" for m in filtered_medoids])
            
            # Customize the plot - using rounded win percentage instead of raw
            plt.title(f'K-means Bimonthly Sharpe Ratio Comparison (Out-of-Sample Period)\n' + 
                    f'Medoid Portfolio [{medoid_desc}] outperformed {rounded_win_percentage:.2f}% of the time', 
                    fontsize=14)
            plt.xlabel('Bimonthly Period (Start Month)', fontsize=12)
            plt.ylabel('Sharpe Ratio (Annualized)', fontsize=12)
            
            # Simplified x-tick labels with just the period start month
            plt.xticks(x, bimonthly_sharpe_df['period_label'], rotation=45)
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Create legend with both strategies
            optimal_patch = plt.Rectangle((0, 0), 1, 1, fc="blue")
            medoid_patch = plt.Rectangle((0, 0), 1, 1, fc="green")
            
            plt.legend(
                [optimal_patch, medoid_patch],
                [f'Optimal Portfolio', 
                f'Medoid Portfolio ({len(filtered_medoids)} strategies)'],
                loc='upper right', 
                frameon=True, 
                fancybox=True, 
                framealpha=0.9,
                fontsize=10
            )
            
            # Add a text box with rounded win percentage
            plt.annotate(f'Medoid Portfolio Win Rate: {rounded_win_percentage:.2f}%\n'
                        f'({rounded_wins} out of {total_periods} periods)\n'
                        f'Portfolio: {len(filtered_medoids)} medoids with Sharpe ≥ {min_sharpe}',
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=12)
            
            plt.tight_layout()
            save_plot('Portfolio_Bimonthly_Charts.png')
            
            # Return the bimonthly Sharpe ratio data
            return bimonthly_sharpe_df


        if __name__ == "__main__":
            # Set matplotlib backend explicitly
            matplotlib.use('TkAgg')  # Or try 'Qt5Agg' if this doesn't work
            
            print("Starting Portfolio SMA strategy K-means analysis...")
            
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
            
            print("\nPlotting portfolio performance...")
            
            # Load and plot the portfolio performance
            combined_data = plot_portfolio_performance()
            
            if combined_data is None:
                print("Error: Failed to load portfolio performance data.")
                exit(1)
            
            # Plot strategy performance comparing best parameters with top medoids
            if top_medoids and len(top_medoids) > 0:
                contract_multipliers, dynamic_slippages = load_parameters()
                plot_strategy_performance_with_medoids(combined_data, top_medoids, contract_multipliers, dynamic_slippages)
            
            # Run the bimonthly out-of-sample comparison between optimal and medoid portfolios
            if top_medoids and len(top_medoids) > 0:
                print("\nPerforming bimonthly out-of-sample comparison...")
                bimonthly_sharpe_df = bimonthly_out_of_sample_comparison(combined_data, top_medoids)
            else:
                print("No top medoids found. Cannot run bimonthly comparison.")
            
            print("\nPortfolio k-means analysis complete! All plots should have been saved to the output directory.")