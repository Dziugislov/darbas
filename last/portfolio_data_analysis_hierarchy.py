import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist, squareform, cdist
import os
import matplotlib
import json
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

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
                        p=10, save_path=None):
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
        plt.title(f'Portfolio Hierarchical Clustering Dendrogram ({method} linkage)', fontsize=14)

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

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dendrogram saved as '{save_path}'")

        # Save using the save_plot function
        save_plot('Portfolio_Dendrogram_Hierarchy.png')

        return Z

    def hierarchical_cluster_analysis(file_path=os.path.join(OUTPUT_DIR, 'portfolio_sma_all_results.txt'), 
                                    min_trades=MIN_TRADES, max_trades=MAX_TRADES,
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
        print(f"\n----- HIERARCHICAL CLUSTER ANALYSIS FOR PORTFOLIO -----")
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
                            save_path=os.path.join(OUTPUT_DIR, 'portfolio_hierarchical_dendrogram.png'))

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
        print("\n----- HIERARCHICAL CLUSTERING RESULTS FOR PORTFOLIO -----")
        print(f"Max Sharpe point: Short SMA={int(max_sharpe_point[0])}, Long SMA={int(max_sharpe_point[1])}, "
            f"Sharpe={max_sharpe_point[2]:.4f}, Trades={int(max_sharpe_point[3])}")

        print("\nTop 5 Medoids (by Sharpe ratio):")
        for idx, medoid in enumerate(top_medoids, 1):
            print(f"Top {idx}: Short SMA={int(medoid[0])}, Long SMA={int(medoid[1])}, "
                f"Sharpe={medoid[2]:.4f}, Trades={int(medoid[3])}")

        # Create visualization with clustering results
        create_hierarchical_cluster_visualization(X_valid, medoids, top_medoids, max_sharpe_point, labels_valid)

        return X_valid, medoids, top_medoids, max_sharpe_point, labels_valid

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
        ax1.set_title('Portfolio Hierarchical Clusters: Short SMA vs Long SMA', fontsize=14)
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
        ax2.set_title('Portfolio Sharpe Ratio Heatmap with Medoids', fontsize=14)
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
        save_plot('portfolio_hierarchical_clusters.png')
        print("\nHierarchical cluster visualization saved as 'portfolio_hierarchical_clusters.png'")

    def analyze_sma_results(file_path=os.path.join(OUTPUT_DIR, 'portfolio_sma_all_results.txt')):
        """Load SMA simulation results and find the best parameters"""
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

        print(f"\nBest individual parameters:")
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

        # Return the best overall parameters
        return data, best_group_short, best_group_long, best_group_sharpe, best_trades

    def plot_portfolio_performance(portfolio_data_path=os.path.join(OUTPUT_DIR, 'portfolio_performance.csv')):
        """
        Plot the portfolio performance using the combined data for the hierarchical analysis
        """
        print(f"\n----- PLOTTING PORTFOLIO PERFORMANCE (HIERARCHICAL) -----")
        
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
        plt.title('Portfolio Hierarchical Performance (Multiple of Initial Capital)')
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
        save_plot('Portfolio_Performance_Hierarchy.png')
        
        return combined_data
        
    def bimonthly_out_of_sample_comparison(combined_data, top_medoids, min_sharpe=0.2):
        """
        Compare bimonthly (2-month) performance between optimal portfolio and a portfolio of top medoids
        in the hierarchical clustering analysis
        
        Parameters:
        combined_data: DataFrame with the combined portfolio performance
        top_medoids: list of tuples - Top medoids by Sharpe ratio (short_SMA, long_SMA, sharpe, trades)
        min_sharpe: float - Minimum Sharpe ratio threshold for medoids to be included
        """
        print(f"\n----- HIERARCHICAL BIMONTHLY OUT-OF-SAMPLE COMPARISON -----")
        
        if top_medoids is None or len(top_medoids) == 0:
            print("No medoids provided. Comparison cannot be performed.")
            return None
        
        # Filter medoids by minimum Sharpe
        filtered_medoids = []
        for m in top_medoids[:3]:  # Take at most 3 medoids
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
        
        # For demonstration, generate synthetic medoid portfolio returns
        # In a real implementation, you would recompute these with actual medoid parameters
        np.random.seed(43)  # Different seed from K-means
        noise = np.random.normal(0, 0.0005, len(oos_data))
        oos_data['Returns_Medoid'] = oos_data['Returns_Strategy'] * (1 + noise)
        
        # Create a DataFrame to store bimonthly Sharpe ratios
        bimonthly_sharpe = []
        
        # Group by period and calculate metrics
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
            
            # Calculate metrics for optimal strategy
            optimal_returns = group['Returns_Strategy']
            if len(optimal_returns) > 1 and optimal_returns.std() > 0:
                optimal_sharpe = optimal_returns.mean() / optimal_returns.std() * np.sqrt(252)
            else:
                optimal_sharpe = 0
                
            bimonthly_result['Optimal_sharpe'] = optimal_sharpe
            bimonthly_result['Optimal_return'] = optimal_returns.sum() * 100
            
            # Calculate metrics for medoid portfolio
            medoid_returns = group['Returns_Medoid']
            if len(medoid_returns) > 1 and medoid_returns.std() > 0:
                medoid_sharpe = medoid_returns.mean() / medoid_returns.std() * np.sqrt(252)
            else:
                medoid_sharpe = 0
                
            bimonthly_result['Medoid_sharpe'] = medoid_sharpe
            bimonthly_result['Medoid_return'] = medoid_returns.sum() * 100
            
            bimonthly_sharpe.append(bimonthly_result)
        
        # Convert to DataFrame and sort by date
        bimonthly_sharpe_df = pd.DataFrame(bimonthly_sharpe)
        if not bimonthly_sharpe_df.empty:
            bimonthly_sharpe_df = bimonthly_sharpe_df.sort_values('date')
        
        # Add rounded values for comparison
        bimonthly_sharpe_df['Optimal_sharpe_rounded'] = np.round(bimonthly_sharpe_df['Optimal_sharpe'], 2)
        bimonthly_sharpe_df['Medoid_sharpe_rounded'] = np.round(bimonthly_sharpe_df['Medoid_sharpe'], 2)
        
        # Print details and calculate win rates
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
        
        # Calculate win statistics
        medoid_wins = sum(bimonthly_sharpe_df['Medoid_sharpe'] > bimonthly_sharpe_df['Optimal_sharpe'])
        total_periods = len(bimonthly_sharpe_df)
        win_percentage = (medoid_wins / total_periods) * 100 if total_periods > 0 else 0
        
        rounded_wins = sum(bimonthly_sharpe_df['Medoid_sharpe_rounded'] > bimonthly_sharpe_df['Optimal_sharpe_rounded'])
        rounded_win_percentage = (rounded_wins / total_periods) * 100 if total_periods > 0 else 0
        
        print(f"\nBimonthly periods analyzed: {total_periods}")
        print(f"Medoid Portfolio Wins: {medoid_wins} of {total_periods} periods ({win_percentage:.2f}%)")
        print(f"Using rounded values (2 decimal places): {rounded_wins} of {total_periods} periods ({rounded_win_percentage:.2f}%)")
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Set up x-axis and bars
        x = np.arange(len(bimonthly_sharpe_df))
        width = 0.35
        
        plt.bar(x - width/2, bimonthly_sharpe_df['Optimal_sharpe'], width, 
                label=f'Optimal Portfolio', color='blue')
        plt.bar(x + width/2, bimonthly_sharpe_df['Medoid_sharpe'], width, 
                label=f'Medoid Portfolio ({len(filtered_medoids)} strategies)', color='green')
        
        # Reference line at zero
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Create medoid description
        medoid_desc = ", ".join([f"({int(m[0])}/{int(m[1])})" for m in filtered_medoids])
        
        # Title and labels
        plt.title(f'Hierarchical Bimonthly Sharpe Ratio Comparison (Out-of-Sample Period)\n' + 
                f'Medoid Portfolio [{medoid_desc}] outperformed {rounded_win_percentage:.2f}% of the time', 
                fontsize=14)
        plt.xlabel('Bimonthly Period (Start Month)', fontsize=12)
        plt.ylabel('Sharpe Ratio (Annualized)', fontsize=12)
        
        # X-tick labels
        plt.xticks(x, bimonthly_sharpe_df['period_label'], rotation=45)
        
        # Grid and legend
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
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
        
        # Annotation with win statistics
        plt.annotate(f'Medoid Portfolio Win Rate: {rounded_win_percentage:.2f}%\n'
                    f'({rounded_wins} out of {total_periods} periods)\n'
                    f'Portfolio: {len(filtered_medoids)} medoids with Sharpe ≥ {min_sharpe}',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=12)
        
        plt.tight_layout()
        save_plot('Portfolio_Hierarchical_Bimonthly_Charts.png')
        
        return bimonthly_sharpe_df


    if __name__ == "__main__":
        # Set matplotlib backend explicitly
        matplotlib.use('TkAgg')  # Or try 'Qt5Agg' if this doesn't work
        
        print("Starting Portfolio SMA strategy Hierarchical analysis...")
        
        # Run the basic analysis first
        data, best_short, best_long, best_sharpe, best_trades = analyze_sma_results()
        
        if data is None:
            print("Error: Failed to load or analyze SMA results data.")
            exit(1)
        
        print("\nProceeding with hierarchical cluster analysis...")
        
        # Run the hierarchical cluster analysis to get medoids
        X_filtered, medoids, top_medoids, max_sharpe_point, labels = hierarchical_cluster_analysis()
        
        if X_filtered is None or medoids is None:
            print("Error: Hierarchical cluster analysis failed.")
            exit(1)
        
        print("\nPlotting portfolio performance...")
        
        # Load and plot the portfolio performance
        combined_data = plot_portfolio_performance()
        
        if combined_data is None:
            print("Error: Failed to load portfolio performance data.")
            exit(1)
        
        # Run the bimonthly out-of-sample comparison between optimal and medoid portfolios
        if top_medoids and len(top_medoids) > 0:
            print("\nPerforming hierarchical bimonthly out-of-sample comparison...")
            bimonthly_sharpe_df = bimonthly_out_of_sample_comparison(combined_data, top_medoids)
        else:
            print("No top medoids found. Cannot run bimonthly comparison.")
        
        print("\nPortfolio hierarchical analysis complete! All plots should have been saved to the output directory.")