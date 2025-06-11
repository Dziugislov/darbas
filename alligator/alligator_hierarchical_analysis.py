"""
Classic Alligator Strategy Hierarchical Analysis

This script performs hierarchical clustering analysis on Classic Alligator strategy optimization results
to identify robust parameter regions and compare their out-of-sample performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist, squareform
import os
import calendar

# Import from input file - make sure the path is correct
from input import TICKER, START_DATE, END_DATE, TRAIN_TEST_SPLIT, INITIAL_CAPITAL, CONTRACT_SIZE, SLIPPAGE
from input import JAW_LIPS_RESULTS_FILE, OUTPUT_DIR, INITIAL_TEETH

# Constants for clustering
MIN_TRADES = 5  # Minimum number of trades to consider in clustering
MAX_TRADES = 100  # Maximum number of trades to consider
MIN_ELEMENTS_PER_CLUSTER = 3  # Minimum number of elements per valid cluster
DEFAULT_NUM_CLUSTERS = 5  # Default number of clusters if not specified


def analyze_alligator_results(file_path=JAW_LIPS_RESULTS_FILE):
    """
    Analyze the Classic Alligator optimization results and create a basic heatmap visualization
    """
    print(f"Loading Classic Alligator optimization results from {file_path}...")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: Results file '{file_path}' not found!")
        print("Please run the optimization first or check file paths.")
        return None, None, None, None, None

    try:
        # Load the data from the CSV file
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return None, None, None, None, None

    # Print basic information about the data
    print(f"Loaded {len(data)} simulation results")
    print(f"Jaw period range: {data['jaw_period'].min()} to {data['jaw_period'].max()}")
    print(f"Lips period range: {data['lips_period'].min()} to {data['lips_period'].max()}")
    print(f"Sharpe ratio range: {data['sharpe_ratio'].min():.4f} to {data['sharpe_ratio'].max():.4f}")
    print(f"Trades range: {data['trades'].min()} to {data['trades'].max()}")

    # Find the best Sharpe ratio
    best_idx = data['sharpe_ratio'].idxmax()
    best_jaw = data.loc[best_idx, 'jaw_period']
    best_teeth = data.loc[best_idx, 'teeth_period']
    best_lips = data.loc[best_idx, 'lips_period']
    best_sharpe = data.loc[best_idx, 'sharpe_ratio']
    best_trades = data.loc[best_idx, 'trades']

    print(f"\nBest parameters:")
    print(f"Jaw period: {best_jaw} (shift=8)")
    print(f"Teeth period: {best_teeth} (shift=5)")
    print(f"Lips period: {best_lips} (shift=3)")
    print(f"Sharpe Ratio: {best_sharpe:.6f}")
    print(f"Number of Trades: {best_trades}")

    # Create a pivot table for the heatmap
    try:
        heatmap_data = data.pivot_table(
            index='jaw_period',
            columns='lips_period',
            values='sharpe_ratio'
        )
    except Exception as e:
        print(f"Error creating pivot table: {e}")
        return None, None, None, None, None

    # Create the heatmap visualization
    plt.figure(figsize=(12, 10))

    # Create a mask for invalid combinations (where lips_period >= jaw_period)
    mask = np.zeros_like(heatmap_data, dtype=bool)
    for i, jaw in enumerate(heatmap_data.index):
        for j, lips in enumerate(heatmap_data.columns):
            if lips >= jaw:
                mask[i, j] = True

    # Plot the heatmap with the mask
    ax = sns.heatmap(
        heatmap_data,
        mask=mask,
        cmap='coolwarm',  # Blue to red colormap
        annot=False,      # Don't annotate each cell with its value
        fmt='.4f',
        linewidths=0,
        cbar_kws={'label': 'Sharpe Ratio'}
    )

    # Invert the y-axis so smaller jaw values are at the top
    ax.invert_yaxis()

    # Find the position of the best Sharpe ratio in the heatmap
    best_y = heatmap_data.index.get_loc(best_jaw)
    best_x = heatmap_data.columns.get_loc(best_lips)

    # Add a star to mark the best Sharpe ratio
    # We need to add 0.5 to center the marker in the cell
    ax.add_patch(Circle((best_x + 0.5, best_y + 0.5), 0.4, facecolor='none',
                         edgecolor='white', lw=2))
    plt.plot(best_x + 0.5, best_y + 0.5, 'w*', markersize=10)

    # Set labels and title
    plt.title(f'Classic Alligator Optimization Heatmap (Best Sharpe: {best_sharpe:.4f} at Jaw={best_jaw}/Lips={best_lips})',
              fontsize=14)
    plt.xlabel('Lips Period (days)', fontsize=12)
    plt.ylabel('Jaw Period (days)', fontsize=12)

    # Rotate tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add a text annotation for the best parameters
    plt.annotate(
        f'Best: Jaw={best_jaw}, Lips={best_lips}\nSharpe={best_sharpe:.4f}, Trades={best_trades}',
        xy=(best_x + 0.5, best_y + 0.5),
        xytext=(best_x + 5, best_y + 5),
        arrowprops=dict(arrowstyle="->", color='white'),
        color='white',
        backgroundcolor='black',
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7)
    )

    # Display and save the plot
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'classic_alligator_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved as '{output_file}'")
    plt.show()

    # Return the data and best parameters
    return data, best_jaw, best_lips, best_sharpe, best_trades


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

    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dendrogram saved as '{save_path}'")

    # Show the plot
    plt.show()

    return Z

def compute_hierarchical_medoids(X, labels, valid_clusters):
    """
    Compute medoids for each valid cluster from hierarchical clustering
    A medoid is the data point in a cluster that has the minimum average distance to all other points in the cluster

    Parameters:
    X: numpy array of shape (n_samples, n_features) - Original data points (not scaled)
    labels: numpy array of shape (n_samples,) - Cluster labels for each data point
    valid_clusters: set - Set of valid cluster IDs

    Returns:
    list of tuples - Each tuple contains (lips_period, jaw_period, sharpe_ratio, trades) for each medoid
    """
    from scipy.spatial.distance import cdist

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
        print(f"Cluster {cluster_id} medoid: Lips Period={int(medoid[0])}, Jaw Period={int(medoid[1])}, "
              f"Sharpe={medoid[2]:.4f}, Trades={int(medoid[3])}")

    return medoids


def create_hierarchical_cluster_visualization(X, medoids, top_medoids, max_sharpe_point, labels):
    """
    Create visualization of hierarchical clustering results with scatter plots for Alligator parameters

    Parameters:
    X: numpy array - Original data points (not scaled)
    medoids: list of tuples - Cluster medoids (lips_period, jaw_period, sharpe_ratio, trades)
    top_medoids: list of tuples - Top medoids by Sharpe ratio 
    max_sharpe_point: tuple - Point with maximum Sharpe ratio (lips_period, jaw_period, sharpe_ratio, trades)
    labels: numpy array - Cluster labels

    Returns:
    None
    """
    from matplotlib.lines import Line2D

    # Create a new figure with 2 subplots (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Get the number of unique clusters
    n_clusters = len(np.unique(labels))

    # Create a colormap with distinct colors for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # Plot 1: Lips Period vs Jaw Period (colored by cluster)
    for i, color in enumerate(colors):
        # Get indices of points in cluster i
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            continue

        # Plot the points without labels to avoid cluttering the legend
        ax1.scatter(
            cluster_points[:, 0],  # Lips Period
            cluster_points[:, 1],  # Jaw Period
            s=50, c=[color], alpha=0.7
            # No label parameter here to keep them out of the legend
        )

    # Plot the medoids with black edges
    for i, medoid in enumerate(medoids):
        ax1.scatter(
            medoid[0],  # Lips Period
            medoid[1],  # Jaw Period
            s=150, c='white', alpha=1,
            edgecolors='black', linewidths=2,
            marker='o'
        )

    # Plot the top medoids with star markers
    for i, medoid in enumerate(top_medoids):
        ax1.scatter(
            medoid[0],  # Lips Period
            medoid[1],  # Jaw Period
            s=200, c='gold', alpha=1,
            edgecolors='black', linewidths=1.5,
            marker='*'
        )

    # Plot the max Sharpe point with a diamond marker
    ax1.scatter(
        max_sharpe_point[0],  # Lips Period
        max_sharpe_point[1],  # Jaw Period
        s=250, c='red', alpha=1,
        edgecolors='black', linewidths=2,
        marker='D'
    )

    # Add labels and title
    ax1.set_xlabel('Lips Period (days)', fontsize=12)
    ax1.set_ylabel('Jaw Period (days)', fontsize=12)
    ax1.set_title('Hierarchical Clusters: Lips Period vs Jaw Period', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add a diagonal line where lips_period = jaw_period
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

    # Plot 2: Lips Period vs Jaw Period (colored by Sharpe ratio)
    scatter = ax2.scatter(
        X[:, 0],  # Lips Period
        X[:, 1],  # Jaw Period
        s=50,
        c=X[:, 2],  # Sharpe ratio for color
        cmap='coolwarm',
        alpha=0.7
    )

    # Plot the top medoids with star markers
    for i, medoid in enumerate(top_medoids):
        ax2.scatter(
            medoid[0],  # Lips Period
            medoid[1],  # Jaw Period
            s=200, c='gold', alpha=1,
            edgecolors='black', linewidths=1.5,
            marker='*'
        )

    # Plot the max Sharpe point with a diamond marker
    ax2.scatter(
        max_sharpe_point[0],  # Lips Period
        max_sharpe_point[1],  # Jaw Period
        s=250, c='red', alpha=1,
        edgecolors='black', linewidths=2,
        marker='D'
    )

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Sharpe Ratio', fontsize=12)

    # Add labels and title
    ax2.set_xlabel('Lips Period (days)', fontsize=12)
    ax2.set_ylabel('Jaw Period (days)', fontsize=12)
    ax2.set_title('Sharpe Ratio Heatmap with Medoids', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add a diagonal line where lips_period = jaw_period
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
    output_file = os.path.join(OUTPUT_DIR, 'classic_alligator_hierarchical_clusters.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Hierarchical cluster visualization saved as '{output_file}'")

    # Show the plot
    plt.show()


def hierarchical_cluster_analysis(file_path=JAW_LIPS_RESULTS_FILE, min_trades=MIN_TRADES, max_trades=MAX_TRADES,
                                  min_elements_per_cluster=MIN_ELEMENTS_PER_CLUSTER):
    """
    Perform hierarchical clustering analysis on Classic Alligator optimization results to find robust parameter regions

    Parameters:
    file_path: str - Path to the file with Alligator results
    min_trades: int - Minimum number of trades to consider
    max_trades: int - Maximum number of trades to consider
    min_elements_per_cluster: int - Minimum number of elements per cluster

    Returns:
    tuple - (X_filtered, medoids, top_medoids, max_sharpe_point, labels)
    """
    print(f"\n----- HIERARCHICAL CLUSTER ANALYSIS FOR CLASSIC ALLIGATOR STRATEGY -----")
    print(f"Loading data from {file_path}...")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Results file '{file_path}' not found!")
        return None, None, None, None, None

    try:
        # Load the data
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None, None, None, None

    # Print data structure
    print(f"Data contains {len(df)} rows with columns: {df.columns.tolist()}")

    # Convert data to numpy array for easier processing
    X = df[['lips_period', 'jaw_period', 'sharpe_ratio', 'trades']].values

    # Filter data by number of trades and ensure lips_period < jaw_period
    X_filtered = X[(X[:, 0] < X[:, 1]) &  # lips_period < jaw_period
                   (X[:, 3] >= min_trades) &  # trades >= min_trades
                   (X[:, 3] <= max_trades)]  # trades <= max_trades

    if len(X_filtered) == 0:
        print(f"No data points meet the criteria after filtering! Adjust min_trades ({min_trades}) and max_trades ({max_trades}).")
        return None, None, None, None, None

    print(f"Filtered data to {len(X_filtered)} points with {min_trades}-{max_trades} trades")

    # Extract the fields for better scaling visibility
    lips_period_values = X_filtered[:, 0]
    jaw_period_values = X_filtered[:, 1]
    sharpe_values = X_filtered[:, 2]
    trades_values = X_filtered[:, 3]

    print(f"Lips Period range: {lips_period_values.min()} to {lips_period_values.max()}")
    print(f"Jaw Period range: {jaw_period_values.min()} to {jaw_period_values.max()}")
    print(f"Sharpe ratio range: {sharpe_values.min():.4f} to {sharpe_values.max():.4f}")
    print(f"Trades range: {trades_values.min()} to {trades_values.max()}")

    # Only use the first 3 dimensions for clustering (lips_period, jaw_period, sharpe_ratio)
    X_filtered_3d = X_filtered[:, 0:3]

    # Scale the data for clustering - using StandardScaler for each dimension
    # This addresses the issue where parameter values have much larger ranges than Sharpe ratio
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_3d)

    # Print scaling info for verification
    print("\nScaled data information:")
    scaled_lips = X_scaled[:, 0]
    scaled_jaw = X_scaled[:, 1]
    scaled_sharpe = X_scaled[:, 2]

    print(f"Scaled Lips Period range: {scaled_lips.min():.4f} to {scaled_lips.max():.4f}")
    print(f"Scaled Jaw Period range: {scaled_jaw.min():.4f} to {scaled_jaw.max():.4f}")
    print(f"Scaled Sharpe ratio range: {scaled_sharpe.min():.4f} to {scaled_sharpe.max():.4f}")

    # Create a dendrogram to help choose the number of clusters
    print("\nCreating dendrogram to visualize hierarchical structure...")
    linkage_method = 'ward'  # Ward minimizes the variance within clusters
    dendrogram_path = os.path.join(OUTPUT_DIR, 'classic_alligator_hierarchical_dendrogram.png')
    Z = create_dendrogram(X_scaled, method=linkage_method,
                          figsize=(12, 8),
                          save_path=dendrogram_path)

    # Determine optimal number of clusters or use default
    try:
        k = int(input(f"Enter number of clusters based on dendrogram (default={DEFAULT_NUM_CLUSTERS}): ")
                or DEFAULT_NUM_CLUSTERS)
    except ValueError:
        print(f"Invalid input, using default value of {DEFAULT_NUM_CLUSTERS}")
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
    max_sharpe_point = df.iloc[max_sharpe_idx][['lips_period', 'jaw_period', 'sharpe_ratio', 'trades']].values

    # Sort medoids by Sharpe ratio
    medoids_sorted = sorted(medoids, key=lambda x: x[2], reverse=True)
    top_medoids = medoids_sorted[:5]  # Get top 5 medoids by Sharpe ratio

    # Print results
    print("\n----- HIERARCHICAL CLUSTERING RESULTS FOR CLASSIC ALLIGATOR STRATEGY -----")
    print(f"Max Sharpe point: Lips={int(max_sharpe_point[0])} (shift=3), Jaw={int(max_sharpe_point[1])} (shift=8), "
          f"Sharpe={max_sharpe_point[2]:.4f}, Trades={int(max_sharpe_point[3])}")

    print("\nTop 5 Medoids (by Sharpe ratio):")
    for idx, medoid in enumerate(top_medoids, 1):
        print(f"Top {idx}: Lips={int(medoid[0])} (shift=3), Jaw={int(medoid[1])} (shift=8), "
              f"Sharpe={medoid[2]:.4f}, Trades={int(medoid[3])}")

    # Create visualization with clustering results
    create_hierarchical_cluster_visualization(X_valid, medoids, top_medoids, max_sharpe_point, labels_valid)

    return X_valid, medoids, top_medoids, max_sharpe_point, labels_valid


def plot_alligator_performance(jaw_period, lips_period, teeth_period, top_medoids=None, slippage=SLIPPAGE, file_path=None):
    """
    Plot the Classic Alligator strategy performance using the best parameters and include top medoids
    
    Parameters:
    jaw_period: int - The jaw SMA period
    lips_period: int - The lips SMA period
    teeth_period: int - The teeth SMA period
    top_medoids: list - List of top medoids, each as (lips_period, jaw_period, sharpe, trades)
    slippage: float - Slippage in price units
    file_path: str - Optional path to save the performance summary
    """
    from Alligator_Strategy import AlligatorStrategy
    
    print(f"\n----- PLOTTING CLASSIC ALLIGATOR STRATEGY PERFORMANCE -----")
    print(f"Using Jaw Period: {jaw_period} (shift=8), Teeth Period: {teeth_period} (shift=5), Lips Period: {lips_period} (shift=3)")
    print(f"Trading with fixed position size of {CONTRACT_SIZE} contracts")
    if top_medoids:
        print(f"Including top {len(top_medoids)} medoids")

    # Download historical data
    print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
    data = yf.download(TICKER, start=START_DATE, end=END_DATE)

    # Simplify the data structure
    data.columns = data.columns.get_level_values(0)

    # Create a dictionary to store results for each strategy
    strategies = {
        'Best': {'jaw': jaw_period, 'teeth': teeth_period, 'lips': lips_period}
    }

    # Add medoids if provided
    if top_medoids:
        for i, medoid in enumerate(top_medoids, 1):
            # Get the closest teeth period
            lips_period = int(medoid[0])
            jaw_period = int(medoid[1])
            # Estimate teeth as the middle value, or use the fixed teeth_period from the optimization
            teeth_period_medoid = teeth_period  # Use the same teeth for all medoids
            strategies[f'Medoid {i}'] = {'jaw': jaw_period, 'teeth': teeth_period_medoid, 'lips': lips_period}

    # Apply the proper strategy for each parameter set
    for name, params in strategies.items():
        # Create a strategy instance for each parameter set
        alligator_strategy = AlligatorStrategy(
            jaw_period=params['jaw'],
            teeth_period=params['teeth'],
            lips_period=params['lips'],
            contract_size=CONTRACT_SIZE,
            slippage=slippage
        )

        # Apply the strategy
        data = alligator_strategy.apply_strategy(
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
                 label=f'{name} (Jaw={params["jaw"]}/Teeth={params["teeth"]}/Lips={params["lips"]})', color=color)

        # Plot out-of-sample portion with thicker line
        plt.plot(data.index[split_index:], data[f'Cumulative_Returns_{name}'].iloc[split_index:],
                 color=color, linewidth=2.5, alpha=0.7)

    plt.axvline(x=split_date, color='black', linestyle='--',
                label=f'Train/Test Split ({int(TRAIN_TEST_SPLIT * 100)}%/{int((1 - TRAIN_TEST_SPLIT) * 100)}%)')
    plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5,
                label=f'Initial Capital (${INITIAL_CAPITAL:,})')
    plt.legend(loc='upper left')
    plt.title('Classic Alligator Strategy Performance (Multiple of Initial Capital)')
    plt.ylabel('Cumulative Return (x initial)')

    # Plot account balance (now as the second subplot)
    plt.subplot(2, 1, 2)

    for i, (name, params) in enumerate(strategies.items()):
        color = colors[i % len(colors)]

        # Plot full period
        plt.plot(data.index, data[f'Capital_{name}'],
                 label=f'{name} (Jaw={params["jaw"]}/Teeth={params["teeth"]}/Lips={params["lips"]})', color=color)

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
    output_file = os.path.join(OUTPUT_DIR, 'classic_alligator_performance_with_medoids.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Strategy performance chart with medoids saved as '{output_file}'")
    plt.show()

    # Create a list to store performance data for saving to file
    performance_data = []

    # Print detailed performance metrics for all strategies with in-sample and out-of-sample breakdown
    print("\n----- PERFORMANCE SUMMARY -----")

    # Open a file to save the performance summary if requested
    if file_path is None:
        file_path = os.path.join(OUTPUT_DIR, 'classic_alligator_performance_summary.txt')
        
    with open(file_path, 'w') as f:
        f.write("----- CLASSIC ALLIGATOR STRATEGY PERFORMANCE SUMMARY -----\n\n")

        # IN-SAMPLE PERFORMANCE
        print("\nIN-SAMPLE PERFORMANCE:")
        f.write("IN-SAMPLE PERFORMANCE:\n")
        header = f"{'Strategy':<10} | {'Jaw/Teeth/Lips':<15} | {'P&L':>12} | {'Return %':>10} | {'Sharpe':>7} | {'Trades':>6}"
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)
        f.write(separator + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")

        for name, params in strategies.items():
            jaw = params['jaw']
            teeth = params['teeth']
            lips = params['lips']

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
            row = f"{name:<10} | {jaw:>2}/{teeth:>2}/{lips:<2} | ${in_sample_cumulative_pnl:>10,.2f} | {in_sample_return_percent:>9.2f}% | {in_sample_sharpe:>6.3f} | {in_sample_trades:>6}"

            # Print and save row
            print(row)
            f.write(row + "\n")

            # Store the data for later export
            performance_data.append({
                'Period': 'In-Sample',
                'Strategy': name,
                'Jaw_Period': jaw,
                'Teeth_Period': teeth,
                'Lips_Period': lips,
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
            jaw = params['jaw']
            teeth = params['teeth']
            lips = params['lips']

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
            row = f"{name:<10} | {jaw:>2}/{teeth:>2}/{lips:<2} | ${out_sample_cumulative_pnl:>10,.2f} | {out_sample_return_percent:>9.2f}% | {out_sample_sharpe:>6.3f} | {out_sample_trades:>6}"

            # Print and save row
            print(row)
            f.write(row + "\n")

            # Store the data for later export
            performance_data.append({
                'Period': 'Out-of-Sample',
                'Strategy': name,
                'Jaw_Period': jaw,
                'Teeth_Period': teeth,
                'Lips_Period': lips,
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
            jaw = params['jaw']
            teeth = params['teeth']
            lips = params['lips']

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
            row = f"{name:<10} | {jaw:>2}/{teeth:>2}/{lips:<2} | ${full_cumulative_pnl:>10,.2f} | {full_return_percent:>9.2f}% | {full_sharpe:>6.3f} | {full_trades:>6}"

            # Print and save row
            print(row)
            f.write(row + "\n")

            # Store the data for later export
            performance_data.append({
                'Period': 'Full',
                'Strategy': name,
                'Jaw_Period': jaw,
                'Teeth_Period': teeth,
                'Lips_Period': lips,
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
        f.write(f"Contract Size: {CONTRACT_SIZE}\n")
        f.write(f"Slippage: {slippage}\n")
        f.write(f"Jaw Shift: 8 periods\n")
        f.write(f"Teeth Shift: 5 periods\n")
        f.write(f"Lips Shift: 3 periods\n")

    print(f"\nPerformance summary saved to '{file_path}'")

    # Save performance data to CSV for further analysis
    csv_file = os.path.join(OUTPUT_DIR, 'classic_alligator_performance_data.csv')
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(csv_file, index=False)
    print(f"Performance data saved to '{csv_file}'")

    return data


def bimonthly_out_of_sample_comparison(data, best_jaw, best_teeth, best_lips, top_medoid, slippage=SLIPPAGE):
    """
    Compare bimonthly (2-month) performance between the best Alligator strategy and the top medoid
    
    Parameters:
    data: DataFrame with market data
    best_jaw: int - The jaw period for the best strategy
    best_teeth: int - The teeth period for the best strategy
    best_lips: int - The lips period for the best strategy
    top_medoid: tuple - The top medoid as (lips_period, jaw_period, sharpe, trades)
    slippage: float - Slippage in price units
    """
    from Alligator_Strategy import AlligatorStrategy
    
    print(f"\n----- BIMONTHLY OUT-OF-SAMPLE COMPARISON -----")
    print(f"Comparing Best Alligator (Jaw={best_jaw}/Teeth={best_teeth}/Lips={best_lips}) vs "
          f"Top Medoid (Jaw={int(top_medoid[1])}/Teeth={best_teeth}/Lips={int(top_medoid[0])})")
    
    # Download historical data if not already provided
    if data is None:
        print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
        data = yf.download(TICKER, start=START_DATE, end=END_DATE)
        # Simplify the data structure
        data.columns = data.columns.get_level_values(0)
    
    # Create strategies
    strategies = {
        'Best': {'jaw': best_jaw, 'teeth': best_teeth, 'lips': best_lips},
        'Medoid_1': {'jaw': int(top_medoid[1]), 'teeth': best_teeth, 'lips': int(top_medoid[0])}
    }
    
    # Apply each strategy to the data
    for name, params in strategies.items():
        strategy = AlligatorStrategy(
            jaw_period=params['jaw'],
            teeth_period=params['teeth'],
            lips_period=params['lips'],
            contract_size=CONTRACT_SIZE,
            slippage=slippage
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
    oos_data['year'] = oos_data.index.year
    # Assign bimonthly period (1-6 for each year)
    oos_data['bimonthly'] = ((oos_data.index.month - 1) // 2) + 1
    
    # Create a simplified period label with format YYYY-MM (showing start month of period)
    # Convert year to int first to handle any float values
    oos_data['year'] = oos_data['year'].astype(int)
    oos_data['bimonthly'] = oos_data['bimonthly'].astype(int)
    
    oos_data['period_label'] = oos_data.apply(
        lambda row: f"{row['year']}-{int((row['bimonthly'] - 1) * 2 + 1):02d}", 
        axis=1
    )
    
    # Create a DataFrame to store bimonthly Sharpe ratios
    bimonthly_sharpe = []
    
    # Group by year and bimonthly period, calculate Sharpe ratio for each period
    for period_label, group in oos_data.groupby('period_label'):
        # Calculate trading days in the period
        if len(group) < 10:  # Skip periods with too few trading days
            continue
            
        # Extract year and month from period_label
        year, month_str = period_label.split('-')
        # Handle potential float strings (e.g., '2021.0')
        year = int(float(year))
        start_month = int(month_str)
        
        # Create a bimonthly result entry
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
        
        bimonthly_sharpe.append(bimonthly_result)
    
    # Convert to DataFrame
    bimonthly_sharpe_df = pd.DataFrame(bimonthly_sharpe)
    
    # Sort the DataFrame by date for proper chronological display
    if not bimonthly_sharpe_df.empty:
        bimonthly_sharpe_df = bimonthly_sharpe_df.sort_values('date')
    
    # Add rounded values to dataframe for calculations
    bimonthly_sharpe_df['Best_sharpe_rounded'] = np.round(bimonthly_sharpe_df['Best_sharpe'], 2)
    bimonthly_sharpe_df['Medoid_1_sharpe_rounded'] = np.round(bimonthly_sharpe_df['Medoid_1_sharpe'], 2)
    
    # Print detailed comparison of Sharpe ratios
    print("\nDetailed Sharpe ratio comparison by period:")
    print(f"{'Period':<10} | {'Best Sharpe':>12} | {'Medoid_1':>12} | {'Difference':>12} | {'Raw Win':<7} | {'Rounded Win':<7}")
    print("-" * 70)
    
    for idx, row in bimonthly_sharpe_df.iterrows():
        period = row['period_label']
        best_sharpe = row['Best_sharpe']
        medoid_sharpe = row['Medoid_1_sharpe']
        best_rounded = row['Best_sharpe_rounded']
        medoid_rounded = row['Medoid_1_sharpe_rounded']
        
        diff = medoid_sharpe - best_sharpe
        diff_rounded = medoid_rounded - best_rounded
        
        raw_win = medoid_sharpe > best_sharpe
        rounded_win = medoid_rounded > best_rounded
        
        print(f"{period:<10} | {best_sharpe:12.6f} | {medoid_sharpe:12.6f} | {diff:12.6f} | {raw_win!s:<7} | {rounded_win!s:<7}")
    
    # Calculate win rate using raw values (original method)
    medoid_wins = sum(bimonthly_sharpe_df['Medoid_1_sharpe'] > bimonthly_sharpe_df['Best_sharpe'])
    total_periods = len(bimonthly_sharpe_df)
    win_percentage = (medoid_wins / total_periods) * 100 if total_periods > 0 else 0
    
    # Calculate win rate using rounded values (new method)
    medoid_wins_rounded = sum(bimonthly_sharpe_df['Medoid_1_sharpe_rounded'] > bimonthly_sharpe_df['Best_sharpe_rounded'])
    win_percentage_rounded = (medoid_wins_rounded / total_periods) * 100 if total_periods > 0 else 0
    
    print(f"\nBimonthly periods analyzed: {total_periods}")
    print(f"Raw comparison (full precision):")
    print(f"Periods where Medoid_1 outperformed Best: {medoid_wins}")
    print(f"Win percentage: {win_percentage:.2f}%")
    
    print(f"\nRounded comparison (2 decimal places):")
    print(f"Periods where Medoid_1 outperformed Best: {medoid_wins_rounded}")
    print(f"Win percentage: {win_percentage_rounded:.2f}%")
    
    # Update the win percentage for display in the chart to use the rounded value
    win_percentage = win_percentage_rounded
    medoid_wins = medoid_wins_rounded
    
    # Create a bar plot to compare bimonthly Sharpe ratios
    plt.figure(figsize=(14, 8))
    
    # Set up x-axis dates
    x = np.arange(len(bimonthly_sharpe_df))
    width = 0.35  # Width of the bars
    
    # Create bars
    plt.bar(x - width/2, bimonthly_sharpe_df['Best_sharpe'], width, 
           label=f'Best (Jaw={best_jaw}/Teeth={best_teeth}/Lips={best_lips})', color='blue')
    plt.bar(x + width/2, bimonthly_sharpe_df['Medoid_1_sharpe'], width, 
           label=f'Medoid 1 (Jaw={int(top_medoid[1])}/Teeth={best_teeth}/Lips={int(top_medoid[0])})', color='green')
    
    # Add a horizontal line at Sharpe = 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    plt.title(f'Bimonthly Sharpe Ratio Comparison (Out-of-Sample Period)\nMedoid_1 outperformed {win_percentage:.2f}% of the time', 
              fontsize=14)
    plt.xlabel('Bimonthly Period (Start Month)', fontsize=12)
    plt.ylabel('Sharpe Ratio (Annualized)', fontsize=12)
    
    # Use improved x-axis labels with 45-degree rotation
    plt.xticks(x, bimonthly_sharpe_df['period_label'], rotation=45)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create legend with both strategies at the top right
    best_sharpe_patch = plt.Rectangle((0, 0), 1, 1, fc="blue")
    medoid_patch = plt.Rectangle((0, 0), 1, 1, fc="green")
    
    plt.legend(
        [best_sharpe_patch, medoid_patch],
        [f'Best (Jaw={best_jaw}/Teeth={best_teeth}/Lips={best_lips})', 
         f'Medoid 1 (Jaw={int(top_medoid[1])}/Teeth={best_teeth}/Lips={int(top_medoid[0])})'],
        loc='upper right', 
        frameon=True, 
        fancybox=True, 
        framealpha=0.9,
        fontsize=10
    )
    
    # Add a text box with win percentage (using rounded values)
    plt.annotate(f'Medoid_1 Win Rate: {win_percentage:.2f}%\n'
                 f'({medoid_wins} out of {total_periods} periods)\n',
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=12)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'classic_alligator_bimonthly_sharpe_comparison.png') 
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nBimonthly comparison chart saved as '{output_file}'")
    plt.show()
    
    # Save the bimonthly data to CSV
    csv_file = os.path.join(OUTPUT_DIR, 'classic_alligator_bimonthly_sharpe_comparison.csv')
    bimonthly_sharpe_df.to_csv(csv_file, index=False)
    print(f"Bimonthly comparison data saved to '{csv_file}'")