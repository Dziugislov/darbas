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

from input_gen import *
from data_gather_gen import *
from data_analysis_gen import *
from SMA_Strategy import SMAStrategy

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",   #no timestamp
    handlers=[
        logging.FileHandler("execution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
SYMBOL = TICKER
WORKING_DIR = "."  # Current directory
DATA_DIR = os.path.join(WORKING_DIR, "data")
OUTPUT_DIR = os.path.join('.', 'output2', SYMBOL)
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXCEL_FILE_PATH = os.path.join(os.getcwd(), 'Results.xlsx')
ANALYSIS_METHOD = "Hierarchical"  # Set to "Kmeans" or "Hierarchical" depending on the analysis type

def calculate_elbow_curve_hierarchical(X_scaled):
    """
    Calculate the elbow curve for KMeans clustering
    
    Parameters:
    X_scaled: numpy array - Scaled data points
    max_clusters: int - Maximum number of clusters to try
    
    Returns:
    distortions: list - Distortion values for each k
    k_values: list - K values used
    optimal_k: int - Optimal number of clusters based on elbow method
    """
    logging.info("\nCalculating elbow curve...")
    distortions = []
    k_values = range(1, DEFAULT_NUM_CLUSTERS + 1)
    
    for k in k_values:
        if k == 1:
            # For k=1, distortion is just the sum of squared distances to mean
            distortion = np.sum((X_scaled - np.mean(X_scaled, axis=0)) ** 2)
            distortions.append(distortion)
            continue
            
        clustering = AgglomerativeClustering(n_clusters=k)
        clustering.fit(X_scaled)
        
        # Calculate distortion for current clustering
        centroids = np.array([X_scaled[clustering.labels_ == i].mean(axis=0) for i in range(k)])
        distortion = np.sum([np.sum((X_scaled[clustering.labels_ == i] - centroids[i]) ** 2) 
                           for i in range(k)])
        distortions.append(distortion)
    
    # Calculate the percentage changes
    pct_changes = np.diff(distortions) / np.array(distortions)[:-1] * 100
    
    # Find optimal k using the threshold method
    optimal_k = 1
    for i, pct_change in enumerate(pct_changes):
        if abs(pct_change) < ELBOW_THRESHOLD:
            optimal_k = i + 1  # +1 because we start from k=1
            break
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, distortions, 'bo-')
    plt.axvline(x=optimal_k, color='r', linestyle='--', 
                label=f'Optimal k={optimal_k}\n(threshold={ELBOW_THRESHOLD}%)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title(f'{SYMBOL} Hierarchical Elbow Method for Optimal k')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(f'{SYMBOL}_Hierarchical_Elbow_Curve.png', OUTPUT_DIR)
    
    logging.info(f"Optimal number of clusters from elbow method: {optimal_k}")
    return distortions, k_values, optimal_k


def compute_hierarchical_clusters(X, labels, valid_clusters):
    """
    Compute clusters for each valid cluster from hierarchical clustering
    A cluster is the data point in a cluster that has the minimum average distance to all other points in the cluster

    Parameters:
    X: numpy array of shape (n_samples, n_features) - Original data points (not scaled)
    labels: numpy array of shape (n_samples,) - Cluster labels for each data point
    valid_clusters: set - Set of valid cluster IDs

    Returns:
    list of tuples - Each tuple contains (short_SMA, long_SMA, sharpe_ratio, trades) for each cluster
    """

    # Initialize list to store clusters for each cluster
    clusters = []

    # Process each valid cluster
    for cluster_id in valid_clusters:
        # Extract points belonging to this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = X[cluster_indices]

        if len(cluster_points) == 0:
            continue

        if len(cluster_points) == 1:
            # If only one point in cluster, it's the cluster
            clusters.append(tuple(cluster_points[0]))
            continue

        # Compute pairwise distances between all points in this cluster
        distances = cdist(cluster_points, cluster_points, metric='euclidean')

        # For each point, compute sum of distances to all other points
        total_distances = np.sum(distances, axis=1)

        # Find index of point with minimum total distance
        medoid_idx = np.argmin(total_distances)

        # Get the actual cluster point
        cluster = cluster_points[medoid_idx]

        # Store the cluster as a tuple
        clusters.append(tuple(cluster))

        # Print cluster info
        logging.info(f"Cluster {cluster_id} cluster: Short SMA={int(cluster[0])}, Long SMA={int(cluster[1])}, "
            f"Sharpe={cluster[2]:.4f}, Trades={int(cluster[3])}")

    return clusters

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
    # Use a sample if there are too many points
    if len(X_scaled) > 1000:
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(X_scaled), size=1000, replace=False)
        X_sample = X_scaled[sample_indices]
        logging.info(f"Using a random sample of 1000 points for dendrogram creation (out of {len(X_scaled)} total points)")
    else:
        X_sample = X_scaled

    # Compute the distance matrix
    dist_matrix = pdist(X_sample, metric='euclidean')

    # Compute the linkage matrix
    Z = shc.linkage(dist_matrix, method=method)

    # Print some statistics about the linkage
    logging.info(f"\nDendrogram using {method} linkage method:")
    logging.info(f"Number of data points: {len(X_sample)}")
    logging.info(f"Cophenetic correlation: {shc.cophenet(Z, dist_matrix)[0]:.4f}")

    # Create a new figure
    plt.figure(figsize=figsize)

    # Set the background color
    plt.gca().set_facecolor('white')

    # Set a title
    plt.title(f'{SYMBOL} Hierarchical Clustering Dendrogram ({method} linkage)', fontsize=14)

    # Generate the dendrogram
    dendrogram = shc.dendrogram(
        Z,
        truncate_mode='lastp',  # Only show the last p merges
        p=30,  # Show only 30 merges
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

    # Save the plot
    save_plot(f'{SYMBOL}_Hierarchical_Dendrogram.png', OUTPUT_DIR)

    return Z

def hierarchical_cluster_analysis(file_path='optimized_strategies.pkl'):
    """
    Perform hierarchical clustering analysis on SMA optimization results to find robust parameter regions

    Parameters:
    file_path: str - Path to the file with SMA results
    min_trades: int - Minimum number of trades to consider
    max_trades: int - Maximum number of trades to consider
    min_elements_per_cluster: int - Minimum number of elements per cluster

    Returns:
    tuple - (X_filtered, clusters, top_clusters, max_sharpe_point, labels)
    """
    logging.info(f"\n----- HIERARCHICAL CLUSTER ANALYSIS -----")
    logging.info(f"Loading data from {file_path}...")

    # Load the data
    df = pd.read_pickle(file_path)

    # Convert data to numpy array for easier processing
    X = df[['short_SMA', 'long_SMA', 'sharpe_ratio', 'trades']].values

    # Filter data by number of trades and ensure short_SMA < long_SMA
    X_filtered = X[(X[:, 0] < X[:, 1]) &  # short_SMA < long_SMA
                (X[:, 3] >= MIN_TRADES) &  # trades >= min_trades
                (X[:, 3] <= MAX_TRADES)]  # trades <= max_trades

    if len(X_filtered) == 0:
        logging.info(
            f"No data points meet the criteria after filtering! Adjust min_trades ({MIN_TRADES}) and max_trades ({MAX_TRADES}).")
        return None, None, None, None, None

    logging.info(f"Filtered data to {len(X_filtered)} points with {MIN_TRADES}-{MAX_TRADES} trades")

    # Extract the fields for better scaling visibility
    short_sma_values = X_filtered[:, 0]
    long_sma_values = X_filtered[:, 1]
    sharpe_values = X_filtered[:, 2]
    trades_values = X_filtered[:, 3]

    logging.info(f"Short SMA range: {short_sma_values.min()} to {short_sma_values.max()}")
    logging.info(f"Long SMA range: {long_sma_values.min()} to {long_sma_values.max()}")
    logging.info(f"Sharpe ratio range: {sharpe_values.min():.4f} to {sharpe_values.max():.4f}")
    logging.info(f"Trades range: {trades_values.min()} to {trades_values.max()}")

    # Scale the data for clustering - using StandardScaler for each dimension
    # This addresses the issue where SMA values have much larger ranges than Sharpe ratio
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Print scaling info for verification
    logging.info("\nScaled data information:")
    scaled_short = X_scaled[:, 0]
    scaled_long = X_scaled[:, 1]
    scaled_sharpe = X_scaled[:, 2]
    scaled_trades = X_scaled[:, 3]

    logging.info(f"Scaled Short SMA range: {scaled_short.min():.4f} to {scaled_short.max():.4f}")
    logging.info(f"Scaled Long SMA range: {scaled_long.min():.4f} to {scaled_long.max():.4f}")
    logging.info(f"Scaled Sharpe ratio range: {scaled_sharpe.min():.4f} to {scaled_sharpe.max():.4f}")
    logging.info(f"Scaled Trades range: {scaled_trades.min():.4f} to {scaled_trades.max():.4f}")

    # Create a dendrogram to help choose the number of clusters
    logging.info("\nCreating dendrogram to visualize hierarchical structure...")
    linkage_method = 'ward'  # Ward minimizes the variance within clusters
    Z = create_dendrogram(X_scaled, method=linkage_method,
                        figsize=(12, 8))

    # Calculate optimal number of clusters using elbow method
    logging.info("\nDetermining optimal number of clusters using elbow method...")
    _, _, k = calculate_elbow_curve_hierarchical(X_scaled)
    logging.info(f"Using {k} clusters based on elbow method (threshold={ELBOW_THRESHOLD}%)")

    # Apply hierarchical clustering
    logging.info(f"Performing hierarchical clustering with {k} clusters using {linkage_method} linkage...")
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

    logging.info("\nCluster sizes:")
    for cluster_id, size in cluster_sizes.items():
        logging.info(f"Cluster {cluster_id}: {size} elements")

    # Filter clusters with enough elements
    valid_clusters = {i for i, count in cluster_sizes.items() if count >= MIN_ELEMENTS_PER_CLUSTER}
    if not valid_clusters:
        logging.info(f"No clusters have at least {MIN_ELEMENTS_PER_CLUSTER} elements! Reducing threshold to 1...")
        valid_clusters = set(unique_labels)
    else:
        logging.info(f"Using {len(valid_clusters)} clusters with at least {MIN_ELEMENTS_PER_CLUSTER} elements")

    filtered_indices = np.array([i in valid_clusters for i in labels])

    # Filter data to only include points in valid clusters
    X_valid = X_filtered[filtered_indices]
    labels_valid = labels[filtered_indices]

    # Compute clusters of hierarchical clusters
    logging.info("Computing clusters for each cluster...")
    clusters = compute_hierarchical_clusters(X_valid, labels_valid, valid_clusters)

    # Find max Sharpe ratio point overall
    max_sharpe_idx = np.argmax(df['sharpe_ratio'].values)
    max_sharpe_point = df.iloc[max_sharpe_idx][['short_SMA', 'long_SMA', 'sharpe_ratio', 'trades']].values

    # Sort clusters by Sharpe ratio
    medoids_sorted = sorted(clusters, key=lambda x: x[2], reverse=True)
    top_clusters = medoids_sorted[:5]  # Get top 5 clusters by Sharpe ratio

    # Print results
    logging.info("\n----- HIERARCHICAL CLUSTERING RESULTS -----")
    logging.info(f"Max Sharpe point: Short SMA={int(max_sharpe_point[0])}, Long SMA={int(max_sharpe_point[1])}, "
        f"Sharpe={max_sharpe_point[2]:.4f}, Trades={int(max_sharpe_point[3])}")

    logging.info("\nTop 5 Medoids (by Sharpe ratio):")
    for idx, cluster in enumerate(top_clusters[:3], 1):
        logging.info(f"Top {idx}: Short SMA={int(cluster[0])}, Long SMA={int(cluster[1])}, "
            f"Sharpe={cluster[2]:.4f}, Trades={int(cluster[3])}")

    # Create visualization with clustering results
    create_hierarchical_cluster_visualization(X_filtered, clusters, top_clusters, max_sharpe_point, labels)

    # Save clustering results to CSV
    clustering_results = []
    for idx, cluster in enumerate(medoids_sorted):
        clustering_results.append({
            'Rank': idx + 1,
            'Short_SMA': int(cluster[0]),
            'Long_SMA': int(cluster[1]),
            'Sharpe': cluster[2],
            'Trades': int(cluster[3])
        })
    
    clustering_df = pd.DataFrame(clustering_results)

    # Return exactly 5 values
    return X_filtered, clusters, top_clusters, max_sharpe_point, labels

def create_hierarchical_cluster_visualization(X, clusters, top_clusters, max_sharpe_point, labels):
    """
    Create visualization of hierarchical clustering results with scatter plots
    Only includes points from valid clusters that meet min_trades and min_elements_per_cluster requirements

    Parameters:
    X: numpy array - Data points that already meet min_trades requirement
    clusters: list of tuples - Cluster clusters (these are already from valid clusters only)
    top_clusters: list of tuples - Top clusters by Sharpe ratio
    max_sharpe_point: tuple - Point with maximum Sharpe ratio
    labels: numpy array - Cluster labels for the points in X
    """
    logging.info("Creating hierarchical cluster visualization...")
    
    # Determine which clusters are represented in the clusters
    # This is our definition of valid clusters - ones that have clusters
    valid_cluster_ids = set([labels[np.where((X[:, 0] == cluster[0]) & (X[:, 1] == cluster[1]))[0][0]] for cluster in clusters])
    logging.info(f"Using {len(valid_cluster_ids)} valid clusters with clusters")
    
    # Filter X to only include points from valid clusters
    valid_indices = np.array([label in valid_cluster_ids for label in labels])
    X_valid = X[valid_indices]
    labels_valid = labels[valid_indices]
    
    logging.info(f"Filtering to {len(X_valid)} points from valid clusters")
    
    # Create a DataFrame for the valid points to use in visualization
    filtered_df = pd.DataFrame(X_valid, columns=['short_SMA', 'long_SMA', 'sharpe_ratio', 'trades'])

    # Create a pivot table for the heatmap from the filtered data
    heatmap_data = filtered_df.pivot_table(
        index='long_SMA',
        columns='short_SMA',
        values='sharpe_ratio',
        fill_value=np.nan  # Use NaN for empty cells
    )
    
    # Create a new figure with 2 subplots (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Get the number of unique valid clusters
    n_clusters = len(valid_cluster_ids)

    # Create a colormap with distinct colors for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Create a mapping from cluster ID to color index
    cluster_to_color = {cluster_id: i for i, cluster_id in enumerate(valid_cluster_ids)}

    # Plot 1: Short SMA vs Long SMA (colored by cluster)
    for cluster_id in valid_cluster_ids:
        # Get indices of points in this valid cluster
        cluster_points = X_valid[labels_valid == cluster_id]
        if len(cluster_points) == 0:
            continue

        # Get the color for this cluster
        color_idx = cluster_to_color[cluster_id]
        color = colors[color_idx % len(colors)]  # Use modulo to handle if we have more clusters than colors
        
        # Plot the points
        ax1.scatter(
            cluster_points[:, 0],  # Short SMA
            cluster_points[:, 1],  # Long SMA
            s=50, c=[color], alpha=0.7
        )

    # Plot the clusters with black edges
    for i, cluster in enumerate(clusters):
        ax1.scatter(
            cluster[0],  # Short SMA
            cluster[1],  # Long SMA
            s=150, c='white', alpha=1,
            edgecolors='black', linewidths=2,
            marker='o'
        )

    # Plot the top clusters with star markers
    for i, cluster in enumerate(top_clusters):
        ax1.scatter(
            cluster[0],  # Short SMA
            cluster[1],  # Long SMA
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
    max_val = max(X_valid[:, 0].max(), X_valid[:, 1].max()) * 1.1
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

    # Create a simplified legend with just the key elements
    custom_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
            markeredgecolor='black', markeredgewidth=2, markersize=10, label='Cluster Cluster'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
            markeredgecolor='black', markersize=10, label='Top Cluster'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
            markeredgecolor='black', markersize=10, label='Best Sharpe Point')
    ]
    
    # Add custom handles to legend
    ax1.legend(handles=custom_handles, loc='upper right')
    
    # Plot 2: Create a mask for the heatmap where short_SMA >= long_SMA or values are NaN
    mask = np.zeros_like(heatmap_data, dtype=bool)
    for i, long_sma in enumerate(heatmap_data.index):
        for j, short_sma in enumerate(heatmap_data.columns):
            if short_sma >= long_sma or np.isnan(heatmap_data.iloc[i, j]):
                mask[i, j] = True

    # Plot the heatmap on ax2
    sns.heatmap(
        heatmap_data,
        ax=ax2,
        mask=mask,
        cmap='coolwarm',  # Blue to red colormap
        annot=False,      # Don't annotate each cell with its value
        fmt='.4f',
        linewidths=0,
        cbar_kws={'label': 'Sharpe Ratio'}
    )
    
    # Invert the y-axis so smaller long_SMA values are at the top
    ax2.invert_yaxis()
    
    # Plot the clusters (Black Squares)
    for cluster in clusters:
        try:
            x_pos = np.where(heatmap_data.columns == cluster[0])[0][0] + 0.5
            y_pos = np.where(heatmap_data.index == cluster[1])[0][0] + 0.5
            ax2.scatter(x_pos, y_pos, marker='s', color='black', s=75, zorder=4)
        except IndexError:
            logging.info(f"Warning: Cluster at ({cluster[0]}, {cluster[1]}) not found in heatmap coordinates")

    # Plot top clusters (Purple Diamonds)
    for cluster in top_clusters:
        try:
            x_pos = np.where(heatmap_data.columns == cluster[0])[0][0] + 0.5
            y_pos = np.where(heatmap_data.index == cluster[1])[0][0] + 0.5
            ax2.scatter(x_pos, y_pos, marker='D', color='purple', s=100, zorder=5)
        except IndexError:
            logging.info(f"Warning: Top cluster at ({cluster[0]}, {cluster[1]}) not found in heatmap coordinates")

    # Plot max Sharpe point (Green Star)
    try:
        best_x_pos = np.where(heatmap_data.columns == max_sharpe_point[0])[0][0] + 0.5
        best_y_pos = np.where(heatmap_data.index == max_sharpe_point[1])[0][0] + 0.5
        ax2.scatter(best_x_pos, best_y_pos, marker='*', color='lime', s=200,
                edgecolor='black', zorder=5)
    except IndexError:
        logging.info(f"Warning: Max Sharpe point at ({max_sharpe_point[0]}, {max_sharpe_point[1]}) not found in heatmap coordinates")

    # Create custom legend for the heatmap
    max_sharpe_handle = mlines.Line2D([], [], color='lime', marker='*', linestyle='None',
                                markersize=15, markeredgecolor='black', label='Max Sharpe')
    medoid_handle = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                            markersize=10, label='Clusters')
    top_medoid_handle = mlines.Line2D([], [], color='purple', marker='D', linestyle='None',
                                markersize=10, label='Top 5 Clusters')

    # Add a legend to the second plot
    ax2.legend(handles=[max_sharpe_handle, medoid_handle, top_medoid_handle],
            loc='best')

    # Add labels and title
    ax2.set_title(f'{SYMBOL} Hierarchical Clustering Analysis (Sharpe Ratio)', fontsize=14)
    ax2.set_xlabel('Short SMA (days)', fontsize=12)
    ax2.set_ylabel('Long SMA (days)', fontsize=12)

    # Rotate tick labels
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    save_plot(f'{SYMBOL}_Hierarchical_Clusters_Visualization.png', OUTPUT_DIR)

def main():

    # Load the parameters
    big_point_value, slippage, capital, atr_period = load_parameters()
    logging.info(f"Big Point Value: {big_point_value}")
    logging.info(f"Slippage: {slippage}")
    logging.info(f"Capital for Position Sizing: {capital:,}")
    logging.info(f"ATR Period: {atr_period}")
    
    # Set matplotlib backend explicitly
    matplotlib.use('Agg')  # Use non-interactive backend for headless environments
    
    logging.info("Starting SMA strategy analysis with hierarchical clustering...")
    
    # Run the basic analysis first
    data, best_short, best_long, best_sharpe, best_trades = analyze_sma_results()

    if data is None:
        logging.info("Error: Failed to load or analyze SMA results data.")
        exit(1)

    logging.info("\nProceeding with hierarchical cluster analysis...")
    
    # Run the hierarchical cluster analysis to get clusters
    X_filtered, clusters, top_clusters, max_sharpe_point, labels = hierarchical_cluster_analysis()
    
    if X_filtered is None or clusters is None:
        logging.info("Error: Hierarchical cluster analysis failed.")
        exit(1)

    logging.info("\nPlotting strategy performance...")
    
    # Plot strategy performance with the best parameters AND top clusters using ATR-based position sizing
    market_data = plot_strategy_performance(
        best_short, best_long, top_clusters, 
        big_point_value=big_point_value, 
        slippage=slippage,
        ANALYSIS_METHOD=ANALYSIS_METHOD
    )
    
    # Run the bimonthly out-of-sample comparison between best Sharpe and top clusters
    if top_clusters and len(top_clusters) > 0:
        logging.info("\nPerforming bimonthly out-of-sample comparison with hierarchical clustering...")
        bimonthly_sharpe_df = bimonthly_out_of_sample_comparison(
            market_data, 
            best_short, 
            best_long, 
            top_clusters,  # Pass the entire list of top clusters
            big_point_value=big_point_value,
            slippage=slippage,
            ANALYSIS_METHOD=ANALYSIS_METHOD
        )
    else:
        logging.info("No top clusters found. Cannot run bimonthly comparison.")
        
    # After bimonthly_out_of_sample_comparison in main, call the new function if top_clusters are available
    if top_clusters and len(top_clusters) > 0:
        logging.info("\nPerforming full out-of-sample performance analysis (Hierarchical)...")
        analyze_full_oos_performance(
            market_data,
            best_short,
            best_long,
            top_clusters,
            big_point_value=big_point_value,
            slippage=slippage,
            ANALYSIS_METHOD=ANALYSIS_METHOD
        )
    else:
        logging.info("No top clusters found. Cannot run full OOS analysis (Hierarchical).")
        
    logging.info("\nAnalysis complete! All plots and result files have been saved to the output directory.")
    logging.info(f"Output directory: {OUTPUT_DIR}")

# Call the main function to execute the script
if __name__ == "__main__":
    main()