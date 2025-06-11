import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines


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

    # Save and display plot
    plt.tight_layout()
    plt.savefig('sma_clusters.png', dpi=300, bbox_inches='tight')
    print("Clustering visualization saved as 'sma_clusters.png'")
    plt.show()
