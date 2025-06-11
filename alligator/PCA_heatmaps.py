import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

from input_gen import MIN_TRADES, MAX_TRADES, TICKER

SYMBOL = TICKER.replace('=F', '')

def save_plot(plot_name):
    output_dir = os.path.join('.', 'output2', SYMBOL)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.close()

def plot_pca_heatmaps(file_path='sma_all_results.txt',
                      min_trades=MIN_TRADES,
                      max_trades=MAX_TRADES):
    """
    1) Loads columns: short_SMA, med_SMA, long_SMA, sharpe_ratio.
    2) Filters out bad orderings and trade‐count extremes.
    3) Standardizes and runs PCA to see which variable contributes least to PC1.
    4) Drops that variable in each heatmap by averaging sharpe over it.
    5) Plots three 2D heatmaps:
       - short vs med (averaging over long)
       - short vs long (averaging over med)
       - med vs long (averaging over short)
       Y‐axis is inverted so that lower SMA values appear at the bottom.
    """
    # 1) Load and filter
    df = pd.read_csv(file_path)
    df = df[
        (df['short_SMA'] < df['med_SMA']) &
        (df['med_SMA'] < df['long_SMA']) &
        (df['trades'] >= min_trades) &
        (df['trades'] <= max_trades)
    ]

    # 2) Extract the 4‐D array and standardize
    X = df[['short_SMA', 'med_SMA', 'long_SMA', 'sharpe_ratio']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) PCA to inspect PC1 loadings
    pca = PCA(n_components=4)
    pca.fit(X_scaled)
    loadings = pca.components_[0]
    var_names = ['short_SMA', 'med_SMA', 'long_SMA', 'sharpe_ratio']
    idx_drop = np.argmin(np.abs(loadings))
    var_drop = var_names[idx_drop]
    print("PCA Loadings (PC1):", dict(zip(var_names, loadings)))
    print("Dropping variable for slicing:", var_drop)

    # 4) Prepare combinations for heatmaps
    combos = [
        ('short_SMA', 'med_SMA', 'long_SMA'),
        ('short_SMA', 'long_SMA', 'med_SMA'),
        ('med_SMA', 'long_SMA', 'short_SMA'),
    ]

    for x_var, y_var, drop_var in combos:
        # 5a) Group by x_var & y_var, then take mean sharpe (i.e. average over drop_var)
        pivot_df = (
            df.groupby([x_var, y_var])['sharpe_ratio']
              .mean()
              .reset_index()
        )

        # 5b) Pivot into a matrix
        heatmap_data = pivot_df.pivot(index=y_var, columns=x_var, values='sharpe_ratio')

        # 5c) Mask invalid SMA orderings and NaNs
        mask = np.zeros_like(heatmap_data, dtype=bool)
        for i, y in enumerate(heatmap_data.index):
            for j, x in enumerate(heatmap_data.columns):
                # Enforce the natural ordering: short < med < long
                valid = False
                if x_var == 'short_SMA' and y_var == 'med_SMA':
                    valid = x < y
                elif x_var == 'short_SMA' and y_var == 'long_SMA':
                    valid = x < y
                elif x_var == 'med_SMA' and y_var == 'long_SMA':
                    valid = x < y

                if not valid or np.isnan(heatmap_data.iloc[i, j]):
                    mask[i, j] = True

        # 5d) Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            mask=mask,
            cmap='coolwarm',
            cbar_kws={'label': 'Mean Sharpe'}
        )
        # Invert Y‐axis so lowest values are at the bottom
        plt.gca().invert_yaxis()

        plt.title(f'{SYMBOL} Heatmap: {x_var} vs {y_var} (avg over {drop_var})')
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        save_plot(f'{SYMBOL}_{x_var}_{y_var}_heatmap.png')

def main():
    plot_pca_heatmaps()

if __name__ == '__main__':
    main()
