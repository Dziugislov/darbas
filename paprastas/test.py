import pandas as pd
import matplotlib.pyplot as plt
import random

# --- Master-file (all top-20-percent strategies) ---------------------------------

# Load the pickled wide PnL DataFrame (top-20-% strategies)
pnl_df = pd.read_pickle("master_daily_pnl.pkl")

# Display first and last five rows
print("MASTER FILE – first 5 rows:")
print(pnl_df.head())
print("\nMASTER FILE – last 5 rows:")
print(pnl_df.tail())

# Pick a random strategy column to visualise
strategy_col = random.choice(pnl_df.columns.tolist())
print(f"\nPlotting raw PnL for randomly chosen strategy: {strategy_col}")

plt.figure(figsize=(10, 6))
plt.plot(pnl_df.index, pnl_df[strategy_col])
plt.xlabel("Date")
plt.ylabel("Cumulative PnL")
plt.title(f"Raw PnL Series for {strategy_col}")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Top-5-strategies file --------------------------------------------------------

try:
    top5_df = pd.read_pickle("top5_strategies_daily_pnl.pkl")
except FileNotFoundError:
    print("\n(top5_strategies_daily_pnl.pkl not found – skipping top-5 summary)\n")
else:
    # Print the first and last five rows
    print("\nTOP-5 FILE – first 5 rows:")
    print(top5_df.head())
    print("\nTOP-5 FILE – last 5 rows:")
    print(top5_df.tail())

    # Pick a random strategy column to visualise
    strategy_col_top5 = random.choice(top5_df.columns.tolist())
    print(f"\nPlotting raw PnL for randomly chosen strategy from top-5: {strategy_col_top5}")

    plt.figure(figsize=(10, 6))
    plt.plot(top5_df.index, top5_df[strategy_col_top5])
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.title(f"Raw PnL Series for {strategy_col_top5}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- KMeans top-5 strategies file -------------------------------------------------

try:
    kmeans_top5_df = pd.read_pickle("top5_kmeans_strategies_daily_pnl.pkl")
except FileNotFoundError:
    print("\n(top5_kmeans_strategies_daily_pnl.pkl not found – skipping KMeans top-5 summary)\n")
else:
    # Print the first and last five rows
    print("\nKMEANS TOP-5 FILE – first 5 rows:")
    print(kmeans_top5_df.head())
    print("\nKMEANS TOP-5 FILE – last 5 rows:")
    print(kmeans_top5_df.tail())

    # Pick a random strategy column to visualise
    strategy_col_kmeans = random.choice(kmeans_top5_df.columns.tolist())
    print(f"\nPlotting raw PnL for randomly chosen strategy from KMeans top-5: {strategy_col_kmeans}")

    plt.figure(figsize=(10, 6))
    plt.plot(kmeans_top5_df.index, kmeans_top5_df[strategy_col_kmeans])
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.title(f"Raw PnL Series for {strategy_col_kmeans}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Hierarchical top-5 strategies file -------------------------------------------

try:
    hierarchy_top5_df = pd.read_pickle("top5_hierarchy_strategies_daily_pnl.pkl")
except FileNotFoundError:
    print("\n(top5_hierarchy_strategies_daily_pnl.pkl not found – skipping Hierarchical top-5 summary)\n")
else:
    # Print the first and last five rows
    print("\nHIERARCHICAL TOP-5 FILE – first 5 rows:")
    print(hierarchy_top5_df.head())
    print("\nHIERARCHICAL TOP-5 FILE – last 5 rows:")
    print(hierarchy_top5_df.tail())

    # Pick a random strategy column to visualise
    strategy_col_hierarchy = random.choice(hierarchy_top5_df.columns.tolist())
    print(f"\nPlotting raw PnL for randomly chosen strategy from Hierarchical top-5: {strategy_col_hierarchy}")

    plt.figure(figsize=(10, 6))
    plt.plot(hierarchy_top5_df.index, hierarchy_top5_df[strategy_col_hierarchy])
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.title(f"Raw PnL Series for {strategy_col_hierarchy}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Combine all loaded DataFrames into a single file ---------------------------

# Collect all DataFrames that were successfully loaded
combined_dfs = [pnl_df]
for optional_df_name in ["top5_df", "kmeans_top5_df", "hierarchy_top5_df"]:
    if optional_df_name in locals():
        combined_dfs.append(locals()[optional_df_name])

# Concatenate along columns, aligning on the index (dates)
combined_df = pd.concat(combined_dfs, axis=1)

# Drop any duplicate column names (may appear if the same strategy exists in
# several collections)
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

# Persist the combined DataFrame
combined_pickle_path = "Original_data_pnl.pkl"
combined_df.to_pickle(combined_pickle_path)
print(f"\nCombined DataFrame saved to '{combined_pickle_path}'.")

# Display a preview
print("\nCOMBINED FILE – first 5 rows:")
print(combined_df.head())
print("\nCOMBINED FILE – last 5 rows:")
print(combined_df.tail())

# Plot a randomly selected strategy from the combined data
strategy_col_combined = random.choice(combined_df.columns.tolist())
print(f"\nPlotting raw PnL for randomly chosen strategy from combined data: {strategy_col_combined}")

plt.figure(figsize=(10, 6))
plt.plot(combined_df.index, combined_df[strategy_col_combined])
plt.xlabel("Date")
plt.ylabel("Cumulative PnL")
plt.title(f"Raw PnL Series for {strategy_col_combined}")
plt.grid(True)
plt.tight_layout()
plt.show()
