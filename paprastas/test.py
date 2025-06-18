import pickle
with open("pnl_temp.pkl", "rb") as f:
    pnl_df = pickle.load(f)

print("\nPnL temp head:")
print(pnl_df.head())
print(pnl_df.columns)