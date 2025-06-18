import pickle
import matplotlib.pyplot as plt

with open("pnl_temp.pkl", "rb") as f:
    pnl_df = pickle.load(f)

print("\nPnL temp tail:")
print(pnl_df.tail())
print(pnl_df.columns)
