# %% [markdown]
# # Justinas Lopas Analysis
# This notebook demonstrates various data analysis and visualization techniques.

# %% Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# %% Basic text output and calculations
print("Hello! This is a demonstration of Jupyter-style cells in Python.")
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nLet's do some math:")
print(f"2 + 2 = {2 + 2}")
print(f"Square root of 16 = {np.sqrt(16)}")

# %% Creating and analyzing an array
arr = np.random.randint(1, 100, 10)
print("Random array:", arr)
print("\nArray statistics:")
print(f"Mean: {arr.mean():.2f}")
print(f"Standard deviation: {arr.std():.2f}")
print(f"Max: {arr.max()}")
print(f"Min: {arr.min()}")

# %% Creating a simple line plot
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)', color='blue')
plt.title('Simple Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.legend()
plt.show()

# %% Creating a scatter plot with random data
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
colors = np.random.rand(100)

plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=colors, alpha=0.6)
plt.title('Random Scatter Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.colorbar(label='Random Values')
plt.show()

# %% Creating a bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(10, 50, 5)

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='green')
plt.title('Random Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()

# %% Creating a heatmap
matrix = np.random.rand(5, 5)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, cmap='viridis', center=0.5)
plt.title('Random Correlation Matrix')
plt.show()

# %% Final summary
print("\nAnalysis Summary:")
print("1. Generated random data and performed basic statistics")
print("2. Created various visualizations:")
print("   - Line plot (sine wave)")
print("   - Scatter plot with random data")
print("   - Bar plot with categories")
print("   - Heatmap of random correlations") 