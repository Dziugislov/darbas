# %% [markdown]
# # Data Analysis and Visualization Demo
# This notebook-style script demonstrates various Python capabilities including:
# * Random data generation
# * Data visualization
# * Statistical analysis
# * Console animations

# %% [markdown]
# ## Setup
# First, let's import all necessary libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import randint, choice
import time

# %% [markdown]
# ## Fun Facts Section
# Let's start with some random Python fun facts!

# %%
fun_facts = [
    "Python was named after Monty Python, not the snake!",
    "The first version of Python was released in 1991",
    "Python uses indentation for code blocks instead of curly braces",
    "Python has a zen philosophy - import this",
]

print("\n=== Random Fun Facts ===")
print(choice(fun_facts))

# %% [markdown]
# ## Random Data Generation
# Generate and visualize some random numbers

# %%
print("\n=== Random Number Generation ===")
random_numbers = [randint(1, 100) for _ in range(10)]
print(f"Random numbers: {random_numbers}")

# %%
# Create a simple line plot
plt.figure(figsize=(10, 6))
plt.plot(random_numbers, marker='o')
plt.title('Random Numbers Line Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('line_plot.png')
plt.close()

# %% [markdown]
# ## Advanced Data Visualization
# Create a more complex dataset and visualize it using different techniques

# %%
print("\n=== Creating Sample Dataset ===")
np.random.seed(42)
data = {
    'x': np.random.normal(0, 1, 1000),
    'y': np.random.normal(0, 1, 1000),
    'category': [choice(['A', 'B', 'C']) for _ in range(1000)]
}
df = pd.DataFrame(data)

# %%
# Create a scatter plot with different categories
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y', hue='category', alpha=0.6)
plt.title('Scatter Plot with Categories')
plt.savefig('scatter_plot.png')
plt.close()

# %%
# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='x', bins=30, kde=True)
plt.title('Distribution of X Values')
plt.savefig('histogram.png')
plt.close()

# %% [markdown]
# ## Statistical Analysis
# Let's look at some basic statistics of our dataset

# %%
print("\n=== Data Statistics ===")
print(df.describe())

# %% [markdown]
# ## Fun Animation
# Finally, let's create a simple loading animation in the console

# %%
print("\n=== Loading Animation ===")
for _ in range(10):
    for char in '|/-\\':
        print(f'\rProcessing {char}', end='')
        time.sleep(0.1)

print("\n\nAll visualizations have been saved!") 