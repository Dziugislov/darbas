# %% Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import sys
import io
from pathlib import Path
import base64

# Create output directories if they don't exist
Path('outputs').mkdir(exist_ok=True)
Path('outputs/figures').mkdir(exist_ok=True)

# Create a custom output capture class
class OutputCapture:
    def __init__(self):
        self.outputs = []
    
    def write(self, text):
        self.outputs.append(text)
        sys.__stdout__.write(text)  # Still print to console
    
    def flush(self):
        pass

    def get_output(self):
        return ''.join(self.outputs)

# Setup output capture
output_capture = OutputCapture()
sys.stdout = output_capture

# %% Basic text and calculations
print("Hello World!")
print("\nLet's do some quick math:")
print(f"3 + 5 = {3 + 5}")
print(f"Square root of 16 = {np.sqrt(16)}")
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Creating and displaying a simple array
arr = np.arange(1, 11)
print("NumPy array:", arr)
print("\nArray statistics:")
print(f"Mean: {arr.mean():.2f}")
print(f"Standard deviation: {arr.std():.2f}")
print(f"Max: {arr.max()}")
print(f"Min: {arr.min()}")

# %% Creating a simple line plot
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.title('Simple Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.legend()
plt.savefig('outputs/figures/sine_wave.png')
plt.close()

# %% Creating a more complex visualization with Seaborn
# Generate random data
np.random.seed(42)
data = {
    'x': np.random.normal(0, 1, 1000),
    'y': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
}
df = pd.DataFrame(data)

# Create a scatter plot with marginal distributions
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='x', y='y', hue='category', alpha=0.6)
plt.title('Scatter Plot with Categories')
plt.savefig('outputs/figures/scatter_plot.png')
plt.close()

# %% Creating a heatmap
# Generate correlation matrix
matrix = np.random.rand(5, 5)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, cmap='viridis', center=0.5)
plt.title('Random Correlation Matrix')
plt.savefig('outputs/figures/heatmap.png')
plt.close()

# %% Bar plot with error bars
categories = ['A', 'B', 'C', 'D']
values = np.random.rand(4) * 10
errors = np.random.rand(4)

plt.figure(figsize=(10, 6))
plt.bar(categories, values, yerr=errors, capsize=5)
plt.title('Bar Plot with Error Bars')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.savefig('outputs/figures/bar_plot.png')
plt.close()

# %% Generate HTML report with all outputs
def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

html_content = f"""
<html>
<head>
    <title>Python Output Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .output-section {{ margin-bottom: 30px; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Python Output Report</h1>
    
    <div class="output-section">
        <h2>Text Output</h2>
        <pre>{output_capture.get_output()}</pre>
    </div>

    <div class="output-section">
        <h2>Visualizations</h2>
        <h3>Sine Wave</h3>
        <img src="data:image/png;base64,{image_to_base64('outputs/figures/sine_wave.png')}" alt="Sine Wave">
        
        <h3>Scatter Plot</h3>
        <img src="data:image/png;base64,{image_to_base64('outputs/figures/scatter_plot.png')}" alt="Scatter Plot">
        
        <h3>Heatmap</h3>
        <img src="data:image/png;base64,{image_to_base64('outputs/figures/heatmap.png')}" alt="Heatmap">
        
        <h3>Bar Plot</h3>
        <img src="data:image/png;base64,{image_to_base64('outputs/figures/bar_plot.png')}" alt="Bar Plot">
    </div>
</body>
</html>
"""

with open('outputs/report.html', 'w') as f:
    f.write(html_content)

print("\nReport generated! Check 'outputs/report.html' for the complete report with text and figures.")

# %% Interactive elements (uncomment to use)
# import ipywidgets as widgets
# from IPython.display import display
# slider = widgets.IntSlider(min=0, max=100, value=50)
# display(slider) 
