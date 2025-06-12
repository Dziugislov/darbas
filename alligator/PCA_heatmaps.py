import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Generate synthetic data
np.random.seed(42)
n = 200
data = pd.DataFrame({
    'vol': np.random.normal(loc=1.0, scale=0.3, size=n),
    'momentum': np.random.normal(loc=0.5, scale=0.2, size=n),
    'count': np.random.randint(50, 200, size=n),
    'shrp': np.random.uniform(0.1, 1.0, size=n)  # Sharpe ratio
})

# Filter only strategies with Sharpe ratio > 0.2
df = data[data['shrp'] > 0.2]

# 2. PCA analysis
features = ['vol', 'momentum', 'count']
X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# 3. PCA scatter plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=df['shrp'], cmap='viridis', s=20)
plt.colorbar(sc, label='Sharpe Ratio')
plt.title('PCA Projection (2D) with Sharpe as Color')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# 4. PCA Loadings Plot
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
plt.figure(figsize=(8, 8))
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
for feature in loadings.index:
    x = loadings.loc[feature, 'PC1']
    y = loadings.loc[feature, 'PC2']
    plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    plt.text(x*1.2, y*1.2, feature, color='red', ha='center', va='center')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Loadings Plot (Variable Influence)')
plt.grid(True)
plt.show()
