import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
features_pca = pca.fit_transform(X_scaled)

# Calculate the covariance matrix
cov_matrix = np.cov(X_scaled.T)
print("Covariance matrix:\n", cov_matrix)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"\nEigenvalues:\n{eigenvalues}")
print(f"\nEigenvectors:\n{eigenvectors}")

# Create a DataFrame with principal components
pca_df = pd.DataFrame(data=features_pca, columns=['PC1', 'PC2'])
pca_df["target"] = target

# Plotting the PCA results
plt.figure(figsize=(8, 6))
for label, color in zip(iris.target_names, ["red", "green", "blue"]):
    plt.scatter(
        pca_df.loc[pca_df["target"] == list(iris.target_names).index(label), "PC1"],
        pca_df.loc[pca_df["target"] == list(iris.target_names).index(label), "PC2"],
        label=label,
        color=color,
        alpha=0.7
    )
plt.title("PCA on IRIS Dataset (4 features to 2)", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(title="Species")
plt.grid(True)
plt.tight_layout()
plt.show()

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("\nExplained variance by each principal component:")
print("PC1:", explained_variance[0])
print("PC2:", explained_variance[1])
print("Total variance retained:", sum(explained_variance))
