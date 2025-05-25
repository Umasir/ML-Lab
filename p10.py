# Import Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
data = pd.read_csv(r"C:\Users\Akhil\Downloads\ML6thSEM_FDP\Datasets\Wisconsin Breast Cancer Data.csv")

# Display Basic Info
print("Shape of data:", data.shape)
print("\nUnique values in diagnosis column:", data['diagnosis'].unique())
print("\nNull values:\n", data.isnull().sum())
print("\nDuplicate entries:", data.duplicated().sum())

# Drop unnecessary columns
df = data.drop(['id', 'Unnamed: 32'], axis=1)

# Encode Diagnosis column (M:1, B:0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Descriptive Statistics
print("\nDescriptive statistics:\n", df.describe().T)

# Drop diagnosis (target) for unsupervised learning
df_features = df.drop(columns=['diagnosis'])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# PCA - Reduce to 2 Dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained Variance
explained_variance = pca.explained_variance_ratio_
print(f"\nVariance explained by PC1: {explained_variance[0]:.4f}")
print(f"Variance explained by PC2: {explained_variance[1]:.4f}")
print(f"Total variance explained by first 2 components: {np.sum(explained_variance):.4f}")

# Elbow Method to Find Optimal k
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method to Determine Optimal k")
plt.grid(True)
plt.show()

# Apply K-Means Clustering with k=2
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# Visualize Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6, label="Data Points")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids', marker='X')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering (k=2) after PCA")
plt.legend()
plt.grid(True)
plt.show()
