import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Step 1: Generate dataset

np.random.seed(42)
values = np.random.rand(100)

print(values)

labels = []

for i in values[:50]:
    if i <=0.5:
        labels.append('Class1')
    else:
        labels.append('Class2')

print(labels)

labels += [None] * 50

print(labels)

data = {
    "Point": [f"x{i+1}" for i in range(100)],
    "Value": values,
    "Label": labels
}

df = pd.DataFrame(data)

df.head(10)

# Table of Meaning of Each Variable
variable_meaning = {
    "Point": "The point number",
    "Value": "The value of the point",
    "Label": "The class of the point"
}

variable_df = pd.DataFrame(list(variable_meaning.items()), columns=["Feature", "Description"])
print("\nVariable Meaning Table:")
print(variable_df)

df.nunique()

df.shape

print("\nBasic Information about Dataset:")
df.info()

print("\nSummary Statistics:")
df.describe().T

# Check for missing values

print("\nMissing Values in Each Column:")
df.isnull().sum()

# Get numeric columns
num_col = df.select_dtypes(include=['int', 'float']).columns

# Histograms for distribution of features
df[num_col].hist(figsize=(12, 8), bins=30, edgecolor='black')

# Title and labels
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Split data into labeled and unlabeled
labeled_df = df[df["Label"].notna()]
X_train = labeled_df[["Value"]]
y_train = labeled_df["Label"]

unlabeled_df = df[df["Label"].isna()]
X_test = unlabeled_df[["Value"]]

# Generate true labels for testing (for accuracy calculation)
true_labels = ["Class1" if x <= 0.5 else "Class2" for x in values[50:]]

# Step 2: Perform KNN classification for different values of k
k_values = [1, 2, 3, 4, 5, 20, 30]
results = {}
accuracies = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    results[k] = predictions
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions) * 100
    accuracies[k] = accuracy
    print(f"Accuracy for k={k}: {accuracy:.2f}%")
    
    # Assign predictions back to the DataFrame for this k
    unlabeled_df[f"Label_k{k}"] = predictions
    
print(predictions)

df1 = unlabeled_df.drop(columns=['Label'], axis=1)
df1

# Display accuracies
print("\nAccuracies for different k values:")
for k, acc in accuracies.items():
    print(f"k={k}: {acc:.2f}%")
