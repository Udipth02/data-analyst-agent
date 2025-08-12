import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from scipy.cluster.hierarchy import linkage
import numpy as np

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00488/Live_20210128.csv"
df = pd.read_csv(url)

# Q1: Shape of the data
shape_data = df.shape

# Q2: Number of features containing null values
null_counts = df.isnull().sum()
num_features_with_null = (null_counts > 0).sum()

# Q3: Number of unique values in 'status_type'
num_unique_status_type = df['status_type'].nunique()

# Drop features with null values
df = df.dropna(axis=1)

# Save target variable
target = df['status_type']

# Drop specified features
X = df.drop(['status_id', 'status_type', 'status_published'], axis=1)

# Encode target
le = LabelEncoder()
y = le.fit_transform(target)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Q4: KMeans clustering at k=2
kmeans_k2 = KMeans(n_clusters=2, random_state=10)
kmeans_k2.fit(X_scaled)
inertia_k2 = kmeans_k2.inertia_

# Q5: KMeans clustering at k=4
kmeans_k4 = KMeans(n_clusters=4, random_state=10)
labels_k4 = kmeans_k4.fit_predict(X_scaled)

# Map clusters to true labels
label_map = {}
for cluster in range(4):
    mask = labels_k4 == cluster
    if np.any(mask):
        label_map[cluster] = mode(y[mask])[0][0]
predicted_labels_k4 = np.vectorize(label_map.get)(labels_k4)
accuracy_k4 = accuracy_score(y, predicted_labels_k4)
correct_preds_k4 = (predicted_labels_k4 == y).sum()

# Q6, Q7, Q8: Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(X_scaled)
first_row_label = agglo_labels[0]

# Hierarchical leaves
Z = linkage(X_scaled, method='ward')
num_leaves = X_scaled.shape[0]

# Map clusters to true labels
agglo_label_map = {}
for cluster in range(4):
    mask = agglo_labels == cluster
    if np.any(mask):
        agglo_label_map[cluster] = mode(y[mask])[0][0]
predicted_labels_agglo = np.vectorize(agglo_label_map.get)(agglo_labels)
accuracy_agglo = accuracy_score(y, predicted_labels_agglo) * 100

# Print all answers
print("Q1 Shape of data:", shape_data)
print("Q2 Features with Null values:", num_features_with_null)
print("Q3 Unique values in 'status_type':", num_unique_status_type)
print("Q4 Inertia at k=2:", inertia_k2)
print("Q5 Correct predictions at k=4:", correct_preds_k4)
print("Q6 Label predicted for first row:", first_row_label)
print("Q7 Number of leaves:", num_leaves)
print("Q8 Accuracy (%) of Agglomerative Clustering:", accuracy_agglo)
