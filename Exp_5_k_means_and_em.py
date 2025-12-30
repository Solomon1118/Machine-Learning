from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Step 2: Cluster the data using k-Means algorithm
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette = silhouette_score(X, kmeans_labels)

# Step 3: Cluster the data using EM algorithm (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)
gmm_silhouette = silhouette_score(X, gmm_labels)

# Step 4: Compare the clustering results and visualize them
print(f"Silhouette Score for k-Means: {kmeans_silhouette}")
print(f"Silhouette Score for EM (GMM): {gmm_silhouette}")

# Visualization of the clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# k-Means Clustering
ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
ax1.set_title('k-Means Clustering')

# EM (GMM) Clustering
ax2.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
ax2.set_title('EM (GMM) Clustering')

plt.show()

# Step 5: Comment on the quality of clustering
if kmeans_silhouette > gmm_silhouette:
    print("k-Means clustering provides better quality clusters according to the silhouette score.")
else:
    print("EM (GMM) clustering provides better quality clusters according to the silhouette score.")