import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.decomposition import PCA

# Download MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Normalize pixel values to a range [0-1]
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Plot 20 random images in a grid of 4 rows and 5 columns
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    rand_idx = np.random.randint(0, X_train.shape[0])
    ax.imshow(X_train[rand_idx].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.savefig('random_mnist_images.png')
plt.close()

# Apply K-means clustering to the training partition using k=10
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)

# Use the learned centroids to predict cluster memberships for the test data
test_clusters = kmeans.predict(X_test)

# Evaluate the clustering performance on the test partition
homogeneity = homogeneity_score(y_test.astype(int), test_clusters)
print(f"Homogeneity score without PCA: {homogeneity:.4f}")

# Apply PCA on the training data
pca = PCA()
pca.fit(X_train)

# Project the test data using different numbers of eigenvectors
n_components_list = [16, 32, 64, 128, 256, 512]
homogeneity_scores = []

for n_components in n_components_list:
    # Project data
    X_train_pca = pca.transform(X_train)[:, :n_components]
    X_test_pca = pca.transform(X_test)[:, :n_components]
    
    # Apply K-means
    kmeans_pca = KMeans(n_clusters=10, random_state=42)
    kmeans_pca.fit(X_train_pca)
    
    # Predict and evaluate
    test_clusters_pca = kmeans_pca.predict(X_test_pca)
    homogeneity_pca = homogeneity_score(y_test.astype(int), test_clusters_pca)
    homogeneity_scores.append(homogeneity_pca)
    
    print(f"Homogeneity score with {n_components} PCs: {homogeneity_pca:.4f}")

# Plot number of principal components vs clustering performance
plt.figure(figsize=(10, 6))
plt.plot(n_components_list, homogeneity_scores, marker='o')
plt.axhline(y=homogeneity, color='r', linestyle='--', label='Without PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Homogeneity Score')
plt.title('Clustering Performance vs Number of Principal Components')
plt.legend()
plt.savefig('pca_performance.png')
plt.close()

# Plot cumulative explained variance
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance vs Number of Principal Components')
plt.savefig('cumulative_variance.png')
plt.close()

print("Analysis complete. Check the generated plots for visualization of results.")