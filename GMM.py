import torch
from sklearn.mixture import GaussianMixture
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import numpy as np

# Load the PCA-transformed data and labels
pca_images, labels = torch.load("pca_data_60pct.pt")

# Convert `pca_images` to a NumPy array if needed (scikit-learn works with NumPy)
pca_images_np = pca_images.numpy()

# Apply GMM on the PCA-transformed data
n_clusters = 3  # Set this to the number of actual classes
gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
gmm.fit(pca_images_np)

# Predict cluster assignments
predicted_clusters = gmm.predict(pca_images_np)

# Convert labels to a NumPy array if they are in torch tensor format
true_labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

# Initialize an array to hold the mapped labels
mapped_labels = np.zeros_like(predicted_clusters)

# Loop through each cluster and assign the most common true label in each cluster
for cluster in range(n_clusters):
    mask = predicted_clusters == cluster
    most_common = mode(true_labels[mask])[0][0]
    mapped_labels[mask] = most_common

# Calculate and print the accuracy
accuracy = accuracy_score(true_labels, mapped_labels)
print(f'Clustering Accuracy: {accuracy * 100:.2f}%')
