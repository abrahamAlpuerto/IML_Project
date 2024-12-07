import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch


images, labels = torch.load('preprocessed_dataset.pt',weights_only=False)

n_samples, rgb, h, w = images.shape

reshaped_images = images.reshape(n_samples,-1)

# Apply PCA
pca = PCA(n_components=0.65)
pca_images = pca.fit_transform(reshaped_images)
# Print results
n_components = pca.n_components_
print(f'Number of components for 65% variance: {n_components}')
print(f'Original dimension: {rgb * h * w}')
print(f'Reduced dimension: {n_components}')

pca_images = torch.tensor(pca_images)

torch.save((pca_images, labels), "pca_data_65pct.pt")
