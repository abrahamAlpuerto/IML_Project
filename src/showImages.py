import matplotlib.pyplot as plt
import numpy as np
import torch
# Function to unnormalize and display an image
def imshow(image_tensor, title=None):
    # Un-normalize
    image = image_tensor.numpy().transpose((1, 2, 0))  # Convert to HWC format
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)  # Clip values to [0, 1] range for display
    
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

images, labels = torch.load('preprocessed_dataset.pt', weights_only='False')

# Visualize a few images
for i in range(5):  # Display first 5 images
    imshow(images[i], title=f"Label: {labels[i]}")
