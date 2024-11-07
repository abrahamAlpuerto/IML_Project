import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# transformation to tensors

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# Testing
# image_path = "Ellie/image.png"
# image = Image.open(image_path).convert('RGB')
# image_tensor = transform(image)
# print(image_tensor.shape)

# load dataset with ImageFolder
dataset = datasets.ImageFolder(root='Dogs', transform=transform)

# dataloader to put everything into memory
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
images, labels = next(iter(dataloader))  # loads all images and labels in one

# save data as .pt
torch.save((images, labels), 'preprocessed_dataset.pt')
print("Dataset saved as preprocessed_dataset.pt")