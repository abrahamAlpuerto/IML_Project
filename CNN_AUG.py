import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

use_cuda = torch.cuda.is_available()
print("Cuda is: "  ,use_cuda)


# import dataset
images, labels = torch.load('preprocessed_dataset.pt', weights_only='False')

# print(images.shape)
# print(labels)

# split data into train and test set using sklearn
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=403)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transform if specified
        if self.transform:
            # Convert tensor to PIL Image for augmentation, then back to tensor
            image = transforms.ToPILImage()(image)  # Convert tensor to PIL Image
            image = self.transform(image)          # Apply transformations
        
        return image, label


# Apply data augmentation dynamically
train_dataset = AugmentedDataset(X_train, y_train, transform=train_transform)
test_dataset = AugmentedDataset(X_test, y_test, transform=test_transform)

# Create DataLoaders
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 5)
        self.bn5 = nn.BatchNorm2d(512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.8))  # Drop 50% of neurons
        self.fc2 = nn.Linear(1024,3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.global_avg_pool(x)

        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)

        return x

net = Net()
# hyper params
epochs = 250
lr = 0.001
momentum=0.9




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)


# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='max', factor=0.1, patience=10, verbose=True
# )

if use_cuda:
    device = torch.device("cuda")
    net = net.to(device=device)

loss_history = []
test_accuracies = []
train_accuracies = []
best_score = 95

for epoch in range(epochs):  # loop over the dataset multiple times

    net.train()  # Set the model to training mode
    running_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:  # Print every 5 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.10f}')
            running_loss = 0.0
        loss_history.append(loss.item())

    # Testing phase
    net.eval()  # Set the model to evaluation mode
    correct_test = 0
    total_test = 0
    correct_train = 0
    total_train = 0

    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

    # Calculate and store test accuracy for the epoch
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    with torch.no_grad():  # No need to track gradients for evaluation
        for data in testloader:
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # Calculate and store test accuracy for the epoch
    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)
    if test_accuracy > best_score:
        torch.save(net.state_dict(), 'models/sgdepcoch_best_AUG.pth')

    # scheduler.step(test_accuracy) # step scheduler 

    print(f'[Epoch {epoch + 1}: Train Accuracy: {train_accuracy:.2f}% Test Accuracy: {test_accuracy:.2f}%]')

    # if test_accuracy >= 95:
    #     break



print('Finished Training')
print("Best test accuracy: ",max(test_accuracies))

classes = ('Ellie','Jessy','Tucker')


# Collect predictions per class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        if use_cuda:
            images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        
        # print("Predictions:", predictions)
        # print("Labels:", labels)
        
        for label, prediction in zip(labels, predictions):
            label = label.item()
            prediction = prediction.item()
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# Print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

xsl = [x for x in range(len(loss_history))]
xs = [x for x in range(epochs)]
plt.plot(xsl, loss_history)
plt.show() 
plt.close()

plt.plot(xs,test_accuracies, label='Test Accuracies')
plt.plot(xs,train_accuracies, label='Train Accuracies')
plt.legend()
plt.show()
plt.close()
