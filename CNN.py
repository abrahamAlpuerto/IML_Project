import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# import dataset
images, labels = torch.load('preprocessed_dataset.pt')

# print(images.shape)
# print(labels)

# split data into train and test set using sklearn
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=244)


# Wrap data in TensorDataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
# print(X_train)
# print(y_train)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers with increasing filters
        self.conv1 = nn.Conv2d(3, 16, 5)  # Increase filters for better feature learning
        self.pool = nn.MaxPool2d(2, 2)  # Use 2x2 max pooling
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(64,128,5)

        # Calculate input size for fully connected layer based on feature map size after conv3
        # Assuming input is 224x224, output size here would be smaller, so adjust accordingly
        # This example assumes the final feature map after conv3 is approximately 53x53
        self.fc1 = nn.Linear(128 * 45 * 45, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # Output layer for 3 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(f'After conv1: {x.shape}')
        x = self.pool2(F.relu(self.conv2(x)))
        # print(f'After conv2: {x.shape}')
        x = F.relu(self.conv3(x))
        # print(f'After conv3: {x.shape}')
        x = F.relu(self.conv4(x))
        # print(f'After conv4: {x.shape}')
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

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

print('Finished Training')

# save CNN
torch.save(net.state_dict(), 'models/v1.pth')

classes = ('Ellie','Jessy','Tucker')


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on test set: {100 * correct // total} %')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')