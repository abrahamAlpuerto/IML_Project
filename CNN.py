import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision

use_cuda = torch.cuda.is_available()


# import dataset
images, labels = torch.load('preprocessed_dataset.pt')

# print(images.shape)
# print(labels)

# split data into train and test set using sklearn
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=473)


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
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.bn4 = nn.BatchNorm2d(128)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final fully connected layer
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024,3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)

        # print(x.shape)
        x = torch.flatten(x, -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

if use_cuda:
    device = torch.device("cuda")
    net = net.to(device=device)

test_accuracies = []

for epoch in range(50):  # loop over the dataset multiple times

    net.train()  # Set the model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
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
    
        # Calculate and store training loss and accuracy for the epoch

    # Testing phase
    net.eval()  # Set the model to evaluation mode
    correct_test = 0
    total_test = 0

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


    print(f'Epoch {epoch + 1}: Test Accuracy: {test_accuracy:.2f}%')

    if test_accuracy >= 95:
        break



print('Finished Training')
print("Best test accuracy: ",max(test_accuracies))
# save CNN
torch.save(net.state_dict(), 'models/adam50epcoch.pth')

classes = ('Ellie','Jessy','Tucker')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
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