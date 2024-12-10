import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
best_acc = 90

class FC(nn.Module):
    def __init__(self,input):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, 3)
        )
        
    def forward(self,x):
        return self.fc(x)
    

def train(model, trainloader, epoch, device, criterion):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)
        

def test(model,trainloader, testloader, device, epoch):
    global best_acc
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for data, label in trainloader:
            data, label = data.to(device), label.to(device)

            output = model(data)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        train_accuracy = 100 * correct / total
    total = 0
    correct = 0
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)

            output = model(data)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        test_accuracy = 100 * correct / total
        if epoch % 100 == 0:
            print(f"[Epoch({epoch} -> Train Accuracy: {train_accuracy}  Test Accuracy: {test_accuracy}]")
        if test_accuracy > best_acc:
            torch.save(model.state_dict(), 'models/fcnn.pth')
        return test_accuracy, train_accuracy


if __name__=="__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 2500
    lr = 5e-4

    images, labels = torch.load('pca_data_70pct.pt', weights_only='False')
    images = images.float()  
    labels = labels.long()  
    in_dim = images.shape[1]
    model = FC(input=in_dim)
    model = model.to(device)
    
    # print(images.shape)
    # print(labels.shape)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=401)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    # print(X_train)
    # print(y_train)
    trainloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    loss_history = []
    test_accuracy = []
    train_accuracy = []
    for i in range(1, epochs + 1):
        loss = train(model, trainloader, i, device, criterion)
        test_acc, train_acc = test(model, trainloader, testloader, device, i)

        loss_history.append(loss)
        test_accuracy.append(test_acc)
        train_accuracy.append(train_acc)
    print(max(test_accuracy))

    xsl = [x for x in range(len(loss_history))]
    xs = [x for x in range(epochs)]
    plt.plot(xsl, loss_history)
    plt.show() 
    plt.close()

    plt.plot(xs,test_accuracy, label='Test Accuracies')
    plt.plot(xs,train_accuracy, label='Train Accuracies')
    plt.legend()
    plt.show()
    plt.close()

        