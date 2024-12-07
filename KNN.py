import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

pca_images, labels = torch.load("pca_data_60pct.pt",weights_only=False)

X_train, X_test, y_train, y_test = train_test_split(pca_images, labels, test_size=0.1, random_state=401)

for i in range(1,5):
    
    neigh = KNeighborsClassifier(n_neighbors=i)

    neigh.fit(X_train,y_train)

    score = neigh.score(X_test,y_test)

    print(f"Accuracy with n = {i} neighbors: {score}")


#confusion matrix
neigh = KNeighborsClassifier(n_neighbors=1)

neigh.fit(X_train,y_train)
pred = neigh.predict(X_test)

import numpy as np

# Ensure y_test is a numpy array
y_test_np = y_test.numpy()

# Optionally decode numerical labels to strings if needed
label_mapping = {0: "Ellie", 1: "Tucker", 2: "Jessie"}
y_test_decoded = [label_mapping[label] for label in y_test_np]
pred_decoded = [label_mapping[label] for label in pred]

# Dynamically determine labels
unique_labels = sorted(set(y_test_decoded))

# Compute and display confusion matrix
cm = confusion_matrix(y_test_decoded, pred_decoded, labels=unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()