import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


pca_images, labels = torch.load("pca_data_90pct.pt",weights_only=False)


X_train, X_test, y_train, y_test = train_test_split(pca_images, labels, test_size=0.1, random_state=473)

for i in range(1,5):
    
    neigh = KNeighborsClassifier(n_neighbors=i)

    neigh.fit(X_train,y_train)

    score = neigh.score(X_test,y_test)

    print(f"Accuracy with n = {i} neighbors: {score}")