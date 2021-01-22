import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

X = np.load('../data/mnist_data.npy')
y = np.load('../data/mnist_labels.npy')

# scaling...
scalar = StandardScaler()
X = scalar.fit_transform(X).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

train_score = []
test_score = []

# knn with k from 1 to 25 with a step size of 2
for i in range(1, 26, 2):
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_score.append(train_accuracy)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_score.append(test_accuracy)

k = np.arange(1, 26, 2)
plt.plot(k, train_score, label='train accuracy')
plt.plot(k, test_score, label='test accuracy')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.legend()
plt.title("Accuracy with k from 1 to 25")
plt.show()

