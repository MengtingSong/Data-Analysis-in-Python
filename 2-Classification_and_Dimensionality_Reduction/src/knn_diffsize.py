import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
import random
from sklearn.decomposition import PCA

X = np.load('../data/mnist_data.npy')
y = np.load('../data/mnist_labels.npy')

# Run kNN on different size of training set
scalar = StandardScaler()
X_kNN = scalar.fit_transform(X).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X_kNN, y, test_size=0.2, random_state=0)
train_size = np.shape(X_train)[0]

train_score_size = []
test_score_size = []
running_time_size = []

for size in np.arange(1, round(train_size/3000)):
    X_train_sample = []
    y_train_sample = []
    sample_index = random.sample(range(train_size), size * 3000)
    for index in sample_index:
        X_train_sample.append(X_train[index])
        y_train_sample.append(y_train[index])

    start_time = time.process_time()
    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_sample, y_train_sample)
    y_train_pred = knn.predict(X_train_sample)
    y_test_pred = knn.predict(X_test)
    elapsed_time_secs = time.process_time() - start_time

    train_accuracy = accuracy_score(y_train_sample, y_train_pred)
    train_score_size.append(train_accuracy)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_score_size.append(test_accuracy)

    running_time_size.append(elapsed_time_secs)

# Run kNN with different component number
running_time_pca = []

for n in np.arange(50, 751, 100):
    pca = PCA(n_components=n, whiten=True)
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)

    start_time = time.process_time()
    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    elapsed_time_secs = time.process_time() - start_time
    print('time: ' + "%.2f" % elapsed_time_secs)

    running_time_pca.append(elapsed_time_secs)

size = np.arange(3000, train_size, 3000)

plt.subplot(121)
plt.plot(size, running_time_size)
plt.xlabel('train set size')
plt.ylabel('running time in seconds')
plt.title("Running Time with Different Training Set Size")

plt.subplot(122)
n = np.arange(50, 751, 100)
plt.plot(n, running_time_pca)
plt.xlabel('Component Number')
plt.ylabel('running time in seconds')
plt.title("Running Time with Different Component Number")

plt.show()

plt.plot(size, train_score_size, label='train accuracy')
plt.plot(size, test_score_size, label='test accuracy')
plt.xlabel('train set size')
plt.ylabel('accuracy')
plt.legend()
plt.title("Accuracy with Different Train Set Size")

plt.show()
