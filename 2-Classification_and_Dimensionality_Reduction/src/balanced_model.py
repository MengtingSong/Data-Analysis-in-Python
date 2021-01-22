import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import time

X = np.load('../data/mnist_data.npy')
y = np.load('../data/mnist_labels.npy')

# Perform PCA decomposition with 50 components
pca = PCA(n_components=50, whiten=True)
X_pca = pca.fit_transform(X)

# Split training and testing set
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)

# Randomly select 9000 samples for training
train_size = np.shape(X_train_pca)[0]
sample_index = random.sample(range(train_size), 9000)
X_train_sample = []
y_train_sample = []
for index in sample_index:
    X_train_sample.append(X_train_pca[index])
    y_train_sample.append(y_train[index])

start_time = time.process_time()
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_sample, y_train_sample)
y_train_pred = knn.predict(X_train_sample)
y_test_pred = knn.predict(X_test_pca)
elapsed_time_secs = time.process_time() - start_time

train_accuracy = accuracy_score(y_train_sample, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train Accuracy: " + "%.2f" % train_accuracy)
print("Test Accuracy: " + "%.2f" % test_accuracy)
print('Running Time: ' + "%.2f" % elapsed_time_secs)