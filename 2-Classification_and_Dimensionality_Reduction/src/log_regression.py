import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = np.load('../data/mnist_data.npy')
y = np.load('../data/mnist_labels.npy')

# scaling...
scalar = StandardScaler()
X = scalar.fit_transform(X).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LogisticRegression(random_state=0, multi_class='auto', max_iter=10000).fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train Accuracy: " + "%.2f" % train_accuracy)
print("Test Accuracy: " + "%.2f" % test_accuracy)
