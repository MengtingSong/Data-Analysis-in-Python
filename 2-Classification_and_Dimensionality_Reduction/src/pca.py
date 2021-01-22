import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X = np.load('../data/mnist_data.npy')
y = np.load('../data/mnist_labels.npy')

# Perform PCA decomposition
pca = PCA()
pca.fit(X)

# Plot CDF of the explained variance
components = np.arange(1, pca.n_components_ + 1)
cdf = []
initial_cdf = 0
var_ratios = pca.explained_variance_ratio_
for n in np.arange(0, pca.n_components_):
    initial_cdf = initial_cdf + var_ratios[n]
    cdf.append(initial_cdf)

plt.plot(components, cdf)
plt.xlabel('Components')
plt.ylabel('Variance CDF')
plt.title('CDF of the Explained Variance  ')
plt.show()

# Define a pipeline to search for the best combination of PCA truncation
knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('pca', pca), ('knn', knn)])

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': [45, 50, 55],
    'knn__n_neighbors': [1, 3, 5]
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# Choose n_components to be 50 and run kNN with k equals to 1 on decomposed dataset
pca = PCA(n_components=50, whiten=True)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_pca, y_train)
y_train_pred = knn.predict(X_train_pca)
y_test_pred = knn.predict(X_test_pca)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train Accuracy: " + "%.2f" % train_accuracy)
print("Test Accuracy: " + "%.2f" % test_accuracy)

