import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

# use the training set to find the optimal model
trainingSet = pd.read_csv("./data/trainingSet.csv")
y = trainingSet['Score'].to_numpy()
y = y.astype(np.float)
X = trainingSet.drop(columns=['Score']).to_numpy()
X = X.astype(np.float)


def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    pred_time = time() - t0
    print("predict time:  %0.3fs" % pred_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    clf_descr = str(clf).split('(')[0]
    print (classification_report(y_test, pred))
    return clf_descr, score, train_time, pred_time


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(max_iter=50, n_jobs=-1), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, X_train, y_train, X_test, y_test))

print(results)

# make prediction on the test set using the winner
model = RandomForestClassifier(n_estimators=100).fit(X, y)
predictionSet = pd.read_csv("./data/predictionSet.csv")
X_pred = predictionSet.drop(columns=['Id', 'Score']).to_numpy()
X_pred = X_pred.astype(np.float)
y_pred = model.predict(X_pred)

predictionSet['Score'] = y_pred
submission = predictionSet[['Id', 'Score']]
print(submission.head())
submission.to_csv("./data/submission.csv", index=False)
