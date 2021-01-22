import numpy as np
from sklearn.model_selection import train_test_split


# split data into train and test set with the test set fraction of t_f
def split_train_test(X, y, t_f):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_f)
    return X_train, y_train, X_test, y_test


# split data into train and test set with test set fraction of t_f and cross_validation set of cv_f
def split_train_test_CV(X, y, t_f, cv_f):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_f+cv_f)
    X_test, X_cv, y_test, y_cv = train_test_split(X_test, y_test, test_size=cv_f/(t_f+cv_f))
    return X_train, y_train, X_test, y_test, X_cv, y_cv
