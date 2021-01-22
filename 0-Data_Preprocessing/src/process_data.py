import numpy as np
np.set_printoptions(suppress=True)


def import_data(filename):
    # read input data and store in arrays of X and y
    data = np.genfromtxt(filename, delimiter=',')
    X = []
    y = []
    for item in data:
        X.append(item[:279])
        y.append(item[279])
    return X, y


# replace missing feature value (NaN) with the median of that feature
def impute_missing(X):
    X_imputed = np.copy(X)
    for x_1 in X_imputed:
        index = 0
        for num in x_1:
            if np.isnan(num):
                rest = []
                for x_2 in X_imputed:
                    if not np.isnan(x_2[index]):
                        rest.append(x_2[index])
                rest.sort()
                # replace with the median
                x_1[index] = rest[int(np.rint(len(rest)/2))]
            index = index + 1
    return X_imputed


# remove entry containing NaN
def discard_missing(X):
    X_discarded = np.copy(X)
    index = 0
    nan_index = []
    for x in X_discarded:
        if np.isnan(x).any():
            nan_index.append(index)
        index = index + 1
    X_discarded = np.delete(X_discarded, nan_index, 0)
    return X_discarded


# shuffle the order of entries
def shuffle_data(X):
    X_shuffled = np.copy(X)
    np.random.shuffle(X_shuffled)
    return X_shuffled


# calculate the mean fo each feature:
def calculate_mean(X):
    num = len(X)
    avg = np.zeros(279)
    for x in X:
        for index in range(279):
            avg[index] = avg[index] + x[index]
    for index in range(279):
        avg[index] = avg[index] / num
    return avg


# calculate the standard deviation of each feature
def calculate_std_deviation(X):
    num = len(X)
    # calculate the average value of each feature and store in an array
    avg = calculate_mean(X)
    # calculate the std deviation
    std_deviation = np.zeros(279)
    for x in X:
        for index in range(279):
            std_deviation[index] = std_deviation[index] + np.square(x[index] - avg[index])
    for index in range(279):
        std_deviation[index] = np.sqrt(std_deviation[index] / (num - 1))
    return std_deviation


# remove entries that contain a value two std deviation away from the mean
def rm_entry_two_std(X, avg, std_deviation):
    X_rm_two_std = np.copy(X)
    # get the index of entries which contain such a value
    entry_index = 0
    entry_index_array = []
    for x in X_rm_two_std:
        for index in range(279):
            if x[index] - avg[index] > std_deviation[index] * 2:
                entry_index_array.append(entry_index)
            break
        entry_index = entry_index + 1
    # remove those entries
    X_rm_two_std = np.delete(X_rm_two_std, entry_index_array, 0)
    return X_rm_two_std


# standardize data
def std_data(X, avg, std_deviation):
    X_std = np.copy(X)
    for x in X_std:
        for index in range(279):
            x[index] = (x[index] - avg[index]) / std_deviation[index]
    return X_std






