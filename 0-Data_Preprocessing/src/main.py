import process_data
import process_non_numerical_data


def main():
    X, y = process_data.import_data("arrhythmia.data")
    # replace nan with median
    X_imputed = process_data.impute_missing(X)

    # discard element including nan
    X_discarded = process_data.discard_missing(X)

    # shuffle entries
    X_shuffled = process_data.shuffle_data(X)

    # calculate mean of each feature
    avg = process_data.calculate_mean(X_imputed)

    # calculate std deviation of each feature
    std_deviation = process_data.calculate_std_deviation(X_imputed)

    # remove entries containing value two std deviation away from the mean
    X_rm_two_std = process_data.rm_entry_two_std(X_imputed, avg, std_deviation)

    # standardize data
    X_std = process_data.std_data(X_imputed, avg, std_deviation)

    # process data with non-numerical feature
    X_non_num, y_non_num = process_non_numerical_data.import_data("titanic/train.csv")

    return


if __name__ == "__main__":
    main()
