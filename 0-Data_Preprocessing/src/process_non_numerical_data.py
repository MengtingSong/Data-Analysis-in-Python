import pandas as pd
import numpy as np


def import_data(filename):
    data = pd.read_csv(filename)
    # drop non-numerical columns
    data = data.drop(columns=['Name', 'Ticket', 'Cabin'])
    # replace gender and embarked columns with numerical values
    data.Sex.replace(['male', 'female'], [1, 0], inplace=True)
    data.Embarked.replace(['C', 'Q', 'S'], [0, 1, 2], inplace=True)
    y = data['Survived'].to_numpy()
    X = data.drop(['Survived'], axis=1).to_numpy()
    return X, y
