import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    Normalizer,
)
from sklearn.model_selection import train_test_split

"""
1. import dataset
2. take care of missing data
3. encode categorical data
4. split train test
5. feature scaling
"""


def import_dataset(url):
    """
    Function to import the dataset from an url (lacaly on the computer)

    :param url:
        String for the file path

    :return: dataset

    :raise : Exception if dataset format isn't .csv or .data
    """
    # different file extension : .csv and .data
    file_extension = url.split(".")[1]
    # different treatment for csv and for .data
    if file_extension == "csv":
        dataset = pd.read_csv(url, sep=";")
    elif file_extension == "data":
        # load text file
        text = np.loadtxt(url)
        # turn text file as pandas dataframe
        dataset = pd.DataFrame(
            text,
            columns=[
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
                "MEDV",
            ],
        )
    else:
        raise Exception("This dataset extension cannot be use. Use .csv or .data file.")
    return dataset


# Normalize the data with standard scaling
###
def standard_scaling(dataset):
    """
    Function to scale the data set with a Standard Sklearn scaler

    :param dataset:
        Pandas dataframe

    :return: result
        the dataset scaled
    """

    scaler = StandardScaler()
    result = scaler.fit_transform(dataset)
    return result


###
# Normalize with min-max scaler
###
def min_max_scaling(dataset):
    """
     Function to scale the data set with a minMax Sklearn scaler

    :param dataset:
        Pandas dataframe

    :return: result
        the dataset sclaled
    """

    scaler = MinMaxScaler()
    result = scaler.fit_transform(dataset)
    return result


###
# Normalize with polynomial scaler
###
def polynomial_scaling(dataset):
    """
    Function to scale the data set with a polynomial Sklearn scaler

    :param dataset:
        Pandas dataframe

    :return: result
        the dataset scaled
    """

    scaler = PolynomialFeatures()
    result = scaler.fit_transform(dataset)
    return result


###
#
###
def normalize(dataset):
    """
    Function that normalize the dataset with sklearn normalizer
    :param dataset:
        Pandas dataframe

    :return: result
        the dataset normalized
    """

    scaler = Normalizer()
    result = scaler.fit_transform(dataset)
    return result


###
#
###
def preprocessing(dataset, n_split, n_norm):
    """
    Function that make all the preprocessing on a given dataset

    :param dataset:
        Pandas dataframe
    :param n_split:
        int : split number to choose the split technique : 1 : random_state = 30, 2 : random_state = 20, 3: random_state = 10
    :param n_norm:
        int : normalization number to choose the normalization technique : 1: StandardScaling, 2: MinMax,
                                                                          3: Polynomial, 4: Normalizer
    :return: x_train, x_test, y_train, y_test

    :raise : Exception if the entry for scaling isn't known (1,2,3,4)
    """
    data = dataset.dropna(axis="index")
    data = data.drop_duplicates()

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # split into train-test set
    np.random.seed(0)
    if n_split == 1:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.5, random_state=30
        )
    elif n_split == 2:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.5, random_state=20
        )
    elif n_split == 3:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.5, random_state=10
        )
    else:
        raise Exception("Nothing developed for this split entry. Choose 1,2 or 3 !")

    # normalize the values x_train, x_test
    if n_norm == 1:
        x_train = standard_scaling(x_train)
        x_test = standard_scaling(x_test)
    elif n_norm == 2:
        x_train = min_max_scaling(x_train)
        x_test = min_max_scaling(x_test)
    elif n_norm == 3:
        x_train = polynomial_scaling(x_train)
        x_test = polynomial_scaling(x_test)
    elif n_norm == 4:
        x_train = normalize(x_train)
        x_test = normalize(x_test)
    else:
        raise Exception("Nothing developed for this scaling entry. Choose 1,2,3 or 4!")

    return x_train, x_test, y_train, y_test
