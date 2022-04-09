import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    Normalizer,
)
from sklearn.model_selection import train_test_split

"""
This module is used to preprocess the data before they can be used for the model training.
"""

def import_dataset(url):
    """
    Import the dataset from a path (locally on the computer).
    The file extension must be .csv or .data

    :param string url:
        the path where to find the dataset.

    :return: (DataFrame) -
        the dataset contained in the URL. Pandas DataFrame object.

    :raise Exception:
        the extension of the file isn't supported

    .. warning:: The files from which we want to extract datasets must be .csv or .data
    """
    # different file extension : .csv and .data
    file_extension = url.split(".")[1]
    filename, file_extension = os.path.splitext(url)
    print(file_extension)

    # different treatment for csv and for .data
    if file_extension == ".csv":
        dataset = pd.read_csv(url, sep=";")
    elif file_extension == ".data":
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
    #else:
        #raise Exception("This dataset extension cannot be use. Use .csv or .data file.")
    return dataset



def standard_scaling(dataset):
    """
    Scale the dataset with a standard Sklearn scaler (StandardScaler object)

    :param DataFrame dataset:
        the dataset to scale

    :return: (DataFrame) -
        the scaled dataset
    """

    scaler = StandardScaler()
    result = scaler.fit_transform(dataset)
    return result



def min_max_scaling(dataset):
    """
    Scale the dataset with a min-max Sklearn scaler (MinMaxScaler object)

    :param DataFrame dataset:
        the dataset to scale

    :return: (DataFrame) -
        the scaled dataset
    """

    scaler = MinMaxScaler()
    result = scaler.fit_transform(dataset)
    return result


def polynomial_scaling(dataset):
    """
    Scale the dataset with a polynomial Sklearn scaler (PolynomialFeatures object)

    :param DataFrame dataset:
        the dataset to scale

    :return: (DataFrame) -
        the scaled dataset
    """

    scaler = PolynomialFeatures()
    result = scaler.fit_transform(dataset)
    return result


def normalize(dataset):
    """
    Normalize the dataset with sklearn normalizer (Normalizer object)

    :param DataFrame dataset:
        the dataset to normalize

    :return: (DataFrame) -
        the normalized dataset
    """

    scaler = Normalizer()
    result = scaler.fit_transform(dataset)
    return result


def preprocessing(dataset, n_split, n_norm):
    """
    Preprocess the given dataset.

    :param DataFrame dataset:
        the dataset to preprocess
    :param int n_split:
        determine the split technique.\n
        n_split = 1 : random_state = 30 \n
        n_split = 2 : random_state = 20 \n
        n_split = 3 : random_state = 10 \n
    :param int n_norm:
        determine the normalization technique.\n
        n_norm = 1 : Standard scaling \n
        n_norm = 2 : MinMax scaling \n
        n_norm = 3 : Polynomial scaling \n
        n_norm = 4 : Normalizer \n

    :return: (array-like) -
        four array-like that correspond to train values and labels and test values and labels.
        These array-like can be directly used in model modules.

    :raise Exception:
        the entry for splitting or scaling isn't known
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
