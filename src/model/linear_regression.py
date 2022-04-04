from sklearn.linear_model import LinearRegression

"""
This module is used to train a model based on a linear regression method
"""

def train(x_train, y_train):
    """
    Train a linear regression-type model

    :param array-like x_train:
        values from the train dataset
    :param array-like y_train:
        labels that correspond to the values from the train dataset

    :return: (LinearRegression) -
        sklearn regressor of the linear model
    """
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    return regressor


def predict(regressor, x_test):
    """
    Predict labels according to the test values x_test

    :param array-like x_test:
        values from the test dataset
    :param LinearRegression regressor:
        a linear model regressor. Typically, the one returned by the train method

    :return: (DataFrame) -
        the labels predicted for the test values
    """
    y_pred = regressor.predict(x_test)
    return y_pred
