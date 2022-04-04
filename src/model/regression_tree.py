from sklearn.tree import DecisionTreeRegressor



def train(x_train, y_train):
    """
    Train a regression tree-type model

    :param array-like x_train:
        values from the train dataset
    :param array-like y_train:
        labels that correspond to the values from the train dataset

    :return (DecisionTreeRegressor) -
        sklearn regressor of the model

    """
    regressor = DecisionTreeRegressor()
    regressor.fit(x_train, y_train)

    return regressor


def predict(regressor, x_test):
    """
    Predict labels according to the test values x_test

    :param DecisionTreeRegressor regressor:
        a decision tree model regressor. Typically, the one returned by the train method
    :param array-like x_test:
        values from the test dataset

    :return: (array-like) -
        the labels predicted for the test values

    """
    y_pred = regressor.predict(x_test)

    return y_pred
