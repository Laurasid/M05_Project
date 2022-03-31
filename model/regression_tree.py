from sklearn.tree import DecisionTreeRegressor


###
# train the model
###
def train(x_train, y_train):
    """
    Function to train a regression tree model
    :param array-like x_train: train values
    :param array-like y_train: train labels
    :return: DecisionTreeRegressor regressor: sklearn regressor of the linear model
    """
    regressor = DecisionTreeRegressor()
    regressor.fit(x_train, y_train)

    return regressor


###
# predict likely values
###
def predict(regressor, x_test):
    """
    Function to train a regression tree model
    :param DecisionTreeRegressor regressor: regressor
    :param array-like x_test: test values
    :return: array-like y_pred: predicted values
    """
    y_pred = regressor.predict(x_test)

    return y_pred
