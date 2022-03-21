from sklearn.linear_model import LinearRegression


def train (x_train, y_train):
    """
    Function to train an linear regression model
    :param x_train:
    :param y_train:
    :return: regressor
        sklearn regressor of the linear model
    """
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    return regressor

def predict (regressor, x_test) :
    """
    Function to make a prediction on a set of data

    :param regressor:

    :param x_test:

    :return: y_pred
        Predicted Value
    """
    y_pred = regressor.predict(x_test)
    return y_pred
