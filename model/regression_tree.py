from sklearn.tree import DecisionTreeRegressor

"""
Function : regression_tree::train
Param : x_train : values of train dataset
        y_train : labels of train dataset
Return : regressor : DecisionTreeRegressor object
"""
def train(x_train, y_train):
    regressor = DecisionTreeRegressor()
    regressor.fit(x_train, y_train)

    return regressor


"""
Function : regression_tree::predict
Param : regressor : DecisionTreeRegressor object returned by train method
        x_test : values of test dataset
Return : y_pred : labels predicted for the values x_test
"""
def predict(regressor, x_test):
    y_pred = regressor.predict(x_test)

    return y_pred
