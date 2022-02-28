from sklearn.linear_model import LinearRegression
from sklearn import metrics

def train (x_train, y_train):
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    return regressor

def predict (regressor, x_test) :
    y_pred = regressor.predict(x_test)
    return y_pred

def MAE(y_test, y_pred):
    return metrics.mean_absolute_error(y_test,y_pred)