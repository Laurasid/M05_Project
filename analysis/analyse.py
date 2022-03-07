from sklearn import metrics
from sklearn.metrics import confusion_matrix

####
# return mean absolute error
####
def MAE(y_test, y_pred):
    return metrics.mean_absolute_error(y_test,y_pred)

####
# return r2 score
####
def r2(y_test, y_pred):
    return metrics.r2_score(y_test,y_pred)
####
# return root mean squared error
####
def rmse(y_test, y_pred):
    return metrics.mean_squared_error(y_test,y_pred)


