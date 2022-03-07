from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def MAE(y_test, y_pred):
    return metrics.mean_absolute_error(y_test,y_pred)

#r2 score
def r2(y_test, y_pred):
    return metrics.r2_score(y_test,y_pred)

# root mean squared error
def rmse(y_test, y_pred):
    return metrics.mean_squared_error(y_test,y_pred)


def conf_mat(y_test, y_pred):

    results = confusion_matrix(y_test, y_pred)
    return results

