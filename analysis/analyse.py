from sklearn import metrics
import matplotlib.pyplot as plt

def MAE(y_test, y_pred):
    return metrics.mean_absolute_error(y_test,y_pred)