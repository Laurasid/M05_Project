from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

####
# return mean absolute error
####
def MAE(y_test, y_pred):
    if(y_test.shape != y_pred.shape):
        raise Exception("Analyse : MAE, not same shape")

    return metrics.mean_absolute_error(y_test,y_pred)

####
# return r2 score
####
def r2(y_test, y_pred):
    if(y_test.shape != y_pred.shape):
        raise Exception("Analyse : r2 score, not same shape")
    return metrics.r2_score(y_test,y_pred)
####
# return root mean squared error
####
def rmse(y_test, y_pred):
    if(y_test.shape != y_pred.shape):
        raise Exception("Analyse : rmse, not same shape")
    return metrics.mean_squared_error(y_test,y_pred)


####
# Plot a correlation matrix
####
def correlation_matrix(dataset):
    sns.heatmap(dataset.corr())
    plt.title('Correlation matrix')
    plt.show()
