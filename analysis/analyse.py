from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

"""
This module is used to analyse the results of the model. It offers different loss functions methods.
"""

def mae(y_test, y_pred):
    """
    Compute the mean absolute error with sklearn lib

    :param array-like y_test:
                labels that correspond to the values of test dataset
    :param array-like y_pred:
                labels that are predicted by the model based on values of train set

    :return float:
                the mean absolute error

    :raise Exception:
                if the arguments don't have same shapes
    """
    if y_test.shape != y_pred.shape:
        raise Exception("Analyse : MAE, not same shape")

    return metrics.mean_absolute_error(y_test, y_pred)


def r2(y_test, y_pred):
    """
    Compute the r-square score with sklearn lib

    :param array-like y_test:
                labels that correspond to the values of test dataset
    :param array-like y_pred:
                labels that are predicted by the model based on values of train set

    :return float:
                r-square score

    :raise Exception:
                if the arguments don't have same shapes
    """
    if y_test.shape != y_pred.shape:
        raise Exception("Analyse : r2 score, not same shape")
    return metrics.r2_score(y_test, y_pred)


def rmse(y_test, y_pred):
    """
    Compute the root-mean-square error with sklearn lib

    :param array-like y_test:
                labels that correspond to the values of test dataset
    :param array-like y_pred:
                labels that are predicted by the model based on values of train set

    :return float:
                the root-mean-square error

    :raise Exception:
                if the arguments don't have same shapes
    """
    if y_test.shape != y_pred.shape:
        raise Exception("Analyse : rmse, not same shape")
    return metrics.mean_squared_error(y_test, y_pred)


def correlation_matrix(dataset):
    """
    Create a correlation matrix with seaborn lib, save it as png into ./analysis/
    Once the correlation matrix is created it's shown on screen.
    :param DataFrame dataset:
                Pandas DataFrame object

    :return: None
    """
    plt.figure(figsize=[15, 10])
    sns.heatmap(dataset.corr())
    plt.title("Correlation matrix")
    print("Write file analysis/correlationMatrix.png")
    plt.savefig("analysis/correlationMatrix.png")
    plt.show()
