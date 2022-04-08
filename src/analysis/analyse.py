from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

"""
This module is used to analyse the results of the model trained. It offers different loss functions methods.
"""

def mae(y_test, y_pred):
    """
    Compute the mean absolute error with sklearn lib

    :param array-like y_test:
        labels that correspond to the values of test dataset
    :param array-like y_pred:
        labels that are predicted by the model based on values of test dataset

    :return: (float) -
        the mean absolute error

    :raise Exception:
        the arguments don't have same shapes
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
        labels that are predicted by the model based on values of test dataset

    :return: (float) -
        r-square score

    :raise Exception:
        the arguments don't have same shapes
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
        labels that are predicted by the model based on values of test dataset

    :return: (float) -
        the root-mean-square error

    :raise Exception:
        the arguments don't have same shapes
    """
    if y_test.shape != y_pred.shape:
        raise Exception("Analyse : rmse, not same shape")
    return metrics.mean_squared_error(y_test, y_pred)


def correlation_matrix(dataset):
    """
    Create a correlation matrix with seaborn lib, save it as png into ./analysis/ \n
    Once the correlation matrix is created it's shown on screen.

    :param DataFrame dataset:
        Pandas DataFrame object

    :return: None
    """
    plt.figure(figsize=[15, 10])
    sns.heatmap(dataset.corr())
    plt.title("Correlation matrix")
    print("Write file analysis/correlationMatrix.png")
    plt.savefig("src/analysis/correlationMatrix.png")
    plt.show()
