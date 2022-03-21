from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

####
# return mean absolute error
####
def mae(y_test, y_pred):
    """
    Function to calculate the mean aboslute error with sklearn lib

    :param y_test:

    :param y_pred:

    :return: mea

    :raise : Exception if the argument haven't same shape
    """
    if y_test.shape != y_pred.shape:
        raise Exception("Analyse : MAE, not same shape")

    return metrics.mean_absolute_error(y_test, y_pred)


####
# return r2 score
####
def r2(y_test, y_pred):
    """
    Function to calculate the r2 with sklearn lib

    :param y_test:

    :param y_pred:

    :return: r2

    :raise : Exception if the argument haven't same shape
    """
    if y_test.shape != y_pred.shape:
        raise Exception("Analyse : r2 score, not same shape")
    return metrics.r2_score(y_test, y_pred)


####
# return root mean squared error
####
def rmse(y_test, y_pred):
    """
    Function to calculate the root mean squarred error with sklearn lib

    :param y_test:

    :param y_pred:

    :return: rmse

    :raise : Exception if the argument haven't same shape
    """
    if y_test.shape != y_pred.shape:
        raise Exception("Analyse : rmse, not same shape")
    return metrics.mean_squared_error(y_test, y_pred)


####
# Plot a correlation matrix
####
def correlation_matrix(dataset):
    """
    Function to create a correlation_matrix with seaborn lib, save it as png into analysis folder

    :param dataset:
        Pandas dataframe

    :return:
    """
    fig = plt.figure(figsize=[15, 10])
    sns.heatmap(dataset.corr())
    plt.title("Correlation matrix")
    print("Write file analysis/correlationMatrix.png")
    plt.savefig("analysis/correlationMatrix.png")
    plt.show()
