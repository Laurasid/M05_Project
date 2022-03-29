import pytest
import sys
sys.path.append('../model')
import linear_regression as lr
import numpy as np
import random

def test_lin_reg():
    """
    Function to test linear regression module. Test for shape and type of values returned.

    :param None

    :return: None

    :raise: AssertionError
    """
    x_train = np.array([np.random.rand(2),
                        np.random.rand(2)])
    y_train = np.array([np.random.rand(2),
                        np.random.rand(2)])
    x_test = np.array([np.random.rand(2),
                       np.random.rand(2)])

    regressor = lr.train(x_train,y_train)
    y_pred = lr.predict(regressor,x_test)

    # test shape and type of the model's output value
    assert np.shape(y_pred) == np.shape(x_test)
    assert isinstance(y_pred,type(x_test))