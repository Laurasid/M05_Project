import pytest
import sys
import os
#sys.path.append(os.path.dirname(sys.path[0]))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

sys.path.append('../model')
import regression_tree as rt
import numpy as np
import random

def test_reg_tree():
    """
    Function to test regression tree module. Test for shape and type of values returned.

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

    regressor = rt.train(x_train,y_train)
    y_pred = rt.predict(regressor,x_test)

    # test shape of the model's output value
    assert np.shape(y_pred) == np.shape(x_test)

    # module should return data with same type as the input
    assert isinstance(y_pred,type(x_test))