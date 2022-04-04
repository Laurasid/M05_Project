import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src/model'))
import linear_regression as lr
import numpy as np

"""
This module is used to test methods of the model.linear_regression module
"""

def test_lin_reg():
    """
    Test shapes and instance of the object returned by model.linear_regression methods.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Shapes or instance differ
    """
    x_train = np.array([np.random.rand(2),
                        np.random.rand(2)])
    y_train = np.array([np.random.rand(2),
                        np.random.rand(2)])
    x_test = np.array([np.random.rand(2),
                       np.random.rand(2)])

    regressor = lr.train(x_train, y_train)
    y_pred = lr.predict(regressor, x_test)

    # test shape and type of the model's output value
    assert np.shape(y_pred) == np.shape(x_test), "Shapes don't match"
    assert isinstance(y_pred, type(x_test)), "Returned object isn't of the expected instance"
