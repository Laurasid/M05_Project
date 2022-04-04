import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
import regression_tree as rt
import numpy as np

"""
This module is used to test methods of the model.regression_tree module
"""

def test_reg_tree():
    """
    Test shapes and instance of the object returned by model.regression_tree methods.

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

    regressor = rt.train(x_train, y_train)
    y_pred = rt.predict(regressor, x_test)

    # test shape and type of the model's output value
    assert np.shape(y_pred) == np.shape(x_test), "Shapes don't match"
    assert isinstance(y_pred, type(x_test)), "Returned object isn't of the expected instance"
    