import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
import regression_tree as rt
import numpy as np


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

    regressor = rt.train(x_train, y_train)
    y_pred = rt.predict(regressor, x_test)

    # test shape of the model's output value
    assert np.shape(y_pred) == np.shape(x_test)

    # module should return data with same type as the input
    assert isinstance(y_pred, type(x_test))
    