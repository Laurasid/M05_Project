import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src/analysis'))

import pytest
import analyse as an
import numpy as np
from sklearn import metrics

"""
This module is used to test methods of the analysis.analyse module
"""


def test_mae():
    """
    Test analyse::mae function (Mean Absolute Error).

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected
    """
    ###
    # mae = 1/n * sum(|xi-yi|)
    ###
    mae = lambda _xi, _yi: 1/len(_xi) * np.sum(np.abs(_xi - _yi))

    # test with null values
    assert an.mae(np.zeros(1), np.zeros(1)) == 0

    # test with negative values (allowed)
    x_n = np.full(1, -5)
    y_n = np.full(1, 6)

    assert np.isclose(an.mae(x_n, y_n), mae(x_n, y_n))

    # test with set of values
    x = np.full(5, [1, 5, 3, 7, 8])
    y = np.full(5, [5, 3, 7, 1, 2])

    assert np.isclose(an.mae(x, y), mae(x, y))

    # check for return type
    assert isinstance(an.mae(x, y), float)

    # test with value sets of different shapes (not allowed)
    x_2 = np.full(5,[1, 5, 3, 7, 8])
    y_2 = np.full(4,[5, 3, 7, 1])

    with pytest.raises(Exception):
        an.mae(x_2, y_2)


def test_r2():
    """
    Test analyse::r2 function (R-square).

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected
    """
    ###
    # r^2 = 1 - sum(yi-yi^)^2 / sum(yi-y_mean)^2
    ###
    r2 = lambda _yi, _y_hat: 1 - (np.sum((_yi-_y_hat)**2) / np.sum((_yi-np.mean(_yi))**2))

    # test with same values.
    # r^2 should be 1 because values that represent regression model and real data are the same
    y = np.array([2, 7, 6, 8, 5, 4])
    y_hat = np.array([2, 7, 6, 8, 5, 4])

    assert an.r2(y, y_hat) == 1.0

    # test with different values
    y = np.array([2, 7, 6, 8, 5, 4])
    y_hat = np.array([8, 7, 5, 3, 9, 1])

    assert np.isclose(an.r2(y, y_hat), r2(y, y_hat))

    # test with negative values (allowed)
    x_n = np.array([2, -3])
    y_n = np.array([-4, 5])

    assert np.isclose(an.r2(x_n, y_n), r2(x_n, y_n))

    # check for return type
    assert isinstance(an.mae(y, y_hat), float)


    # test with values set of different shapes (not allowed)
    x_2 = np.full(5, [1, 5, 3, 7, 8])
    y_2 = np.full(4, [5, 3, 7, 1])

    with pytest.raises(Exception):
        an.r2(x_2, y_2)


def test_rmse():
    """
    Test analyse::rmse function (Root Mean Square Error).

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected
    """
    ###
    # rmse = sqrt(1/n * sum(yi^-yi)^2)
    ###
    # TODO : check the rmse lambda function. Give not the result that excepted
    rmse = lambda _yi, _y_hat: np.sqrt(np.mean((_yi - _y_hat)**2))

    # Test for null values
    assert an.rmse(np.zeros(2), np.zeros(2)) == 0

    # Test for identical values
    x = np.array([1, 6, 3, 8, 5])
    y = np.array([1, 6, 3, 8, 5])

    assert an.rmse(x, y) == 0.0

    # Test for different values
    x_1 = np.array([1, 6, 3, 8, 5])
    y_1 = np.array([9, 5, 2, 3, 5])

    assert np.isclose(an.rmse(x_1, y_1), metrics.mean_squared_error(x_1, y_1))

    # test with negative values (allowed)
    x_n = np.array([2, -3])
    y_n = np.array([-4, 5])

    assert np.isclose(an.rmse(x_n, y_n), metrics.mean_squared_error(x_n, y_n))

    # check for return type
    assert isinstance(an.mae(x_1, y_1), float)

    # test with values set of different shapes (not allowed)
    x_2 = np.full(5, [1, 5, 3, 7, 8])
    y_2 = np.full(4, [5, 3, 7, 1])

    with pytest.raises(Exception):
        an.rmse(x_2, y_2)
