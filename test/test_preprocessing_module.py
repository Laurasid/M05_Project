import sys
import os

import numpy
import pandas.core.frame
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src/preprocessor'))
import preprocessing as pr
import pkg_resources
DATAFILE = pkg_resources.resource_filename(__name__, "data.csv")


def test_import_dataset():
    """
    Test instance of the object returned by preprocessor.import_dataset method.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Instances differ
    """
    dataset_csv = pr.import_dataset(pkg_resources.resource_filename(__name__, "../src/Data/winequality-red.csv"))
    dataset_data = pr.import_dataset(pkg_resources.resource_filename(__name__, "../src/Data/housing.data"))

    # test instances of returned objects
    assert type(dataset_csv) == pandas.core.frame.DataFrame, "Dataset isn't imported as a DataFrame object"
    assert type(dataset_data) == pandas.core.frame.DataFrame, "Dataset isn't imported as a DataFrame object"


def test_standard_scaling():
    """
    Test instance of the object returned by preprocessor.standard_scaling method.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Instances differ
    """
    dataset_scaled = pr.standard_scaling(pr.import_dataset\
                    (pkg_resources.resource_filename(__name__, "../src/Data/winequality-red.csv")))

    # test instances of returned objects
    assert type(dataset_scaled) == numpy.ndarray, "Scaled dataset isn't return as a numpy.ndarray object"


def test_min_max_scaling():
    """
    Test instance of the object returned by preprocessor.min_max_scaling method.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Instances differ
    """
    dataset_scaled = pr.min_max_scaling(pr.import_dataset \
                    (pkg_resources.resource_filename(__name__, "../src/Data/winequality-red.csv")))

    # test instances of returned objects
    assert type(dataset_scaled) == numpy.ndarray, "Scaled dataset isn't return as a numpy.ndarray object"


def test_polynomial_scaling():
    """
    Test instance of the object returned by preprocessor.polynomial_scaling method.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Instances differ
    """
    dataset_scaled = pr.polynomial_scaling(pr.import_dataset \
                    (pkg_resources.resource_filename(__name__, "../src/Data/winequality-red.csv")))

    # test instances of returned objects
    assert type(dataset_scaled) == numpy.ndarray, "Scaled dataset isn't return as a numpy.ndarray object"


def test_normalize():
    """
    Test instance of the object returned by preprocessor.normalize method.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Instances differ
    """
    dataset_scaled = pr.normalize(pr.import_dataset \
                    (pkg_resources.resource_filename(__name__, "../src/Data/winequality-red.csv")))

    # test instances of returned objects
    assert type(dataset_scaled) == numpy.ndarray, "Normalized dataset isn't return as a numpy.ndarray object"


def test_preprocessing():
    dataset = pr.import_dataset(pkg_resources.resource_filename(__name__, "../src/Data/winequality-red.csv"))

    x_train, x_test, y_train, y_test = pr.preprocessing(dataset, 1, 1)

    # test instances of returned objects
    assert type(x_train) == numpy.ndarray and type(y_train) == numpy.ndarray and \
           type(x_test) == numpy.ndarray and type(y_test) == numpy.ndarray

    # test assertion of preprocessing::preprocessing method.
    # method should raise an exception in the splitting or scaling option aren't known

    # check for splitting option exception
    with pytest.raises(Exception):
        pr.preprocessing(dataset, 0, 2)

    # check for scaling method
    with pytest.raises(Exception):
        pr.preprocessing(dataset, 2, 0)
