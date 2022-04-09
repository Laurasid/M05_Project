import sys
import os

import numpy
import pandas.core.frame

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
    dataset = pr.import_dataset(pkg_resources.resource_filename(__name__, "../src/Data/winequality-red.csv"))

    assert type(dataset) == pandas.core.frame.DataFrame, "Dataset isn't imported as a DataFrame object"

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

    assert type(dataset_scaled) == numpy.ndarray, "Scaled dataset isn't return as a numpy.ndarray object"


def test_min_max_scaling():
    """
    Test instance of the object returned by preprocessor.min_max_scaling method.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Instances differ
    """
    dataset_scaled = pr.standard_scaling(pr.import_dataset \
                                             (pkg_resources.resource_filename(__name__,
                                                                              "../src/Data/winequality-red.csv")))

    assert type(dataset_scaled) == numpy.ndarray, "Scaled dataset isn't return as a numpy.ndarray object"


def test_polynomial_scaling():
    """
    Test instance of the object returned by preprocessor.polynomial_scaling method.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Instances differ
    """
    dataset_scaled = pr.standard_scaling(pr.import_dataset \
                                             (pkg_resources.resource_filename(__name__,
                                                                              "../src/Data/winequality-red.csv")))

    assert type(dataset_scaled) == numpy.ndarray, "Scaled dataset isn't return as a numpy.ndarray object"


def test_normalize():
    """
    Test instance of the object returned by preprocessor.normalize method.

    :param: None

    :return: None

    :raise AssertionError:
        the result is not the one that's expected. Instances differ
    """
    dataset_scaled = pr.standard_scaling(pr.import_dataset \
                                             (pkg_resources.resource_filename(__name__,
                                                                              "../src/Data/winequality-red.csv")))

    assert type(dataset_scaled) == numpy.ndarray, "Normalized dataset isn't return as a numpy.ndarray object"
