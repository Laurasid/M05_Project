import pytest
import sys
sys.path.append('../model')
import linear_regression as lr
import numpy as np
import sklearn
from sklearn import metrics
import random

def test_lin_reg():
    x_train = np.array([np.random.rand(2),
                        np.random.rand(2)])
    y_train = np.array([np.random.rand(2),
                        np.random.rand(2)])
    x_test = np.array([np.random.rand(2),
                       np.random.rand(2)])

    regressor = lr.train(x_train,y_train)
    y_pred = lr.predict(regressor,x_test)

    # test shape of the model's output value
    assert np.shape(y_pred) == np.shape(x_test)