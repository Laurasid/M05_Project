import sys
sys.path.append('../model')
import regression_tree as rt
import numpy as np
import sklearn
from sklearn import metrics
import random

def test_reg_tree():
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