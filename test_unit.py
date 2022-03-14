import pytest
from preprocessor import preprocessing as pp
from model import linear_regression as lr
import numpy as np

# test functions need "test_" as prefixe
def test_shape_standard_scaling():
	numb_list = np.ones((2,3))
	scaling_list = pp.standardScaling(numb_list)
	assert  scaling_list.shape[0] == numb_list.shape[0] and scaling_list.shape[1] == numb_list.shape[1] 

