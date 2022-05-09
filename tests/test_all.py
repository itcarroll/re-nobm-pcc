import numpy as np

from re_nobm_pcc import preprocess #, perceptron, cnn
from re_nobm_pcc.kit import ecdf


def test_split():
    train = preprocess.train.sizes['pxl']
    validate = preprocess.validate.sizes['pxl']
    test = preprocess.test.sizes['pxl']
    assert (train > validate) and (validate > test)


def test_perceptron_model():
    assert perceptron.Full().built


def test_cnn_model():
    assert cnn.Full().built


def test_ecdf():
    ds = np.array([[.5, .7, .2], [.8, .4, .9]])
    probabilities = ecdf(ds)
    assert np.allclose([[1/2, 1, 1/2], [1, 1/2, 1]], probabilities)
    probabilities = ecdf(ds, axis=1)
    assert np.allclose([[2/3, 1, 1/3], [2/3, 1/3, 1]], probabilities)
