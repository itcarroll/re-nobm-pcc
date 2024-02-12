from pathlib import Path
import os

import numpy as np
import pytest
import xarray as xr

# from re_nobm_pcc import preprocess  # , perceptron, cnn
from re_nobm_pcc.core import ecdf, read_nobm
from re_nobm_pcc.simulate import oasim
from re_nobm_pcc.preprocess import combine_nobm_oasim


@pytest.fixture
def cache_oasim(pytestconfig):
    path = Path(pytestconfig.cache.makedir("data"))
    if not (path / "oasim").exists():
        oasim(np.datetime64("1998-01"), path, days=2)
        oasim(np.datetime64("1998-02"), path, days=2)
    yield path
    # TODO cleanup?


def test_ecdf():
    ds = np.array([[0.5, 0.7, 0.2], [0.8, 0.4, 0.9]])
    probabilities = ecdf(ds)
    assert np.allclose([[1 / 2, 1, 1 / 2], [1, 1 / 2, 1]], probabilities)
    probabilities = ecdf(ds, axis=1)
    assert np.allclose([[2 / 3, 1, 1 / 3], [2 / 3, 1 / 3, 1]], probabilities)


def test_read_nobm():
    period = np.datetime64("2007-01")
    ds = read_nobm(period, days=4)
    assert ds.sizes["date"] == 4


def test_oasim(tmpdir):
    period = np.datetime64("2003-07")
    oasim(period, Path(tmpdir), days=2)
    yearmonth = period.item().strftime("%Y%m")
    dataset = xr.open_dataset(tmpdir / "oasim" / f"rrs{yearmonth}.nc", engine="netcdf4")
    assert dataset.sizes["date"] == 2


def test_preprocess(cache_oasim):
    combine_nobm_oasim(cache_oasim)
    assert False


def test_split():
    train = preprocess.train.sizes["pxl"]
    validate = preprocess.validate.sizes["pxl"]
    test = preprocess.test.sizes["pxl"]
    assert (train > validate) and (validate > test)


def test_perceptron_model():
    assert perceptron.Full().built


def test_cnn_model():
    assert cnn.Full().built
