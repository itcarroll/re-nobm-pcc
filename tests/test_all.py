from pathlib import Path
import shutil

import numpy as np
import pytest
import xarray as xr
import dask.config

from re_nobm_pcc import CHUNKSIZE
from re_nobm_pcc.core import ecdf, read_nobm
from re_nobm_pcc import simulate
from re_nobm_pcc import preprocess


@pytest.fixture(params=["synchronous", "threads"])
def cache_simulate(request):
    scheduler = request.param
    dask.config.set(scheduler=scheduler)
    path = Path(request.config.cache.makedir(scheduler))
    if not (path / "oasim").exists():
        period = np.arange("1998-01", "1998-03", dtype=np.dtype("datetime64[M]"))
        simulate.main(period, path=path, days=2)
    yield path


@pytest.fixture
def cache_preprocess(cache_simulate):
    path = cache_simulate
    if not (path / "labelled.zarr").exists():
        split = 2 ** np.array((5, 4, 3)) * CHUNKSIZE
        preprocess.main(split=split, path=path)
    yield path


def test_ecdf():
    ds = np.array([[0.5, 0.7, 0.2], [0.8, 0.4, 0.9]])
    probabilities = ecdf(ds)
    assert np.allclose([[1 / 2, 1, 1 / 2], [1, 1 / 2, 1]], probabilities)
    probabilities = ecdf(ds, axis=1)
    assert np.allclose([[2 / 3, 1, 1 / 3], [2 / 3, 1 / 3, 1]], probabilities)


def test_read_nobm():
    period = np.datetime64("2007-01")
    dataset = read_nobm(period, days=4)
    assert dataset.sizes["date"] == 4


def test_oasim(tmpdir):
    period = np.datetime64("2003-07")
    simulate.oasim(period, Path(tmpdir), days=2)
    yearmonth = period.item().strftime("%Y%m")
    dataset = xr.open_dataset(tmpdir / "oasim" / f"rrs{yearmonth}.nc", engine="netcdf4")
    assert dataset.sizes["date"] == 2


def test_simulate(cache_simulate):
    try:
        dataset = xr.open_dataset(cache_simulate / "oasim" / "rrs199802.nc")
        assert dataset.sizes["date"] == 2
    except Exception as fail:
        shutil.rmtree(cache_preprocess / "labelled.zarr")
        raise fail


def test_preprocess(cache_preprocess):
    path = cache_preprocess / "labelled.zarr"
    try:
        dataset = xr.open_dataset(path, engine="zarr", group="train", chunks={})
        assert set(dataset.variables) == {"x", "y"}
        assert dataset["x"][0, 0].notnull()
        assert dataset["y"][0, 0].notnull()
    except Exception as fail:
        shutil.rmtree(path)
        raise fail


def test_split():
    train = preprocess.train.sizes["pxl"]
    validate = preprocess.validate.sizes["pxl"]
    test = preprocess.test.sizes["pxl"]
    assert (train > validate) and (validate > test)


def test_perceptron_model():
    assert perceptron.Full().built


def test_cnn_model():
    assert cnn.Full().built
