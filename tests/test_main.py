from pathlib import Path
import shutil

from dask.distributed import Client
import numpy as np
import pytest
import xarray as xr

from re_nobm_pcc import CHUNKSIZE


@pytest.fixture(scope="session")
def client():
    c = Client()
    yield c
    c.shutdown()


@pytest.fixture
def cache_simulate(request, client):
    from re_nobm_pcc import simulate

    path = Path(request.config.cache.makedir("data"))
    if not (path / "oasim").exists():
        period = np.arange("1998-01", "1998-06", dtype=np.dtype("datetime64[M]"))
        simulate.main(period, path=path, days=2)
    yield path


@pytest.fixture
def cache_preprocess(cache_simulate, client):
    from re_nobm_pcc import preprocess

    path = cache_simulate
    if not (path / "labelled.zarr").exists():
        split = 2 ** np.array((3, 2, 1)) * CHUNKSIZE
        preprocess.main(split=split, path=path)
    yield path


@pytest.fixture
def cache_learn(cache_preprocess, client):
    from re_nobm_pcc import learn

    path = cache_preprocess
    if not (path / "fit.zarr").exists():
        learn.main(epochs=2, path=path)
    yield path


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
        for item in ("train", "validate", "test"):
            dataset = xr.open_dataset(path, engine="zarr", group=item, chunks={})
        assert set(dataset.variables) == {"x", "y"}
        assert dataset["x"][0, 0].notnull()
        assert dataset["y"][0, 0].notnull()
    except Exception as fail:
        shutil.rmtree(path)
        raise fail


def test_learn(cache_learn):
    path = cache_learn / "fit.zarr"
    try:
        dataset = xr.open_dataset(path, engine="zarr")
    except Exception as fail:
        shutil.rmtree(path)
