from pathlib import Path

import numpy as np
import xarray as xr

from re_nobm_pcc.core import ecdf, read_nobm
from re_nobm_pcc.simulate import oasim


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
    oasim(period, Path(tmpdir), days=2)
    yearmonth = period.item().strftime("%Y%m")
    dataset = xr.open_dataset(tmpdir / "oasim" / f"rrs{yearmonth}.nc", engine="netcdf4")
    assert dataset.sizes["date"] == 2
