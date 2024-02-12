from pathlib import Path
from typing import BinaryIO
from contextlib import contextmanager
from os import getcwd, chdir

import numpy as np
import xarray as xr


DATADIR = (Path(__file__).parents[1] / "data").absolute()
TAXA = ("dia", "chl", "cya", "coc", "din", "pha")
OC = ("tot", "dtc", "pic", "cdc", "t", "s")
BGC = ("alk", "dic", "doc", "fco", "h", "irn", "pco", "pp", "rnh", "rno", "sil", "zoo")
WAVELENGTH = tuple(range(350, 731))
NUMNAN = np.array(9.99e11, dtype="f4")


@contextmanager
def oasim_param():
    """Within this context, the oasim modules will see the hard-coded `oasim_params`
    directory.
    """
    _cwd = getcwd()
    chdir(DATADIR)
    try:
        yield
    finally:
        chdir(_cwd)


def ecdf(data, axis=0):
    """Empirical cumulative distribution function(s) from a data array.

    :param data: an array-like of samples from a random variable
    :param axis: the axis over which to calculate probabilities

    :return: a matching array-like of (marginal) cumulative probabilities
    """

    prb = np.expand_dims(
        np.linspace(1 / data.shape[axis], 1, data.shape[axis]),
        tuple(range(axis + 1, len(data.shape))),
    )
    idx = np.argsort(data, axis=axis)
    arr = np.empty_like(data, dtype=prb.dtype)
    np.put_along_axis(arr, idx, prb, axis=axis)

    return arr


def svd(data, dim, k=None):
    """Singular Value Decomposition (SVD) with PCA interpretation

    :param data: an array-like of samples from a random variable
    :param dim: the dimension to "reduce" via PCA

    :return: a triple of PCA scores, singular values, and components
    """

    sizes = dict(data.sizes)
    k_default = sizes.pop(dim)
    pc = "percentage"
    if k is None:
        k = k_default
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    s2 = s**2
    return xr.Dataset(
        {
            "weights": (list(sizes) + [pc], u[..., :k] * s[:k]),
            "vectors": ([pc, dim], vh[:k, ...]),
            pc: (pc, (s2 / s2.sum())[:k]),
            dim: (dim, data[dim].data),
        }
    )


def read_nobm(start: np.datetime64, days: int | None = None) -> xr.Dataset:
    """Read the daily NOBM data files provided by C. Rouseaux"""

    # ## container dataset with coordinates
    stop = start + (np.timedelta64(days, "D") if days else np.timedelta64(1, "M"))
    step = np.timedelta64(1, "D")
    ds = xr.Dataset(
        coords={
            "date": np.arange(start, stop, step).astype("datetime64[ns]"),
            "lon": np.linspace(0, 360, 288, endpoint=False, dtype="float32"),
            "lat": np.linspace(-84, 71.4, 234, endpoint=False, dtype="float32"),
        },
    )
    dims = tuple(ds.sizes)
    shape = (ds.sizes["lon"], ds.sizes["lat"])
    size = np.prod(shape)

    # ## read all variables
    yearmonth = start.item().strftime("%Y%m")
    for item in TAXA + OC + BGC:
        with open(DATADIR / f"nobm/{item}/{item}{yearmonth}", "rb") as f:
            da = []
            for _ in ds.groupby("date.day"):
                # discard mystery array prepended to some files
                if item not in ["fco", "pco"]:
                    _ = fromfile(f)
                da.append(fromfile(f, shape))
                if item in ["fco", "pco", "pp"]:
                    if item == "pp":
                        for _ in TAXA:
                            da.append(fromfile(f, shape))
                    continue
                # skip remaining 13 layers bytes
                f.seek((4 * (1 + size + 1)) * 13, 1)
        if item == "pp":
            da = np.stack(da).reshape((-1, len(TAXA) + 1, *shape)).transpose(0, 2, 3, 1)
            ds["tpp"] = xr.DataArray(da[..., 0], dims=dims)
            ds[item] = xr.DataArray(da[..., 1:], dims=dims + ("component",))
        else:
            ds[item] = xr.DataArray(np.stack(da), dims=dims)

    # ## combine
    # convert phy variables to one xr.DataArray
    phy = ds[list(TAXA)]
    ds = ds.drop_vars(TAXA)
    ds["phy"] = phy.to_array(dim="component").transpose(..., "component")
    # set numbers representing nan to np.nan
    # tot has an odd NaN flag
    da = ds["tot"]
    ds["tot"] = da.where(da != np.float32(5.9939996e12))
    # everything else uses the same NaN flag
    ds = ds.where(ds != NUMNAN)

    return ds


def fromfile(file: BinaryIO, shape: tuple[int] | None = (-1,)) -> np.ndarray:

    # read start record size
    size = np.fromfile(file, "i4", 1)[0]
    # skip f"{size}" bytes of unknown purpose
    array = np.fromfile(file, "f4", size // 4)
    array = array.reshape(shape, order="F")
    # verify at end of record
    check_size = np.fromfile(file, "i4", 1)[0]
    assert size == check_size

    return array
