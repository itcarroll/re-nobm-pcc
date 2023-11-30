from pathlib import Path

import numpy as np
import xarray as xr


DATADIR = (Path(__file__).parents[1] / "data").absolute()
TAXA = ("dia", "chl", "cya", "coc", "din", "pha")
WAVELENGTH = tuple(range(350, 731))
LONLAT = (288, 234)


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
