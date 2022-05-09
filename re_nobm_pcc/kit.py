import os
import pathlib

import numpy as np
import xarray as xr


PROJECT_DIR = pathlib.Path(os.environ['PWD'])
DATA_DIR = PROJECT_DIR/'data'


def ecdf(data, axis=0):
    """Empirical cumulative distribution function(s) from a data array.

    :param data: an array-like of samples from a random variable
    :param axis: the axis over which to calculate probabilities

    :return: a matching array-like of (marginal) cumulative probabilities
    """
    prb = np.expand_dims(
        np.linspace(1/data.shape[axis], 1, data.shape[axis]),
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
    if k is None:
        k = k_default
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    scores = xr.DataArray(
        data=u[:, :k]*s[:k],
        coords={'pc': ('pc', [f'PC{i}' for i in range(k)])},
        dims=tuple(sizes) + ('pc',),
        name='coefficient',
    )
    vectors = xr.DataArray(
        data=vh[:k, :],
        coords={dim: data[dim]},
        dims=('pc', dim),
        name='weight',
    )
    return scores, s[:k], vectors