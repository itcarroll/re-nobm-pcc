from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import xarray as xr

from . import DATADIR, TAXA, CHUNKSIZE
from .core import read_nobm

if TYPE_CHECKING:
    import pathlib


WAVELENGTH = tuple(range(350, 731))


def main(split: np.ndarray, path: "pathlib.Path") -> None:

    # container for labelled data
    labelled = xr.Dataset()

    # open the "x" data and attach a random index
    oasim = xr.open_mfdataset(list((path / "oasim").glob("*")))  # note: is chunked
    oasim = oasim["rrs"].stack({"m": ["date", "lon", "lat"]}, create_index=False)
    oasim = oasim.sel({"wavelength": list(WAVELENGTH)})
    oasim = oasim.transpose("m", ...)

    # sample "n" from "m" exampes by taking from a random permutation of positions
    n = split.sum()
    m = oasim.sizes["m"]
    seed = 53946417194839354210102657616485775828  # from `secrets.randbits(128)`
    rng = np.random.default_rng(seed)
    position = rng.permutation(m)

    # iteratively take only samples that are not missing
    sample = np.array((), dtype=int)
    start = 0
    while sample.size < n and start < m:
        candidate = position[start : (start + n)]
        candidate.sort()  # note: eliminates a dask performance warning
        missing = oasim.isel({"m": candidate}).isnull().any(dim="wavelength")
        sample = np.concatenate((sample, candidate[~missing]))
        start += n
    sample = sample[:n]
    oasim = oasim.isel({"m": sample})

    # copy "x" and "y" samples to prepared store
    labelled = preprocess_oasim(labelled, oasim)
    labelled = preprocess_nobm(labelled)

    # write train, validate, and test groups to store on disk
    path = path / "labelled.zarr"
    i = 0
    for item in zip(split, ("train", "validate", "test")):
        n, group = item
        dataset = labelled.isel({"n": slice(i, i + n)})
        coords = dataset.coords
        dataset.drop_vars(list(coords)).to_zarr(path, group=group)
        coords.to_dataset().load().to_zarr(path, group=f"{group}/coords")
        i += n


def preprocess_oasim(dataset: xr.Dataset, array: xr.DataArray) -> xr.Dataset:

    array = array.rename({"m": "n", "wavelength": "s"})
    dataset["x"] = array.chunk({"n": CHUNKSIZE, "s": -1})

    return dataset


def preprocess_nobm(dataset: xr.Dataset) -> xr.Dataset:

    # create an empty dask array for parallel reading of nobm
    n = dataset.sizes["n"]
    t = len(TAXA)
    meta = {"dtype": "float32", "chunks": (CHUNKSIZE, -1)}  # TODO too small?
    array = xr.DataArray(
        da.empty((n, t), **meta),
        coords={"t": list(TAXA)},
        dims=("n", "t"),
        name="y",
    )
    array = xr.merge((dataset.coords, array))["y"]

    # map the read function over chunks
    coords = list(array.coords)
    coords.remove("t")
    dataset["y"] = array.map_blocks(update_nobm, args=(coords,), template=array)

    return dataset


def update_nobm(array: xr.DataArray, coords: list[str]) -> xr.DataArray:

    # container for nobm phy variable
    array = xr.full_like(array, np.nan)
    array = array.set_xindex(coords)
    # round the date to first of month
    yearmonth = array["date"].data.astype("datetime64[M]")
    yearmonth = xr.DataArray(yearmonth.astype("datetime64[ns]"), dims="n")
    for key, value in array.groupby(xr.DataArray(yearmonth, dims="n")):
        period = key.astype("datetime64[M]")
        dataset = read_nobm(period, days=value["date.day"].max().item())
        dataset = dataset[["phy"]].rename({"phy": "y", "component": "t"})
        dataset = dataset.stack({"n": coords}).transpose("n", ...)
        y, _ = xr.align(dataset["y"], array)
        array.loc[{"n": y["n"]}] = y
    array["t"] = y["t"]
    array = array.reset_index("n")

    return array


if __name__ == "__main__":
    from dask.distributed import Client

    with Client() as client:
        print(f"Dashboard link: {client.dashboard_link}")
        main(
            split=2 ** np.array((9, 7, 5)) * CHUNKSIZE,
            path=DATADIR,
        )
