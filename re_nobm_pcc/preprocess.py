import numpy as np
import tensorflow as tf
import xarray as xr

from . import DATADIR, WAVELENGTH


def array_to_tensor(array: xr.DataArray):
    def py_function(isel):
        return tf.convert_to_tensor(array.isel({"example": isel}).values)

    return py_function


def convert_dataset(
    dataset: xr.Dataset,
    split: int,
    batch_size: int,
) -> tf.data.Dataset:
    # subset on split
    example = dataset["example"].where(dataset["split"] == split, other=0, drop=True)
    # shuffle and batch the indices
    index = (
        tf.data.Dataset.from_tensor_slices(example)
        .shuffle(buffer_size=example.size)
        .batch(batch_size)
        .map(tf.sort)  # nb. sorting within the batch does not negate shuffling
    )
    return index.map(
        lambda i: (
            tf.py_function(array_to_tensor(dataset["x"]), [i], Tout=tf.float32),
            tf.py_function(array_to_tensor(dataset["y"]), [i], Tout=tf.float32),
        )
    )


def open_dataset(
    p: tuple[float],
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[tf.data.Dataset]:
    # open connection to data files and organize
    path = (DATADIR / "oasim").glob("*.nc")
    if __debug__:
        path = (DATADIR / "oasim").glob("*806.nc")
    dataset = xr.open_mfdataset(
        sorted(path),
        concat_dim="example",
        combine="nested",
    )
    dataset["example"] = ("example", np.arange(dataset.sizes["example"]))
    dataset = dataset.sel({"wavelength": slice(WAVELENGTH[0], WAVELENGTH[-1])})

    # build or select input (x) and output (y) variables
    data_vars = list(dataset)
    dataset["x"] = dataset["rrs"].astype("float32")
    dataset["y"] = dataset["phy"].astype("float32")
    dataset = dataset.drop_vars(data_vars)

    # pseudo-random labeled data split
    split = rng.choice(len(p), size=dataset.sizes["example"], p=p)
    dataset["split"] = ("example", split)

    return tuple(convert_dataset(dataset, i, batch_size) for i in range(len(p)))
