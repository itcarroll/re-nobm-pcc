from typing import BinaryIO
import logging

import numpy as np
import xarray as xr
from oasim_rrs import modlwn1nm, rrs1nm

from . import DATADIR, TAXA, LONLAT


OC = ("tot", "dtc", "pic", "cdc", "t", "s")
ENV = ("alk", "dic", "doc", "fco", "h", "irn", "pco", "pp", "rnh", "rno", "sil", "zoo")
NUMNAN = np.array(9.99e11, dtype="f4")


def fromfile(file: BinaryIO, shape: tuple[int] | None = (-1,)) -> np.ndarray:
    # read start record size
    size = np.fromfile(file, "i4", 1).reshape(())
    # skip f"{size}" bytes of unknown purpose
    array = np.fromfile(file, "f4", size // 4).reshape(shape, order="F")
    # verify at end of record
    assert size == np.fromfile(file, "i4", 1).reshape(())
    return array


def read_nobm(year: int, month: int) -> xr.Dataset:
    """Read the daily NOBM data files provided by C. Rouseaux"""

    # ## container dataset with coordinates
    start = np.datetime64(f"{year}-{month:02}")
    stop = start + np.timedelta64(1, "M")
    step = np.timedelta64(1, "D")
    ds = xr.Dataset(
        coords={
            "date": np.arange(start, stop, step).astype("datetime64[ns]"),
            "lon": np.arange(LONLAT[0], dtype=np.float32),
            "lat": np.arange(LONLAT[1], dtype=np.float32),
        },
    )
    dims = tuple(ds.dims)
    shape = (ds.dims["lon"], ds.dims["lat"])
    size = np.prod(shape)

    # ## read all variables
    for item in TAXA + OC + ENV:
        with open(DATADIR / f"nobm/{item}/{item}{year}{month:02}", "rb") as f:
            da = []
            for _ in ds.groupby("date.day"):
                # mystery array prepended to some files
                if item not in ["fco", "pco"]:
                    _ = fromfile(f)
                da.append(fromfile(f, shape))
                if item in ["fco", "pco", "pp"]:
                    if item == "pp":
                        for jtem in TAXA:
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
    # set numbers representing nan to nan
    # tot has an odd NaN flag
    da = ds["tot"]
    ds["tot"] = da.where(da != np.float32(5.9939996e12))
    # everything else uses the same NaN flag
    ds = ds.where(ds != NUMNAN)

    return ds


def main(argv: list[str]) -> None:
    # ## logging
    logging.basicConfig(level=logging.INFO)

    # ## arguments
    # cast argv to year and month
    idx = int(argv[1])
    year = 1998 + (idx // 12)
    month = 1 + (idx % 12)
    output = DATADIR / "oasim" / f"{year}{month:02}.nc"

    # ## inputs
    logging.info(f"reading nobm for {year}-{month:02}")
    ds = read_nobm(year, month)

    # ## outputs
    logging.info("calculating remote sensing reflectance (rrs)")
    rrs = []
    for _, value in ds.groupby("date.day"):
        value = value.squeeze("date")
        rlwn = modlwn1nm(*[value[i].data for i in ("phy",) + OC])
        rrs.append(rrs1nm(rlwn))
    rrs = xr.DataArray(
        np.stack(rrs),
        coords={"wavelength": np.arange(250, 751)},
        dims=("date", "lon", "lat", "wavelength"),
    )
    ds["rrs"] = rrs.where(rrs != NUMNAN, np.nan)
    # roll on lon and assign true coordinates
    ds = ds.roll({"lon": ds.sizes["lon"] // 2})
    ds["lon"] = ("lon", np.linspace(-180, 180, ds.sizes["lon"]))
    ds["lat"] = ("lat", np.linspace(-84, 71.4, ds.sizes["lat"]))

    # ## save
    logging.info(f"writing phy and rrs to {output}")
    ds.to_netcdf(output)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
