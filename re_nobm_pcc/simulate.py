from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import dask.bag as db
import dask.diagnostics

from . import DATADIR, NUMNAN, OC
from .core import read_nobm, oasim_param

if TYPE_CHECKING:
    import pathlib


def main() -> None:

    # run the oasim calculation (includes writing to zarr)
    period = np.arange("1998-01", "2022-01", dtype=np.dtype("datetime64[M]"))
    bag = db.from_sequence(period, 12)
    with dask.diagnostics.ProgressBar():
        bag.map(oasim).compute()


def oasim(period: np.datetime64, path: "pathlib.Path" = DATADIR, **kwargs) -> None:

    # import f2py modules here, b/c they cannot be pickled by dask
    from oasim_rrs import modlwn1nm, rrs1nm

    # load inputs
    nobm = read_nobm(period, **kwargs)

    # loop through output calculations
    rrs = []
    # change to folder containing "oasim_param" as it's hardcoded in oasim
    with oasim_param():
        for _, value in nobm.groupby("date.day"):
            value = value.squeeze("date")
            rlwn = modlwn1nm(*[value[i].data for i in ("phy",) + OC])
            rrs.append(rrs1nm(rlwn))

    # combine into a monthly dataset
    rrs = xr.DataArray(
        np.stack(rrs),
        coords={"wavelength": np.arange(250, 751, dtype="int32")},
        dims=("date", "lon", "lat", "wavelength"),
    )
    rrs = rrs.where(rrs != NUMNAN)
    dataset = xr.merge((xr.Dataset({"rrs": rrs}), nobm["t"].coords))

    # save outputs
    yearmonth = period.item().strftime("%Y%m")
    output = path / "oasim" / f"rrs{yearmonth}.nc"
    output.parent.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(output)


if __name__ == "__main__":
    main()
