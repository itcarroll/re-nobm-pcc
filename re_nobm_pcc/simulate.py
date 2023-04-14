import logging

import numpy as np
import pandas as pd
import xarray as xr

from oasim import modlwn1nm, rrs1nm


# variable names, order matters
phytop = ['dia', 'chl', 'cya', 'coc', 'pha', 'din']
biogeo = ['tot', 'dtc', 'pic', 'cdc', 't', 's']
numnan = np.array(9.99e11, dtype='f4')


def read_nobm_day(year: int, month: int) -> xr.Dataset:
    # container dataset with coordinates
    start = f'{year}-{month:02}-01'
    if month == 12:
        end = f'{year + 1}-01-01'
    else:
        end = f'{year}-{month + 1:02}-01'
    ds = xr.Dataset(
        coords={
            'date': pd.date_range(start, end, freq='D', inclusive='left'),
            'lon': np.arange(288),
            'lat': np.arange(234),
            },
        )
    shape = (ds.dims['lon'], ds.dims['lat'])
    # read all variables
    for item in phytop + biogeo:
        with open(f'data/nobm_day/{item}/{item}{year}{month:02}', 'rb') as stream:
            da = []
            for _ in ds.groupby('date.day'):
                # read start record size
                size = np.fromfile(stream, 'i4', 1).reshape(())
                # skip size bytes of unknown purpose
                stream.seek(size, 1)
                # skip end record size
                stream.seek(4, 1)
                # assert size == np.fromfile(stream, 'i4', 1).reshape(())
                # read start record size
                size = np.fromfile(stream, 'i4', 1).reshape(())
                # read first layer of NOBM output
                da.append(
                    np.fromfile(stream, 'f4', size // 4)
                    .reshape(shape, order='F')
                    )
                # skip end record size
                stream.seek(4, 1)
                # assert size == np.fromfile(stream, 'i4', 1).reshape(())
                # skip remaining 13 layers bytes
                stream.seek((4 + size + 4)*13, 1)
        ds[item] = xr.DataArray(np.stack(da), dims=('date', 'lon', 'lat'))
    phy = ds[phytop]
    ds = ds.drop_vars(phytop)
    ds['phy'] = phy.to_array(dim='component').transpose(..., 'component')
    # tot has an odd NaN flag
    da = ds['tot']
    ds['tot'] = da.where(da != np.float32(5.9939996e12))
    # everything else uses the same NaN flag
    ds = ds.where(ds != numnan)
    return ds


def main(argv: list[str]) -> None:

    # logging
    logging.basicConfig(level=logging.INFO)

    # cast argv to year and month
    idx = int(argv[1])
    year = 1998 + (idx // 12)
    month = 1 + (idx % 12)
    logging.info(f'month is {year}-{month:02}')

    # load nobm model results
    ds = read_nobm_day(year, month)
    logging.info(f'read nobm day')

    # calculate remote sensing reflectance (rrs)
    rrs = []
    for _, value in ds.groupby('date.day'):
        value = value.squeeze('date')
        rlwn = modlwn1nm(*[value[i].data for i in ['phy'] + biogeo])
        rrs.append(rrs1nm(rlwn))
    rrs = xr.DataArray(
        np.stack(rrs),
        coords={'wavelength': np.arange(250, 751)},
        dims=('date', 'lon', 'lat', 'wavelength'),
        )
    ds['rrs'] = rrs.where(rrs != numnan, np.nan)
    logging.info(f'calculated rrs')

    # save to file
    path = f'data/rrs_day/rrs{year}{month:02}.nc'
    ds.to_netcdf(path)
    logging.info(f'wrote to {path}')


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
