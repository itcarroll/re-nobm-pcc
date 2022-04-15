import dask
import numpy as np
import xarray as xr

from .kit import PROJECT_DIR


DATA_DIR = PROJECT_DIR/'data'

rng = np.random.default_rng(seed=7033760348669894684)

PhytoChl = xr.open_mfdataset(
    DATA_DIR.glob('monthly/mon2007*.R2014.nc4'),
    decode_times=False,
    mask_and_scale=True,
)
PhytoChl = PhytoChl.sortby('time')
PhytoChl['time'] = ('time', np.arange(1, 13))

HyperLwn = xr.open_dataarray(
    DATA_DIR /'HyperLwn.R2014.nc4',
    decode_times=False,
    mask_and_scale=False,
    chunks={},
)
HyperLwn = HyperLwn.where(
    HyperLwn != np.array(9.99e11, dtype=np.float32),
)
HyperLwn['wavelength'] = ('wavelength', np.arange(250, 776))
HyperLwn['lat'] = ('lat', PhytoChl['lat'].data)
HyperLwn = HyperLwn.roll(lon=HyperLwn.sizes['lon']//2)
HyperLwn['lon'] = ('lon', PhytoChl['lon'].data)
HyperLwn = HyperLwn.rename(months='time')
HyperLwn['time'] = ('time', PhytoChl['time'].data)

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    HyperLwn = HyperLwn.sortby('lon')
    dataset = (
        xr.merge(
            (
                (
                    PhytoChl
                    .stack({'pxl': ('time', 'lon', 'lat')})
                    .reset_index('pxl')
                    .get(['chl', 'dia', 'coc', 'cya'])
                    .to_array(dim='component', name='y')
                    .transpose()
                ),
                (
                    HyperLwn
                    .stack({'pxl': ('time', 'lon', 'lat')})
                    .drop('pxl')
                    .rename('x')
                    .transpose()
                ),
            ),
        )
        .dropna('pxl')
    )

pxl = dataset.sizes['pxl']
sample = dataset.isel(
    pxl=rng.random(size=pxl)<(1e3/pxl)
)
split = xr.DataArray(
    rng.choice((0, 1, 2), p=(0.5, 0.3, 0.2), size=pxl),
    dims='pxl',
)
train, validate, test = iter(
    i for _, i in dataset.groupby(split)
)
