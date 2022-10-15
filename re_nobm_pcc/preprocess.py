import dask
import numpy as np
import xarray as xr

from .kit import DATA_DIR, TAXA

# FIXME still getting different sample and shards on each run
rng = np.random.default_rng(seed=7033760348669894684)
num = 1000

PhytoChl = (
    xr.open_mfdataset(
        DATA_DIR.glob('nobm/monthly/mon2007*.R2014.nc4'),
        decode_times=False,
        mask_and_scale=True,
    )
    .rename(time='month')
    .sortby('month')
    .assign_coords(month=('month', np.arange(1, 13)))
    .get(TAXA)
    # .to_array(dim='component', name='labels') # TODO multivariate output rather than multiple outputs
    .transpose('lon', 'lat', 'month', ...)
)

HyperLwn = (
    xr.open_dataarray(
        DATA_DIR/'nobm'/'HyperLwn.R2014.nc4',
        decode_times=False,
        mask_and_scale=False,
        chunks={},
    )
    .rename('features')
)
HyperLwn = (
    HyperLwn
    .where(
        HyperLwn != np.array(9.99e11, dtype=np.float32),
    )
    .assign_coords(wavelength=('wavelength', np.arange(250, 776)))
    .assign_coords(lat=('lat', PhytoChl['lat'].data))
    .roll(lon=HyperLwn.sizes['lon']//2)
    .assign_coords(lon=('lon', PhytoChl['lon'].data))
    .rename(months='month')
    .assign_coords(month=('month', PhytoChl['month'].data))
    .transpose('lon', 'lat', 'month', 'wavelength')
)


if __name__ == '__main__':
    ## imports
    import tensorflow as tf
    ## reshape data
    # stack pixels and merge features to labels
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        dataset = (
            xr.merge(
                (
                    (
                        PhytoChl
                        .stack({'pxl': ('lon', 'lat', 'month')})
                        .reset_index('pxl')
                    ),
                    (
                        HyperLwn
                        .stack({'pxl': ('lon', 'lat', 'month')})
                        .drop('pxl')
                    ),
                ),
            )
            .transpose('pxl', ...)
            .dropna('pxl')
        )
        pxl = dataset.sizes['pxl']
        sample = dataset.isel({
            'pxl': np.sort(rng.choice(pxl, replace=False, size=num))
        })
    ## write outputs
    # write a data sample to file for previewing
    sample.to_netcdf(DATA_DIR/'sample.nc')
    # split the labeled data and write to tf shards
    split = xr.DataArray(
        rng.choice(('train', 'validate', 'test'), p=(0.6, 0.2, 0.2), size=pxl),
        dims='pxl',
    )
    for k, v in dataset.groupby(split):
        ds = tf.data.Dataset.from_tensor_slices((
            v['features'].expand_dims('features', axis=-1),
            # v['labels'], # TODO multivariate output rather than multiple outputs
            tuple(v[i] for i in TAXA),
        ))
        ds.save(str(DATA_DIR/k))
