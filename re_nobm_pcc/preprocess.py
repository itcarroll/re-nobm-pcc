import itertools
import os

import dask
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import xarray as xr

from . import DATA_DIR, TAXA, WAVELENGTH


def load_tensors(split: str) -> callable:
    """load modelled data for 1998-2019, holding 2020-2021 for analysis

    the function returns a tensorflow.data.Dataset.from_generator object
    that returns a train, validate, or test subset
    """

    # the generator needs the same random seed to re-use splits
    random = np.random.default_rng(seed=6788239663313185733)

    def generator() -> tuple:
        for i, j in itertools.product(range(1), range(1)): ## 22, 12
            year = 1998 + i
            month = 1 + j
            file = DATA_DIR / 'rrs_day' / f'rrs{year}{month:02}.nc'
            ds = (
                xr.open_dataset(file, chunks={})
                .stack({'pxl': ('date', 'lon', 'lat')})
            )
            category = random.choice(
                a=['train', 'validate', 'test'],
                p=(0.8, 0.1, 0.1),
                size=ds.sizes['pxl'],
            )
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = ds[{'pxl': category == split}]
                ds = (
                    ds[['rrs', 'phy']]
                    .loc[{'wavelength': slice(WAVELENGTH[0], WAVELENGTH[-1])}]
                    .dropna('pxl')
                    .transpose('pxl', ...)
                )
                predictor = ds['rrs'].values
                response = ds['phy'].values
            yield {'rrs': predictor, 'phy': response}

    # the `None` dimension is necessary for variable batch sizes
    spec = {
        'rrs': tf.TensorSpec((None, len(WAVELENGTH)), dtype=np.float32),
        'phy': tf.TensorSpec((None, len(TAXA)), dtype=np.float32),
    }

    return tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=spec,
    )


def main(
        argv: list[str] | None = None,
    ) -> None:
    # FIXME write as tfds, but still have to resolve split and shuffle
    # but reproducibility looks okay ....
    # map with category would read data multiple times, different rands

    # ## prepare a tf.data.Dataset from files
    # FIXME why is this at most 300% CPU?
    train = load_tensors('train').unbatch()
    validate = load_tensors('validate').unbatch()
    test = load_tensors('test').unbatch()

    features = tfds.features.FeaturesDict({
        'rrs': tfds.features.Tensor(shape=(len(WAVELENGTH),), dtype=np.float32),
        'phy': tfds.features.Tensor(shape=(len(TAXA),), dtype=np.float32),
    })
    tfds.dataset_builders.store_as_tfds_dataset(
        name='tfds_rrs_day',
        version='0.1.0',
        data_dir='data',
        split_datasets={'train': train, 'validate': validate, 'test': test},
        features=features,
    )

    return

    # path = DATA_DIR / 'dataset.tfrecord'
    # # TODO using random numbers for shards seems weird ...
    # random = tf.random.Generator.from_seed(
    #     np.array([6788239663313185733, 5594829877522150623], dtype=np.int64)
    # )
    # dataset.save(
    #     path=str(path),
    #     shard_func=lambda *i: random.uniform((), 0, 512, tf.dtypes.int64),
    # )

    # # ## a TFRecord workaround
    # # remove reference to run_id and creation_timestamp for reproducibility
    # new_run_id = '0'
    # new_message = SnapshotMetadataRecord(run_id=new_run_id)
    # with open(path / 'snapshot.metadata', 'rb') as stream:
    #     old_message = SnapshotMetadataRecord.FromString(stream.read())
    # run_id = next(
    #     (v for k, v in old_message.ListFields() if k.name == 'run_id')
    # )
    # old_message.ClearField('run_id')
    # old_message.ClearField('creation_timestamp')
    # new_message.MergeFrom(old_message)
    # os.rename(path / run_id, path / new_run_id)
    # with open(path / 'snapshot.metadata', 'wb') as stream:
    #     stream.write(new_message.SerializeToString())

    # TODO don't create sample.nc (use tensorflow dataset API instead, how?)
    #     pxl = dataset.sizes['pxl']
    #     sample = dataset.isel({
    #         'pxl': np.sort(rng.choice(pxl, replace=False, size=num))
    #     })

    # TODO figure out how to split tf.data.Dataset, rather than
    # split = xr.DataArray(
    #     rng.choice(('train', 'validate', 'test'), p=(0.6, 0.2, 0.2), size=pxl),
    #     dims='pxl',
    # )
    # for k, v in dataset.groupby(split):
    #    ...

    # TODO an old line about thresholding, if needed again
    # idx = v > PRESENT_ABOVE

    # TODO the labels, for taxa specific weights, if multivariate doesn't work
    # dict(
    #     **{f'abundance_{i}': v[i].where(idx[i], 0.0) for i in TAXA},
    # ),


if __name__ == '__main__':
    main()
