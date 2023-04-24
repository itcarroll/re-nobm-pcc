import itertools
import os

from tensorflow.core.protobuf.snapshot_pb2 import SnapshotMetadataRecord
import numpy as np
import tensorflow as tf
import xarray as xr

from . import DATA_DIR, TAXA, WAVELENGTH


def load_tensors():
    """load modelled data for 1998-2019, save 2020-2021 for subsequent analysis
    """
    for i, j in itertools.product(range(22), range(12)): ## 22, 12
        year = 1998 + i
        month = 1 + j
        ds = (
            xr.open_dataset(DATA_DIR / 'rrs_day' / f'rrs{year}{month:02}.nc')
            [['rrs', 'phy']]
            .loc[{'wavelength': slice(WAVELENGTH[0], WAVELENGTH[-1])}]
            .stack({'pxl': ('date', 'lon', 'lat')})
            .dropna('pxl')
            .transpose('pxl', ...)
        )
        predictor = ds['rrs'].values
        response = ds['phy'].values
        yield (predictor, response)


def main(argv: list[str]) -> None:

    # ## prepare a tf.data.Dataset from files
    # FIXME why is this at most 300% CPU?
    dataset = tf.data.Dataset.from_generator(
        generator=load_tensors,
        output_signature=(
            tf.TensorSpec((None, len(WAVELENGTH)), dtype=np.float32),
            tf.TensorSpec((None, len(TAXA)), dtype=np.float32),
        ),
    )
    dataset = dataset.unbatch()
    path = DATA_DIR / 'dataset.tfrecord'
    # TODO using random numbers for shards seems weird ...
    random = tf.random.Generator.from_seed(
        np.array([6788239663313185733, 5594829877522150623], dtype=np.int64)
    )
    dataset.save(
        path=str(path),
        shard_func=lambda *i: random.uniform((), 0, 512, tf.dtypes.int64),
    )

    # ## a TFRecord workaround
    # remove reference to run_id and creation_timestamp for reproducibility
    new_run_id = '0'
    new_message = SnapshotMetadataRecord(run_id=new_run_id)
    with open(path / 'snapshot.metadata', 'rb') as stream:
        old_message = SnapshotMetadataRecord.FromString(stream.read())
    run_id = next(
        (v for k, v in old_message.ListFields() if k.name == 'run_id')
    )
    old_message.ClearField('run_id')
    old_message.ClearField('creation_timestamp')
    new_message.MergeFrom(old_message)
    os.rename(path / run_id, path / new_run_id)
    with open(path / 'snapshot.metadata', 'wb') as stream:
        stream.write(new_message.SerializeToString())

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
    import sys
    sys.exit(main(sys.argv))
