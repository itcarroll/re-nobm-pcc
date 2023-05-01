import itertools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import xarray as xr

from . import DATA_DIR, TAXA, WAVELENGTH

class RrsDayTfds(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        keys = {'rrs': len(WAVELENGTH), 'phy': len(TAXA)}
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                k: tfds.features.Tensor(shape=(v,), dtype=np.float32)
                for k, v in keys.items()
            }),
            supervised_keys=tuple(keys),
        )

    def _split_generators(self, *args):
        return {
            'train': self._generate_examples(path=DATA_DIR / 'rrs_day')
        }

    def _generate_examples(self, path):
        for i, item in enumerate(path.iterdir()):
            if i % 13: # DEBUG
                continue
            dataset = self.load_tensors(item)
            for j, jtem in enumerate(dataset.as_numpy_iterator()):
                yield f'{item.name}-{j}', jtem

    def load_tensors(self, path):
        vars = list(self.info.features)
        ds = (
            xr.open_dataset(path)
            [vars]
            .loc[{'wavelength': slice(WAVELENGTH[0], WAVELENGTH[-1])}]
            .stack({'pxl': ('date', 'lon', 'lat')})
            .dropna('pxl')
            .transpose('pxl', ...)
        )
        return tf.data.Dataset.from_tensor_slices({i: ds[i].values for i in vars})


# def load_tensors() -> tf.data.Dataset:
#     """load modelled data for 1998-2019, reserving 2020-2021 for analysis
#     """
#     vars = {'rrs': len(WAVELENGTH), 'phy': len(TAXA)}

#     def generator() -> tuple:
#         for i, j in itertools.product(range(1), range(1)): ## 22, 12
#             year = 1998 + i
#             month = 1 + j
#             file = DATA_DIR / 'rrs_day' / f'rrs{year}{month:02}.nc'
#             ds = (
#                 xr.open_dataset(file)
#                 [list(vars)]
#                 .loc[{'wavelength': slice(WAVELENGTH[0], WAVELENGTH[-1])}]
#                 .stack({'pxl': ('date', 'lon', 'lat')})
#                 .dropna('pxl')
#                 .transpose('pxl', ...)
#             )
#             yield {i: ds[i].values for i in vars}

#     # the `None` dimension is necessary for variable batch sizes
#     spec = {
#         k: tf.TensorSpec((None, v), dtype=np.float32)
#         for k, v in vars.items()
#     }
#     features = tfds.features.FeaturesDict({
#         k: tfds.features.Tensor(shape=(v,), dtype=np.float32)
#         for k, v in vars.items()
#     })
#     dataset = tf.data.Dataset.from_generator(
#         generator=generator,
#         output_signature=spec,
#     )
#     return dataset, features


def main(
        argv: list[str] | None = None,
    ) -> None:

    # TODO or use tf.data.Dataset.window and shuffle

    # ## prepare a tf.data.Dataset from files
    builder = TfdsRrsDay(data_dir='data-test')
    builder.download_and_prepare()

    # dataset, features = load_tensors()
    # dataset = dataset.unbatch()

    # # FIXME why not parallel?
    # # TODO is this seed having any effect?
    # tf.random.set_seed(3187705664850833354)
    # tfds.disable_progress_bar()
    # tfds.dataset_builders.store_as_tfds_dataset(
    #     name='tfds_rrs_day',
    #     version='0.1.0',
    #     data_dir='data-test', # DEBUG
    #     split_datasets={'unsplit': dataset},
    #     features=features,
    # )

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
