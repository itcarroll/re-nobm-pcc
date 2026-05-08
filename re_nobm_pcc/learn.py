from typing import TYPE_CHECKING
import json

import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow_probability as tfp

from . import DATADIR, CHUNKSIZE

if TYPE_CHECKING:
    import pathlib

EPOCHS = 300
PATIENCE = 10
BATCHSIZE = 64
LEARNING_RATE = 3e-5


def main(epochs: int, path: "pathlib.Path") -> None:

    # prepare data for training
    train, validate = open_dataset(path / "labelled.zarr")
    shape = (i.shape for i in train.element_spec)

    # the untrained model
    model = prepare_model(*shape)

    # model training
    fit = model.fit(
        train,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=path / "checkpoint/epoch-{epoch:03d}",
                save_weights_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(patience=PATIENCE),
        ],
        validation_data=validate,
    )

    # save results
    # network with fitted parameters as keras format
    model.save(str(path / "model.keras"))
    # training history as a dataset
    fit.history["epoch"] = fit.epoch
    fit = xr.Dataset({k: ("epoch", v) for k, v in fit.history.items()})
    fit.to_zarr(path / "fit.zarr")


def prepare_model(input: tf.TensorShape, output: tf.TensorShape) -> tf.keras.Model:

    # parameters for layers input to the tfp.distribution output
    units, shape, tfd = prepare_tfd(output[1:])

    # build model with the Keras Functional API
    input = tf.keras.Input(input[-1:])
    layer = tf.keras.layers.Dense(64, "sigmoid")(input)
    layer = tf.keras.layers.Dense(units, "linear")(layer)
    layer = tf.keras.layers.Reshape(shape)(layer)
    output = tfp.layers.DistributionLambda(tfd)(layer)
    model = tf.keras.Model([input], [output])

    # compile with negative log-likelihood loss
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=nll,
    )

    return model


def prepare_tfd(shape: tf.TensorShape) -> tuple:

    shape = (*shape, 3)
    units = np.prod(shape)

    # function to map final layer to distribution parameters
    def tfd(t):
        distribution = tfp.distributions.Gamma(
            concentration=tf.math.softplus(t[..., 0]),
            log_rate=t[..., 1],
        )
        inflated = tfp.distributions.Inflated(
            distribution,
            inflated_loc_logits=t[..., 2],
        )
        return tfp.distributions.Independent(
            inflated,
            reinterpreted_batch_ndims=1,
        )

    return units, shape, tfd


def nll(y_true, y_pred):
    # negative log-likelihood
    return -y_pred.log_prob(y_true)


def open_dataset(path: "pathlib.Path") -> tuple[tf.data.Dataset]:

    ds = []
    for item in ["train", "validate"]:
        dataset = xr.open_dataset(path, engine="zarr", group=item, chunks={})
        x = dataset["x"]
        y = dataset["y"]
        y = y.where(np.log10(y) > -10, 0)  # nb. kludge, but safe b/c no NaN
        ds.append(tf.data.Dataset.from_tensor_slices((x.data, y.data)))

    train, validate = ds
    train = train.batch(BATCHSIZE).shuffle(CHUNKSIZE).prefetch(tf.data.AUTOTUNE)
    validate = validate.batch(CHUNKSIZE)

    return train, validate


def make_network(event_size: int) -> tf.keras.Model:
    # ## probability model
    model = tfp.layers.MultivariateNormalTriL
    params_size = model.params_size(event_size)
    # TODO worried I'm misuing bijectors, even in tfp.layers
    # fill_tril = tfp.bijectors.FillScaleTriL(
    #     diag_bijector=tfp.bijectors.Exp(),
    #     diag_shift=DIAG_SHIFT,
    # )
    # model = tfp.layers.DistributionLambda(
    #     lambda x: tfp.distributions.MultivariateNormalTriL(
    #         loc=x[..., :event_size],
    #         scale_tril=fill_tril(x[:, event_size:]),
    #     ),
    # )

    # ## neural network via the sequential api
    network = tf.keras.Sequential(
        [
            # tf.keras.layers.Normalization(), # TODO causes problems, BatchNormalization?
            tf.keras.layers.Dense(1024, "relu"),
            tf.keras.layers.Dense(params_size, "linear"),
            model(event_size, tfp.distributions.Distribution.mean),
        ]
    )

    # ## optimization
    network.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=lambda y, model: tf.reduce_sum(-model.log_prob(y)),
    )

    return network


def add_metrics(network: tf.keras.Model) -> tf.keras.Model:
    network.trainable = False
    inputs = network.inputs
    layer = network(inputs[0])
    outputs = tf.keras.layers.Lambda(
        lambda t: tf.unstack(t, axis=-1),
        name="mean",
    )
    outputs = outputs(layer)
    metrics = tf.keras.Model(inputs=inputs, outputs=outputs)
    metrics.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[
            tf.keras.metrics.MeanMetricWrapper(
                fn=lambda y_true, y_pred: y_pred - y_true,
                name="ME",
            ),
            tf.keras.metrics.MeanAbsoluteError(name="MAE"),
            tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
            tfa.metrics.RSquare(name="R2"),
        ],
    )
    return metrics


def _main(args: list[str] | None = None) -> None:
    if __debug__:
        tf.config.run_functions_eagerly(True)

    # set Python, Numpy, Tensorflow seeds
    tf.keras.utils.set_random_seed(1241551894)
    random = np.random.default_rng(9190807622626)

    # ## open datasets
    split = (0.7, 0.2, 0.1)
    if __debug__:
        split = (0.07, 0.02, 0.01, 0.9)
    train, validate, _ = open_dataset(split, random, batch_size=BATCH)

    # batching adapt necessary for correct sequential input shape
    train = train.prefetch(tf.data.AUTOTUNE)
    validate = validate.cache().prefetch(tf.data.AUTOTUNE)
    # FIXME HERE

    # test = test.batch(BATCH).map(lambda x, y: (x, tf.unstack(y, axis=-1)))
    # TODO adapt = validate.map(lambda x, _: x)

    # ## Multiple GPUS
    # TODO this was slower ... but GPU:1 is slightly used even without
    # TODO okay, GPU is just plain slower. why?
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

    # ## fit and save
    # optimize parameters
    network = make_network(dataset.info.features["phy"].shape[0])
    # TODO normalization.adapt(adapt)
    fit = network.fit(
        train,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=DATADIR / "fit/epoch-{epoch:03d}",
                save_weights_only=True,
            ),
        ],
        validation_data=validate,
        verbose=1 if __debug__ else 2,
    )
    # network with fitted parameters as tf format
    network.save(str(DATADIR / "network"))
    # training history as Numpy archive
    np.savez(DATADIR / "fit.npz", epoch=fit.epoch, **fit.history)
    metrics = {i: fit.history[i][-1] for i in ("loss", "val_loss")}

    # ## calculate metrics
    network = add_metrics(network)
    evaluation = network.evaluate(test, verbose=1 if __debug__ else 0)
    items = zip(network.metrics, evaluation)
    metrics.update({k.name: v for k, v in items if not "loss" in k.name})
    with (DATADIR / "metrics.json").open("w") as stream:
        json.dump(metrics, stream)


if __name__ == "__main__":
    main(
        epochs=EPOCHS,
        path=DATADIR,
    )
