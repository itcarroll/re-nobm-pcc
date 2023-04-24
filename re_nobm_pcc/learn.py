from cmath import isnan
import json

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# import tensorflow_addons as tfa

from . import DATA_DIR, TAXA

BATCH = 256
EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 3e-4
PRESENCE_PREDICTION_THRESHOLD = 0.0


def loss(y_obs, y_hat):
    """negative log-likelihood. pass the covariance as a vector of the
       components of the Cholesky decomposition, in order compatible with
       `tensorflow_probability.math.fill_triangular`.
    """
    # Multivariate Normal:
    #   Det(Sigma) + (x - Mu)^T @ Sigma^-1 @ (x - Mu)
    mu, vec_cholesky_sigma = y_hat
    cholesky_sigma = tfp.math.fill_triangular(vec_cholesky_sigma)
    y_obs_centered = y_obs - mu
    return (
        2 * tf.reduce_sum(
            tf.math.log(tf.math.abs(tf.linalg.diag_part(cholesky_sigma)))
        ) +
        tf.tensordot(
            y_obs_centered,
            tf.squeeze(
                tf.linalg.cholesky_solve(
                    cholesky_sigma,
                    y_obs_centered[..., tf.newaxis],
                )
            ),
            axes=1,
        )
    )


def main(args: list[str] | None = None) -> None:

    # ## load dataset
    dataset = tf.data.Dataset.load(str(DATA_DIR / 'dataset.tfrecord'))

    # ## preprocessing layers
    normalization = tf.keras.layers.Normalization()
    normalization.adapt(dataset.map(lambda x, y: x).take(10 ** 1)) ## DEBUG

    # ## neural network layers
    predictors = tf.keras.Input(shape=dataset.element_spec[0].shape)
    layer = normalization(predictors)
    layer = tf.keras.layers.Dense(64, 'relu')(layer)
    layer = tf.keras.layers.Dense(64, 'relu')(layer)
    layer = tf.keras.layers.Dense(64, 'relu')(layer)
    means = tf.keras.layers.Dense(6, 'relu')(layer)
    covariance = tf.keras.layers.Dense(6, 'linear')(layer)
    model = tf.keras.Model(
        inputs=(predictors),
        outputs=(means, covariance),
    )

    # ## optimization
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        run_eagerly=True, # DEBUG
    )

    # ## dataset splits
    # nb: elements are (more-or-less) ordered in time, but random in space
    dataset = tf.data.Dataset.load(str(DATA_DIR / 'dataset.tfrecord'))
    train, dataset = tf.keras.utils.split_dataset(dataset, 0.6)
    validate, test = tf.keras.utils.split_dataset(dataset, 0.5)

    # ## fit and save
    # optimize parameters
    fit = model.fit(
        train.batch(BATCH),
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
            ),
        ],
        validation_data=validate,
    )
    # save to custom tf model format
    model.save(str(DATA_DIR/'model'))
    # save training history to Numpy archive
    np.savez(DATA_DIR/'fit.npz', epoch=fit.epoch, **fit.history)
    # TODO add metrics like individual R2s, likelhihood


if __name__ == '__main__':
    main()

    # TODO everything below is deprecated
    import sys
    sys.exit()
    # ## load datasets
    train = (
        tf.data.Dataset.load(str(DATA_DIR/'train'))
        .shuffle(16*BATCH)
        .batch(BATCH)
    )
    validate = tf.data.Dataset.load(str(DATA_DIR/'validate'))
    validate = validate.batch(BATCH)
    test = tf.data.Dataset.load(str(DATA_DIR/'test'))
    test = test.batch(BATCH)
    ## compute loss weights
    # TODO how to use Normalization for a structured Dataset?
    y = [
        np.stack(tuple(i[f'abundance_{j}'] for j in TAXA))
        for _, i in train.as_numpy_iterator()
    ]
    y = np.concatenate(y, axis=1)
    weights = (1/y.mean(axis=1)).tolist()
    ## build model
    # single input with normalization
    x = tf.keras.Input(shape=train.element_spec[0].shape[1:])
    y = tf.keras.layers.Normalization()
    y.adapt(train.map(lambda x, _: x))
    y = y(x)
    # sequential layers
    y = tf.keras.layers.Dense(32, 'relu')(y)
    # multiple outputs for 1) different taxa and 2) presence and abundance
    outputs = []
    compile_kwargs = {
        'loss': {},
        'loss_weights': {},
    }
    for i, item in enumerate(TAXA):
        name = f'abundance_{item}'
        y_i = tf.keras.layers.Dense(1, activation='softplus', name=name)(y)
        outputs.append(y_i)
        compile_kwargs['loss'][name] = tf.keras.losses.MeanAbsoluteError()
        compile_kwargs['loss_weights'][name] = weights[i]
    model = tf.keras.Model(inputs=[x], outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        **compile_kwargs,
        # run_eagerly=True, # DEBUG
    )
    ## fit and save
    fit = model.fit(
        train,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
            ),
        ],
        validation_data=validate,
    )
    model.save(str(DATA_DIR/'model'))
    np.savez(DATA_DIR/'fit.npz', epoch=fit.epoch, **fit.history)
    ## add metrics for evaluation only
    model.compile(
        **compile_kwargs,
        metrics=[
            tf.keras.metrics.MeanMetricWrapper(
                fn=lambda y_true, y_pred: y_pred - y_true,
                name='ME',
            ),
            tf.keras.metrics.MeanAbsoluteError(name='MAE'),
            tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
            tfa.metrics.RSquare(name='R2'),
        ],
    )
    ## calculate metrics and write to json
    all_metrics = {}
    metrics = model.evaluate(test)
    all_metrics.update({k.name: v for k, v in zip(model.metrics, metrics)})
    with (DATA_DIR/'metrics.json').open('w') as stream:
        json.dump(all_metrics, stream)
