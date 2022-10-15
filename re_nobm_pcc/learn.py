import json

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .kit import DATA_DIR, TAXA


BATCH = 128
LEARNING_RATE = 0.001
EPOCHS = 500


if __name__ == '__main__':
    ## load datasets
    # TODO does batching the validate and test Datasets hit performance?
    train = (
        tf.data.Dataset.load(str(DATA_DIR/'train'))
        .shuffle(10*BATCH)
        .batch(BATCH)
    )
    validate = tf.data.Dataset.load(str(DATA_DIR/'validate'))
    validate = validate.batch(validate.cardinality())
    test = tf.data.Dataset.load(str(DATA_DIR/'test'))
    test = test.batch(test.cardinality())
    ## compute loss weights
    # TODO why doesn't Normalization work?
    _, y = next(test.as_numpy_iterator())
    y = np.stack(y, axis=1)
    weights = (1/y.mean(axis=0)).tolist()
    ## build model
    x = tf.keras.Input(shape=train.element_spec[0].shape[1:])
    y = tf.keras.layers.Normalization()
    y.adapt(train.map(lambda x, _: x))
    y = y(x)
    y = tf.keras.layers.Conv1D(filters=16, kernel_size=7, activation='relu')(y)
    y = tf.keras.layers.MaxPooling1D(pool_size=3)(y)
    y = tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu')(y)
    y = tf.keras.layers.MaxPooling1D(pool_size=3)(y)
    y = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(y)
    y = tf.keras.layers.MaxPooling1D(pool_size=3)(y)
    y = tf.keras.layers.Dense(units=64, activation='relu')(y)
    y = [
        tf.keras.layers.Dense(units=1, activation='exponential', name=i)(y)
        for i in TAXA
    ]
    model = tf.keras.Model(inputs=[x], outputs=y)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanAbsoluteError(),
        loss_weights=weights,
        # run_eagerly=True, # DEBUG
    )
    ## fit and save
    fit = model.fit(
        train,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
            ),
        ],
        validation_data=validate,
    )
    model.save(str(DATA_DIR/'model'))
    np.savez(DATA_DIR/'fit.npz', epoch=fit.epoch, **fit.history)
    ## write evaluation metrics
    model.compile(
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
    metrics = model.evaluate(test)
    metrics = {
        k.name: v for k, v in zip(model.metrics, metrics)
    }
    with (DATA_DIR/'metrics.json').open('w') as stream:
        json.dump(metrics, stream)
