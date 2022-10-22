from cmath import isnan
import json

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .kit import DATA_DIR, TAXA

BATCH = 256
EPOCHS = 2
PATIENCE = 50
LEARNING_RATE = 0.001
PRESENCE_PREDICTION_THRESHOLD = 0.0


if __name__ == '__main__':
    ## load datasets
    train = (
        tf.data.Dataset.load(str(DATA_DIR/'train'))
        .shuffle(16*BATCH)
        .batch(BATCH)
    )
    validate = tf.data.Dataset.load(str(DATA_DIR/'validate'))
    validate = validate.batch(BATCH)
    test = tf.data.Dataset.load(str(DATA_DIR/'test'))
    test = test.batch(BATCH)
    ## compute loss weights # FIXME weights
    # # TODO how to use Normalization for this Dataset?
    # y = [np.stack(i) for _, i in train.as_numpy_iterator()]
    # y = np.concatenate(y, axis=1)
    # weights = (1/y.mean(axis=1)).tolist()
    ## build model
    # single input with normalization
    x = tf.keras.Input(shape=train.element_spec[0].shape[1:])
    y = tf.keras.layers.Normalization()
    y.adapt(train.map(lambda x, _: x))
    y = y(x)
    # sequential layers
    y = tf.keras.layers.Dense(32, 'swish')(y)
    # multiple outputs for 1) different taxa and 2) presence and abundance
    outputs = []
    compile_kwargs = {
        'loss': {},
        # 'loss_weights': {}, # FIXME weights
    }
    for i, item in enumerate(TAXA):
        # compile_kwargs['loss_weights'][name] = weights[i] # FIXME weights
        name = f'abundance_{item}'
        y_i = tf.keras.layers.Dense(1, activation='softplus', name=name)(y)
        outputs.append(y_i)
        compile_kwargs['loss'][name] = tf.keras.losses.MeanAbsoluteError()
        # compile_kwargs['loss_weights'][name] = weights[i] #FIXME weights
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
