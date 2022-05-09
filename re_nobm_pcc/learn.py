import json

import numpy as np
import tensorflow as tf

from .kit import DATA_DIR


BATCH = 256
EPOCHS = 2


class PCC(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self._layers = (
                tf.keras.layers.Dense(units=16, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=4),
            )

    def call(self, inputs):
        outputs = inputs
        for item in self._layers:
            outputs = item(outputs)
        return outputs


if __name__ == '__main__':
    ## load datasets
    # TODO does batching the validate and test Datasets hit performance?
    train = tf.data.experimental.load(str(DATA_DIR/'train')).batch(BATCH)
    validate = tf.data.experimental.load(str(DATA_DIR/'validate')).batch(BATCH)
    test = tf.data.experimental.load(str(DATA_DIR/'test')).batch(BATCH)
    ## compile model
    model = PCC()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        run_eagerly=True, # DEBUG
    )
    ## fit and save
    fit = model.fit(
        train,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
            ),
        ],
        validation_data=validate,
    )
    model.save(str(DATA_DIR/'model'))
    np.savez(DATA_DIR/'fit.npz', epoch=fit.epoch, **fit.history)
    ## write evaluation metrics
    # TODO residuals
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            # TODO R2
        ],
    )
    metrics = {
        k: v for k, v in zip(['MSE', 'MAE'], model.evaluate(test))
    }
    with open(DATA_DIR/'metrics.json', 'w') as stream:
        json.dump(metrics, stream)
