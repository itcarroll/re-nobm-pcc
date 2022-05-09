import numpy as np
import tensorflow as tf

from .kit import PROJECT_DIR
from .preprocess import train, validate


TENSORBOARD_LOGS_DIR = PROJECT_DIR/'models'/'logs'/'cnn'
CHECKPOINTS_DIR = PROJECT_DIR/'models'/'checkpoints'/'cnn'

rng = np.random.default_rng(seed=196125464453949136)
tf.random.set_seed(rng.integers(np.iinfo(np.int64).max))


class Full(tf.keras.Model):

    def __init__(self, **kwargs):
        # input
        input = tf.keras.Input(shape=(train.sizes['wavelength'], 1))
        output = input
        # hidden layer
        layer = tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=5,
            activation=tf.nn.elu,
        )
        output = layer(output)
        layer = tf.keras.layers.MaxPooling1D(pool_size=3)
        output = layer(output)
        # hidden layer
        layer = tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation=tf.nn.elu,
        )
        output = layer(output)
        layer = tf.keras.layers.MaxPooling1D(pool_size=3)
        output = layer(output)
        # hidden layer
        layer = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation=tf.nn.elu,
        )
        output = layer(output)
        layer = tf.keras.layers.MaxPooling1D(pool_size=3)
        output = layer(output)
        # hidden layer
        layer = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation=tf.nn.elu,
        )
        output = layer(output)
        layer = tf.keras.layers.Flatten()
        output = layer(output)
        # hidden layer
        layer = tf.keras.layers.Dense(units=64, activation=tf.nn.elu)
        output = layer(output)
        # output
        layer = tf.keras.layers.Dense(units=train.sizes['component'])
        output = layer(output)
        functional = super().__init__(
            inputs=[input],
            outputs=[output],
            **kwargs,
        )
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        return functional

    def fit(self, train, validate, **kwargs):
        return super().fit(
            x=tf.expand_dims(train['x'].values, -1),
            y=train['y'].values,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=16,
                    min_delta=10**-4,
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=TENSORBOARD_LOGS_DIR/'full',
                    histogram_freq=1,
                ),
            ],
            validation_data=(
                tf.expand_dims(validate['x'].values, -1),
                validate['y'].values,
            ),
            **kwargs,
        )
