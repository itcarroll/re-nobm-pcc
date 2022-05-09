import numpy as np
import tensorflow as tf

from .kit import PROJECT_DIR
from .preprocess import train, validate


TENSORBOARD_LOGS_DIR = PROJECT_DIR/'models'/'logs'/'perceptron'
CHECKPOINTS_DIR = PROJECT_DIR/'models'/'checkpoints'/'perceptron'

rng = np.random.default_rng(seed=6776103101287072045)
tf.random.set_seed(rng.integers(np.iinfo(np.int64).max))


class Full(tf.keras.Model):

    def __init__(self, **kwargs):
        input = tf.keras.Input(shape=(train.sizes['wavelength'],))
        output = input
        for _ in range(2):
            layer = tf.keras.layers.Dense(units=1024, activation=tf.nn.elu)
            output = layer(output)
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
            x=train['x'].values,
            y=train['y'].values,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=32,
                ),
                tf.keras.callbacks.TensorBoard(TENSORBOARD_LOGS_DIR/'full'),
            ],
            validation_data=(validate['x'].values, validate['y'].values),
            **kwargs,
        )


class Reduced(tf.keras.Model):

    def __init__(self, **kwargs):
        input1 = tf.keras.Input(shape=(1,))
        dense1 = tf.keras.layers.Dense(
            units=train.sizes['component'],
            use_bias=False,
        )
        output1 = dense1(input1)
        functional = super().__init__(
            inputs=[input1],
            outputs=[output1],
            **kwargs,
        )
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        return functional

    def fit(self, train, validate, **kwargs):
        return super().fit(
            x=tf.ones((train.sizes['pxl'],)),
            y=train['y'].values,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                ),
                tf.keras.callbacks.TensorBoard(TENSORBOARD_LOGS_DIR/'reduced'),
            ],
            validation_data=(
                tf.ones((validate.sizes['pxl'],)),
                validate['y'].values,
            ),
            **kwargs,
        )


if __name__ == '__main__':

    SIZES = 256
    EPOCHS = 100
    Full().fit(train, validate, batch_size=SIZES, epochs=EPOCHS, verbose=0)
    Reduced().fit(train, validate, batch_size=SIZES, epochs=EPOCHS, verbose=0)


# TODO checkpoints for weights
# TODO learning rate decay
#     lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     0.001,
#     decay_steps=STEPS_PER_EPOCH*1000,
#     decay_rate=1,
#     staircase=False,
#     )
