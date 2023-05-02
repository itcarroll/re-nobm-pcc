import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from . import DATA_DIR

BATCH = 64
EPOCHS = 300
PATIENCE = 10
DIAG_SHIFT = 1e-5 # TODO working? avoidable?
LEARNING_RATE = 3e-6 # TODO possible to speed up?


def main(args: list[str] | None = None) -> None:

#    tf.config.run_functions_eagerly(True) # DEBUG

    # set Python, Numpy, Tensorflow seeds
    tf.keras.utils.set_random_seed(1241551894)

    # ## load dataset
    # batching adapt necessary for correct sequential input shape
    dataset = tfds.builder('rrs_day_tfds', data_dir=DATA_DIR)
    train, validate = dataset.as_dataset(
        split=['split[:8%]', 'split[8%:9%]'], # DEBUG
        as_supervised=True,
        shuffle_files=True, # TODO breaks determinism?
    )
    train = train.cache()
    validate = validate.cache()
    train = train.shuffle(BATCH * 8).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    validate = validate.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    adapt = train.map(lambda x, _: x)

    # ## Multiple GPUS
    # TODO this was slower ... but GPU:1 is slightly used even without
    # TODO okay, GPU is just plain slower. why?
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

    # ## preprocessing layer
    normalization = tf.keras.layers.Normalization()
    # normalization.adapt(adapt) # DEBUS

    # ## probability model layer
    event_size, *_ = dataset.info.features['phy'].shape
    params_size = event_size + event_size * (event_size + 1) // 2
    fill_tril = tfp.bijectors.FillScaleTriL(
        diag_bijector=tfp.bijectors.Exp(),
        diag_shift=DIAG_SHIFT,
    )
    model = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.MultivariateNormalTriL(
            loc=x[:, :event_size],
            scale_tril=fill_tril(x[:, event_size:]),
            # validate_args=True, # DEBUG
        ),
    )

    # ## neural network via the sequential api
    network = tf.keras.Sequential([
        # normalization, # FIXME i think this causes problems
        tf.keras.layers.Dense(64, 'relu'),
        tf.keras.layers.Dense(params_size, 'linear'),
        model,
    ])

    # ## optimization
    network.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=lambda y, model: tf.reduce_sum(-model.log_prob(y)),
    )

    # ## fit and save
    # optimize parameters
    fit = network.fit(
        train,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=DATA_DIR / 'fit/epoch-{epoch}',
                save_weights_only=True,
            ),
        ],
        validation_data=validate,
    )

    # save to custom tf model format
    network.save(str(DATA_DIR/'network'))
    # save training history to Numpy archive
    np.savez(DATA_DIR/'fit.npz', epoch=fit.epoch, **fit.history)

    # TODO add metrics like individual R2s, likelhihood
    metrics = {}
    with (DATA_DIR/'metrics.json').open('w') as stream:
        json.dump(metrics, stream)


if __name__ == '__main__':
    main()
