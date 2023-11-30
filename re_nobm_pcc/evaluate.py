import json
import tensorflow as tf
import tensorflow_addons as tfa  #

from . import DATADIR


BATCH = 64


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


def main(args: list[str] | None = None) -> None:
    if __debug__:
        tf.config.run_functions_eagerly(True)

    # ## calculate metrics
    network = add_metrics(network)
    evaluation = network.evaluate(test, verbose=1 if __debug__ else 0)
    items = zip(network.metrics, evaluation)
    metrics.update({k.name: v for k, v in items if not "loss" in k.name})
    with (DATADIR / "metrics.json").open("w") as stream:
        json.dump(metrics, stream)


if __name__ == "__main__":
    main()  # FIXME this is all copypasta
