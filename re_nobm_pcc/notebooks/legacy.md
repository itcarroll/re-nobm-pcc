---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: re-nobm-pcc
    language: python
    name: re-nobm-pcc
---

# Chl-a


- load directly fom netcdf
- fit a linear regressin with least square
- fit a regression with variance estimated too
- make it bayesian


## Imports

```python
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
```

```python
import importlib
import warnings
import datetime as dt

from IPython.display import Markdown
from scipy.stats import zscore
import holoviews as hv
import hvplot.xarray
import numpy as np
import pandas as pd
import panel as pn
import param as p
import xarray as xr

# import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_probability as tfp

# from re_nobm_pcc import preprocess
from re_nobm_pcc import DATADIR, TAXA, WAVELENGTH
from re_nobm_pcc import kit

warnings.filterwarnings(action="ignore", category=FutureWarning)
hv.opts.defaults(
    hv.opts.Curve(active_tools=[]),
    hv.opts.Image(active_tools=[]),
    hv.opts.Scatter(active_tools=[]),
    hv.opts.HexTiles(active_tools=[], tools=["hover"]),
)
```

## Abbreviations

```python
long_name = {
    "alk": "alkalinity",
    "cdc": "colored dissolved carbon",
    "chl": "chlorophytes",
    "coc": "coccolithophores",
    "cya": "cyanobacteria",
    "dia": "diatoms",
    "dic": "dissolved organic carbon",
    "din": "dinoflagellate",
    "doc": "dissolved organic carbon",
    "dtc": "dissolved total carbon",
    "fco": "carbon dioxide flux",
    "h": "mixed layer depth",
    "irn": "iron",
    "pco": "carbon dioxide concentration",
    "pha": "phaeocystis",
    "pic": "particulate inorganic carbon",
    "pp": "phytoplankton primary productivity",
    "tpp": "total primary productivity",
    "rnh": "ammonium",
    "rno": "nitrate",
    "s": "salinity",
    "t": "temperature",
    "tot": "total chlorophyl",
    "zoo": "zooplankton",
}
```

## Chl-a OC4 Algorithm


OC4 (SeaWiFS) from https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/

```python
a = [0.32814, -3.20725, 3.22969, -1.36769, -0.81739]
blue = [443, 489, 510]
green = 555

dim = "wavelength"
da = xr.DataArray(
    np.arange(len(WAVELENGTH)),
    coords={dim: ds[dim].loc[WAVELENGTH[0] : WAVELENGTH[-1]]},
)
blue = da.sel({dim: blue}, method="nearest").values.tolist()
green = da.sel({dim: green}, method="nearest").values.item()

a = tf.expand_dims(tf.constant(a), 1)
blue, green
```

```python
@tf.function
def log_blue_green_ratio(x, y):
    return (
        tf.expand_dims(
            tf.experimental.numpy.log10(
                tf.reduce_max(tf.gather(x, blue, axis=1), axis=1) / x[:, green]
            ),
            axis=1,
        ),
        tf.expand_dims(tf.experimental.numpy.log10(tf.reduce_sum(y, axis=1)), axis=1),
    )


batch_size = 2**10
tfds_rrs_day = tfds.builder("rrs_day_tfds", data_dir=DATA_DIR)
train, test = tfds_rrs_day.as_dataset(
    split=["split[:8%]", "split[9%:10%]"], as_supervised=True
)
train_size = train.cardinality()
test_size = test.cardinality()
train = train.batch(batch_size).cache()
test = test.batch(batch_size).cache()
```

### no retraining

```python
log_x, log_y = test.map(log_blue_green_ratio).rebatch(test_size).get_single_element()
log_y = log_y[:, 0].numpy()
log_y_hat = (log_x ** tf.range(5, dtype=np.float32) @ a)[:, 0].numpy()
```

```python
R2 = 1 - ((log_y - log_y_hat) ** 2).sum() / ((log_y - log_y.mean()) ** 2).sum()
print(f"R2: {R2}")
```

```python
(
    hv.Scatter(
        (log_y_hat[: 2**16], log_y[: 2**16]),
        kdims="prediction",
        vdims="truth",
    )
    + hv.HexTiles(
        (log_y_hat, log_y),
        kdims=["prediction", "truth"],
    ).opts(logz=True)
) * hv.Slope(1, 0).opts(color="black", line_width=1)
```

### re-trained

```python
network = tf.keras.Sequential(
    [
        tf.keras.layers.Lambda(lambda x: x ** tf.range(1, 5, dtype=np.float32)),
        tf.keras.layers.Dense(1),
    ]
)
network.compile(
    optimizer=tf.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.MeanSquaredError(),
)
```

```python
fit = network.fit(
    train.map(log_blue_green_ratio).shuffle(batch_size * 4),
    epochs=10,
)
```

```python
network.layers[1].weights
```

```python
log_x, log_y = test.map(log_blue_green_ratio).rebatch(test_size).get_single_element()
log_y_hat = network(log_x)[:, 0].numpy()
log_y = log_y[:, 0].numpy()
```

```python
R2 = 1 - ((log_y - log_y_hat) ** 2).sum() / ((log_y - log_y.mean()) ** 2).sum()
print(f"R2: {R2}")
```

```python
(
    hv.Scatter(
        (log_y_hat[: 2**16], log_y[: 2**16]),
        kdims="prediction",
        vdims="truth",
    )
    + hv.HexTiles(
        (log_y_hat, log_y),
        kdims=["prediction", "truth"],
    ).opts(logz=True)
) * hv.Slope(1, 0).opts(color="black", line_width=1)
```

## mlp

```python
@tf.function
def four_wavelengths(x, y):
    return (
        tf.experimental.numpy.log10(
            tf.concat(
                (tf.gather(x, blue, axis=1), tf.gather(x, [green], axis=1)),
                axis=1,
            )
        ),
        tf.experimental.numpy.log10(tf.reduce_sum(y, axis=1)),
    )


network = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, "relu"),
        tf.keras.layers.Dense(64, "relu"),
        tf.keras.layers.Dense(64, "relu"),
        tf.keras.layers.Dense(1),
    ]
)
network.compile(
    optimizer=tf.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.MeanSquaredError(),
)
```

```python
fit = network.fit(
    train.map(four_wavelengths).shuffle(batch_size * 4),
    epochs=10,
)
```

```python
log_x, log_y = test.map(four_wavelengths).rebatch(test_size).get_single_element()
log_y_hat = network(log_x)[:, 0].numpy()
log_y = log_y.numpy()
```

```python
R2 = 1 - ((log_y - log_y_hat) ** 2).sum() / ((log_y - log_y.mean()) ** 2).sum()
print(f"R2: {R2}")
```

```python
(
    hv.Scatter(
        (log_y_hat[: 2**16], log_y[: 2**16]),
        kdims="prediction",
        vdims="truth",
    )
    + hv.HexTiles(
        (log_y_hat, log_y),
        kdims=["prediction", "truth"],
    ).opts(logz=True)
) * hv.Slope(1, 0).opts(color="black", line_width=1)
```

## mlp, loc scale out

```python
network = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, "relu"),
        tf.keras.layers.Dense(64, "relu"),
        tf.keras.layers.Dense(64, "relu"),
        tf.keras.layers.Dense(2),
        tfp.layers.IndependentNormal(1),
    ]
)
network.compile(
    optimizer=tf.optimizers.Adam(learning_rate=3e-4),
    loss=lambda y, model: tf.reduce_sum(-model.log_prob(y)),
)
```

```python
fit = network.fit(
    train.map(four_wavelengths).shuffle(batch_size * 4),
    epochs=10,
)
```

```python
log_x, log_y = test.map(four_wavelengths).rebatch(test_size).get_single_element()
log_y_model = network(log_x)
```

```python
idx = log_y_model.stddev() / tf.abs(log_y_model.mean()) < 0.5
```

```python
log_y_hat = tf.boolean_mask(log_y_model.sample(), idx).numpy()
log_y = tf.boolean_mask(log_y, idx[:, 0]).numpy()
```

```python
R2 = 1 - ((log_y - log_y_hat) ** 2).sum() / ((log_y - log_y.mean()) ** 2).sum()
print(f"R2: {R2}")
```

```python
(
    hv.Scatter(
        (log_y_hat[: 2**16], log_y[: 2**16]),
        kdims="prediction",
        vdims="truth",
    )
    + hv.HexTiles(
        (log_y_hat, log_y),
        kdims=["prediction", "truth"],
    ).opts(logz=True)
) * hv.Slope(1, 0).opts(color="black", line_width=1)
```

```python
# HERE transform to standard normal and plot ecdf
```
