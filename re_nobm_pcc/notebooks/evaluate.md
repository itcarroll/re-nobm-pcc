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

## Imports

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

```python
import json
from importlib import reload

import holoviews as hv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import xarray as xr

from re_nobm_pcc import DATA_DIR, TAXA
from re_nobm_pcc import viz

hv.extension('bokeh')
```

## Model

```python
DATA_DIR = DATA_DIR/'../.dvc/tmp/exps/standalone/tmpffedw8na/data'
network = tf.keras.models.load_model(DATA_DIR/'network', compile=False)
network.summary()
```

## Loss by Epoch

```python
fit = xr.Dataset({
    k: ('epoch', v) for k, v in np.load(DATA_DIR / 'fit.npz').items()
})
offset = 1 - min(*tuple(v.item() for k, v in fit.min().items()))
viz.loss(fit, offset)
```

```python
# network.load_weights(DATA_DIR / 'fit' / 'epoch-90')
```

## Test: Metrics

```python
fix_index = pd.Series(['mean'] + [str(i+1) for i in range(5)], index=TAXA)
with (DATA_DIR/'metrics.json').open() as stream:
    metrics = json.load(stream)
```

```python
table = (
    pd.DataFrame.from_dict(
        {tuple(k.split('_'))[-2:]: [v] for k, v in metrics.items()},
        orient='columns',
    )
    .stack(level=0).droplevel(0)
)
columns = ['ME', 'MAE', 'RMSE', 'R2']
table = pd.concat((pd.DataFrame(columns=columns), table))
table = table.loc[fix_index]
table.index = fix_index.index
table[columns]
```

## Test: True vs. Predicted

```python
TAKE = 8

dataset = tfds.builder('rrs_day_tfds', data_dir=DATA_DIR)
test = dataset.as_dataset(split='split[9%:10%]', as_supervised=True)
test = test.batch(2 ** 12)
y_true = []
y_pred = []
for item in test.take(TAKE):
    y_true.append(item[1].numpy())
    y_pred.append(network(item[0]).numpy()) # TODO hopefully .mean()
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
ds = xr.Dataset({
    'y_true': (('pxl', 'phy'), y_true),
    'y_pred': (('pxl', 'phy'), y_pred),
    'phy': ('phy', list(TAXA)),
})
```

```python
viz.hexbin(np.log10(ds))
```

```python

```
