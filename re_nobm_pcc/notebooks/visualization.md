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

```python
import datetime as dt

from cartopy import crs
import holoviews as hv
import geoviews as gv
import numpy as np
import xarray as xr

from re_nobm_pcc import DATADIR
from re_nobm_pcc.core import svd, read_nobm

gv.extension("bokeh")
```

```python
fontscale = 3
height = 400
hv.opts.defaults(
    hv.opts.Curve(active_tools=[], fontscale=fontscale),
    hv.opts.Bars(active_tools=[], fontscale=fontscale),
    hv.opts.Image(active_tools=[], tools=["hover"], fontscale=fontscale),
    hv.opts.Scatter(active_tools=[], fontscale=fontscale, frame_height=height),
    hv.opts.HexTiles(active_tools=[], tools=["hover"]),
)
```

# Model & Evaluation Approach

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

```python
import tensorflow as tf
import tensorflow_probability as tfp
```

```python
rng = np.random.default_rng(12345)

n = 0x400
x_true = rng.beta(6, 3, n) * 20 - 10
p = 1 / (1 + np.exp(-1.5 * x_true))
y_true = 1.2 ** rng.normal(x_true) * (rng.random(n) < p).astype(int)

x_true = x_true[:, np.newaxis]
y_true = y_true[:, np.newaxis]

units = 3
n = 1
m = 2

def tfd(t):
    distribution = tfp.distributions.Gamma(
        concentration=tf.math.softplus(t[..., :n]),
        log_rate=t[..., n:m],
    )
    inflated = tfp.distributions.Inflated(
        distribution,
        inflated_loc_logits=t[..., m:],
    )
    return tfp.distributions.Independent(inflated, reinterpreted_batch_ndims=1)

input = tf.keras.Input((1,))
layer = tf.keras.layers.Dense(64, 'sigmoid')(input)
layer = tf.keras.layers.Dense(units)(layer)
output = tfp.layers.DistributionLambda(tfd)(layer)

model = tf.keras.Model([input], [output])
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=3e-4),
    loss=lambda y, tfd: -tfd.log_prob(y),
    # run_eagerly=True,
)

fit = model.fit(x=x_true, y=y_true, epochs=300, verbose=0)
y_dist = model(x_true)
y_pred = y_dist.sample()
```

```python
(
    hv.Scatter((x_true, y_true), kdims=["Input"], vdims=["Output"])
    .opts(aspect=1, alpha=0.2, size=15, color="black", fontsize={'xticks': 0, 'yticks': 0})
)
```

```python
(
    hv.Scatter((y_true, y_pred), kdims=["True"], vdims=["Predicted"])
    .opts(aspect=1, alpha=0.2, size=15, color="black", fontsize={'xticks': 0, 'yticks': 0})
    * hv.Slope(1, 0).opts(color="darkorange", line_dash="dashed", line_width=5)
)
```

# Labelled Data

```python
paths = (DATADIR / "oasim").glob("*.nc")
oasim = xr.open_mfdataset(paths)
```

```python
date = np.datetime64("2020-09-03")
example = [
    [(-32, 52, "a"), (-18, -3, "b")],
    [(-160, -5, "c"), (150, 14, "d")],
]
points = [gv.Points(i, kdims=["lon", "lat"], vdims=["label"]) for i in example]
labels = [gv.Labels(i, kdims=["lon", "lat"], vdims=["label"]) for i in example]
for item in labels:
    item.dataset.data["lat"] = item.dataset.data["lat"] - 5
    item.dataset.data["lon"] = item.dataset.data["lon"] - 5
```

```python
nobm = read_nobm(date.astype("datetime64[M]"))
```

```python
dataset = xr.merge(
    (oasim.sel({"date": date}), nobm.sel({"date": date})),
)
dataset = dataset.roll({"lon": dataset.sizes["lon"] // 2}, roll_coords=True)
lon = dataset["lon"]
dataset["lon"] = lon.where(lon < 180, lon-360)
```

```python
layout = []
colorbar = False
for item in zip([-30, 150], points, labels):
    layout.append(
        gv.Image(da).opts(
            cmap="viridis",
            colorbar=colorbar,
            projection=crs.Geostationary(item[0]),
            global_extent=True,
            aspect=1,
            frame_height=400,
            clabel="log [Chl-a] (mg / m^3)"
        )
        * item[1].opts(
            size=15,
            color="label",
            cmap="rainbow",
            line_color="black",
            show_legend=False,
        )
        * item[2].opts(text_color="white")
        * gv.feature.coastline
    )
    colorbar = True
hv.Layout(layout)
```

```python
coords = xr.concat(
    (i.data.to_xarray().drop_vars("index") for i in points),
    dim="index",
)
coords = coords.swap_dims({"index": "label"})
point_dataset = dataset.sel({"lon": coords["lon"], "lat": coords["lat"]}, method="nearest")
point_dataset["tot"].load()
```

```python
phy = point_dataset["phy"].drop_vars(["lon", "lat", "date"])
(
    phy.hvplot.bar(by="label", cmap="rainbow", aspect=3/2)
    .opts({"Bars": {"line_width": 2}})
    .opts(
        xlabel="Taxonomic Group",
        ylabel="[Chl-a] (mg/m^3)",
        multi_level=False,
        frame_height=400,
        show_legend=False,
    )
)
```

```python
np.round(phy, 2)
```

```python
cycle = hv.plotting.util.process_cmap("rainbow", 4)
#cycle.reverse()
```

```python
rrs = point_dataset["rrs"].drop_vars(["lon", "lat", "date"])
rrs = rrs.sel({"wavelength": slice(350, 700)})
(
    rrs.hvplot.line(by="label", aspect=3/2)
    .opts({"Curve": {"line_width": 4, "color": hv.Cycle(cycle)}})
    .opts(
        xlabel="Wavelength",
        ylabel=r"Rrs  (1 / sr)",
        legend_position="top_right",
        frame_height=400,
    )
)
```

```python
rrs.max("wavelength").load()
```

```python

```
