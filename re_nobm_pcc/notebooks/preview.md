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

# Preview


## Issues


ideas
- oasim to zarr
- mixture-density networks
- dashboard with phy shifted to one species, OC vars otherwise the same, with tap to compare spectra
- transform of outputs
- pca outputs, to reduce dimensionality as needed
- pca inputs, to reduce complexity
- test for signal
  - 1 vs 2 nearest neighbor outputs
- dealing with unbalanced data (are they unbalanced?)
- try classification only
- relative abundances


## Imports

```python
import warnings
import datetime as dt

# from IPython.display import Markdown
# from scipy.stats import zscore
import holoviews as hv
import hvplot.xarray
import numpy as np
import pandas as pd
import panel as pn
import param as p
import xarray as xr

from re_nobm_pcc import DATADIR, TAXA, WAVELENGTH
#from re_nobm_pcc import kit, preprocess
#from re_nobm_pcc.simulate import NUMNAN, OC
#from oasim_rrs import modlwn1nm, rrs1nm

hv.opts.defaults(
    hv.opts.Curve(active_tools=[]),
    hv.opts.Image(active_tools=[]),
    hv.opts.Scatter(active_tools=[]),
    hv.opts.HexTiles(active_tools=[], tools=["hover"]),
)
pn.extension()
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

## Raw Data


### Phytoplankton Absorption and Scattering


The OASIM model requires absorption and backscattering for each phytoplankton group.

```python
ds = []
for item in TAXA:
    path = DATADIR / f"oasim_param/{item}1.txt"
    df = pd.read_table(path, sep="\t", dtype={0: int})
    df.columns = ("wavelength", "absorption", "scattering")
    da = df.set_index("wavelength").to_xarray().expand_dims("component")
    da["component"] = [item]
    ds.append(da)
ds = xr.concat(ds, "component")
(
    ds.hvplot.line(x="wavelength", y="absorption", by="component")
    + ds.hvplot.line(x="wavelength", y="scattering", by="component")
).cols(1)
```

The NOBM data provided by Cecile contains the ocean constituents that are sufficient inputs for the OASIM Fortran library to calculte Rrs.

Below 350nm however, there is no phytoplankton absorption data so those Rrs values should be ignored.

```python
paths = (DATADIR / "oasim").glob("*.nc")
ds = xr.open_mfdataset(sorted(paths), concat_dim="example", combine="nested")
ds = ds.set_index({"example": ["date", "lon", "lat"]})
```

```python
class BGCDashboard(p.peterized):
    # part of the GUI
    date = p.Date(dt.date(1998, 1, 1))
    bgc = p.Selector(
        list(set(long_name) - set(TAXA)),
        default="tpp",
        label="Ocean Property Variable",
    )
    phy = p.Selector(ds["component"].values.tolist(), label="Phytoplankton Group")
    # needed as dependencies, not part of the GUI
    data = p.ClassSelector(xr.Dataset)

    @p.depends("date", watch=True, on_init=True)
    def _load_date(self):
        self.data = (
            ds.sel({"date": np.datetime64(self.date, "ns")})
            .drop_vars(['phy', 'rrs'])
            .load()
            .unstack()
        )

    @p.depends("data", "bgc")
    def plt_bgc(self):
        da = self.data[self.bgc]
        return da.hvplot.image(x="lon", y="lat", clabel=self.bgc, title="Ocean Property")

    @p.depends("data", "phy")
    def plt_phy(self):
        da = self.data["pp"].sel({"component": self.phy})
        plt = da.hvplot.image(x="lon", y="lat", clabel=self.phy, title="Phytoplankton Primary Productivity")
        return plt

dash = BGCDashboard(name="NOBM Variables (without Phytoplankton Chl)")
pn.Row(
    pn.Column(
        dash.plt_bgc,
        dash.plt_phy,
    ),
    pn.panel(
        dash.param,
        parameters=["date", "bgc", "phy"],
        widgets={"date": pn.widgets.DatePicker},
    )
)
```

```python
class RRSDashboard(p.peterized):
    # part of the GUI
    date = p.Date(dt.date(1998, 1, 1))
    phy = p.Selector(ds["component"].values.tolist(), label="Phytoplankton Group")
    # needed as dependencies, not part of the GUI
    data = p.ClassSelector(xr.Dataset)
    stream = hv.streams.Tap(x=0, y=0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tap = xr.DataArray(
            np.empty((ds.sizes["wavelength"], 0), dtype=ds["rrs"].dtype),
            dims=("wavelength", "tap"),
            name="rrs",
        )

    @p.depends("date", watch=True, on_init=True)
    def _load_date(self):
        self.data = (
            ds.sel({"date": np.datetime64(self.date, "ns")})
            .load()
            .unstack()
        )

    @p.depends("data", "phy")
    def plt_phy(self):
        da = self.data["phy"].sel({"component": self.phy})
        plt = da.hvplot.image(x="lon", y="lat", clabel=self.phy, title="Phytoplankton specific Chl-a")
        self.stream.source = plt
        return plt

    @p.depends("stream.x", "stream.y")
    def plt_rrs(self):
        da = self.data["rrs"].sel(
            {"lon": self.stream.x, "lat": self.stream.y},
            method="nearest",
        )
        da = da.expand_dims("tap")
        self.tap = xr.concat((self.tap, da), dim="tap")
        return self.tap.hvplot(x="wavelength", by="tap", title="R_rs")


dash = RRSDashboard(name="NOBM Variables and Computed Rrs")
pn.Row(
    pn.Column(
        dash.plt_phy,
        dash.plt_rrs,
    ),
    pn.panel(
        dash.param,
        parameters=["date", "phy"],
        widgets={"date": pn.widgets.DatePicker},
    )
)
```

## Sample of Preprocessed Data

```python
rng = np.random.default_rng(1234)
sample, *_ = preprocess.open_dataset((1,), rng, 2 ** 14)
sample = sample.as_numpy_iterator()
```

```python
sample = next(sample)
```

```python
x = xr.DataArray(sample[0], coords={"wavelength": np.array(WAVELENGTH)}, dims=["example", "wavelength"], name="rrs")
y = xr.DataArray(np.sqrt(sample[1]), coords={"components": np.array(TAXA)}, dims=["example", "components"], name="phy")
```

### Inputs

```python
x.hvplot.line(groupby="example", ylim=(0, 0.02))
```

```python
svd = kit.svd(x - x.mean("example"), "wavelength", 8)
```

```python
svd["vectors"].hvplot.line(groupby="percentage", frame_width=400).layout().cols(2)
```

```python
os.chdir(DATADIR)
```

```python
ds_at_date = (
    ds.sel({
        "date": np.datetime64(dash.date, 'ns'),
    })
    .unstack()
    .reindex({"lat": np.linspace(-84, 71.4, 234)})
    .transpose("lon", "lat", ...)
    [["phy",] + list(OC)]
    .load()
)
tot_at_date = ds_at_date["phy"].sum("component")
```

```python
phy = []
for i in range(ds_at_date.sizes["component"]):
    phy_at_date = ds_at_date["phy"] * 0
    phy_at_date[:, :, i] = tot_at_date
    rrs = rrs1nm(modlwn1nm(phy_at_date, *[ds_at_date[i].data for i in OC]))
    rrs = xr.DataArray(
        rrs,
        coords={
            "lon": ds_at_date["lon"],
            "lat": ds_at_date["lat"],
            "wavelength": ds["wavelength"],
        },
        name="Rrs",
    )
    rrs = rrs.where(rrs != NUMNAN)
    phy.append(rrs.sel({"lon": dash.stream.x, "lat": dash.stream.y}, method="nearest"))
phy = xr.concat(phy, dim="component")
phy["component"] = ds["component"]
```

```python
os.chdir(PWD)
```

```python
phy.hvplot.line(x="wavelength", by="component")
```

### Outputs

```python
scores = xr.merge((svd, y)).drop_dims("wavelength")
```

```python
(
    scores
    # .isel({"example": slice(None, 1024)})
    .hvplot.hexbin(
        x="weights",
        y="phy",
        # groupby=["percentage", "components"],
        aspect=1,
        widgets={"percentage": pn.widgets.Select, "components": pn.widgets.Select},
    )
)
```

## Spectum with Taxa

```python
x, y = next(train.shuffle(32).as_numpy_iterator())
(hv.Curve(x) + hv.Bars(y)).opts(shared_axes=False)
```

---

## Outdated Below

<!-- #region id="OUK41jPxQHeT" -->
## Preprocessed Data
<!-- #endregion -->

<!-- #region id="uZUWOMYQgYBK" -->
The features and labels are both model output from NASA GMAO using the [NOBM and OASIM](https://gmao.gsfc.nasa.gov/gmaoftp/NOBM) models. The labels are four phytoplankton chlorophyll densities output by NOBM. The features are normalized water leaving radiances output by OASIM, using the NOBM model as input.
<!-- #endregion -->

<!-- #raw -->
preprocess = importlib.reload(preprocess)
kit = importlib.reload(kit)
HyperLwn = preprocess.HyperLwn
PhytoChl = preprocess.PhytoChl

sample = xr.open_dataset(kit.DATA_DIR/'sample.nc')
sample['pxl'] = range(sample.sizes['pxl'])
sample['labels'] = (
    sample[kit.TAXA]
    .to_array(dim='component')
    .transpose('pxl', 'component', ...)
)
sample_n = (sample - sample.mean('pxl')) #/sample.std('pxl')
<!-- #endraw -->

<!-- #region id="Y0gHiexEgq3T" -->
### Features
<!-- #endregion -->

<!-- #region id="1CNGKtURguXF" -->
One NetCDF file contains all the predictor data. Note that the `FillValue` attribute is not set to `9.99e11` in the netCDF file (Cecile will fix in next version). There are no explicit coordinates given; they are documented as attributes.
<!-- #endregion -->

<!-- #raw -->
!ncdump -h {os.environ['PWD']}/data/nobm/HyperLwn.R2014.nc4
<!-- #endraw -->

<!-- #raw -->
nonnull_grid = int((~HyperLwn.isel(wavelength=0, month=0).isnull()).sum())
Markdown(f"""
Variable `HyperLwn` has non-null values at {nonnull_grid:,} pixels for each month
and wavelength.

In total, that gives {nonnull_grid * HyperLwn.sizes['month']:,} samples (that are highly non-independent!).
""")
<!-- #endraw -->

<!-- #raw -->
nonnull = int(HyperLwn.size - HyperLwn.isnull().sum())
Markdown(f"""
Augmented with coordinates, variable `HyperLwn` is a xarray.DataArray with {nonnull:,} values.
""")
<!-- #endraw -->

<!-- #raw -->
HyperLwn
<!-- #endraw -->

<!-- #region id="EpkjtJG-iRTW" -->
## Labels
<!-- #endregion -->

<!-- #region id="TFz1ooc6iaCY" -->
Each of twelve NetCDF files contain a month of NOBM model output. The first is representative. Unlike the HyperLwn file, this one contains coordinates.
<!-- #endregion -->

<!-- #raw -->
!ncdump -h {os.environ['PWD']}/data/nobm/monthly/mon200701.R2014.nc4
<!-- #endraw -->

The `PhytoChl` xarray.Dataset includes the different phytoplankton groups as variables.

<!-- #raw -->
PhytoChl
<!-- #endraw -->

<!-- #region id="FLvJkZOpjPzn" -->
## Plot your Data
<!-- #endregion -->

<!-- #region id="UaHaKBVujTGO" -->
## Features
<!-- #endregion -->

The radiances currently make a nice map, but the data should be more sparsely sampled.

<!-- #raw -->
dmap = (
    HyperLwn
    .sel(month=[2, 6, 10], wavelength=[465, 665], method='nearest')
    .hvplot.image(
        groupby=['month', 'wavelength'],
        subplots=True,
        clabel='Lwn (mW cm-2 microm-1 sr-1)',
        rasterize=True,
    )
    .opts(shared_axes=False)
)
dmap
<!-- #endraw -->

A few "typical" hyperspectral radiances.

<!-- #raw -->
dmap = (
    HyperLwn
    .sel({'lon': -120, 'lat': -15, 'month': [2, 6, 10]}, method='nearest')
    .hvplot
    .line(by='month', ylabel='Lwn')
    # * hv.Slope(0, -0.2).options(color=hv.dim('wavelength'))
)
dmap
<!-- #endraw -->

Mean centered radiances and corresponding phytoplankton abundances.

<!-- #raw -->
pxl = [4, 34, 53, 283]
grays = ['#000000', '#444444', '#777777', '#aaaaaa']
pigments = ['#47AC5F', '#FBEC2C', '#F884AB', '#E93429']
line = (
    sample['features'].sel(pxl=pxl)
    .hvplot
    .line(x='wavelength', by='pxl', ylabel='Lwn', legend=True)
    .options('Curve', fontscale=1.4, color=hv.Cycle(grays))
    .options('NdOverlay', legend_position='top_right')
)
(
    line
    + (
        sample['labels']
        .reset_coords(drop=True)
        .isel(pxl=pxl)
        .hvplot.bar(by='component')
        .options('Bars', fontscale=1.4, color=hv.Cycle(pigments))
    )
).cols(1)
<!-- #endraw -->

SVD to reduce the wavelength dimension to `k` vectors accounting for the most variation in the features. The singular values are:

<!-- #raw -->
k = 5
scores, s, vectors = kit.svd(sample_n['features'], dim='wavelength', k=k)
list(s.round(6))
<!-- #endraw -->

The corresponding vectors:

<!-- #raw -->
vectors.hvplot.line(x='wavelength', by='pc')
<!-- #endraw -->

A matrix of univariate (diagonal) and bivariate (off-diagonal) histograms of the `scores`, or coefficients generating each wavelength by linear combination of the `vectors` above.

<!-- #raw -->
(
    hvplot.scatter_matrix(
        scores.to_dataset(dim='pc').to_dataframe(),
        chart='hexbin',
        gridsize=16,
    )
    .opts(hv.opts.HexTiles(cmap='Viridis', tools=['hover']))
)
<!-- #endraw -->

<!-- #region id="uEXjgfVPjspm" -->
## Labels
<!-- #endregion -->

A map of the phytoplankton labels in `PhytoChl` at one month.

<!-- #raw -->
(
    PhytoChl
    .sel(month=[2, 5, 8, 11])
    .hvplot.image(
        z=kit.TAXA,
        groupby=['month'],
        subplots=True,
        clabel='chl-a',
        rasterize=True,
    )
)
<!-- #endraw -->

The distribution of the four phytoplankton groups.

<!-- #raw -->
sample['labels_p'] = (sample['labels'].dims, kit.ecdf(sample['labels']))
<!-- #endraw -->

<!-- #raw -->
(
    sample[['labels', 'labels_p']]
    .drop_vars('pxl')
    .hvplot
#    .line(x='labels', y='labels_p', by='component')
#    .opts(hv.opts.Curve(interpolation='steps-pre'))
    .scatter(x='labels', y='labels_p', by='component', xlabel='chl-a', ylabel='probability')
    .opts(title='ECDF of phytoplankton by component')
)
<!-- #endraw -->

<!-- #raw -->
scores, s, vectors = kit.svd(sample_n['labels'], dim='component')
s
<!-- #endraw -->

<!-- #raw -->
np.cov(scores, rowvar=False).round(8)
<!-- #endraw -->

<!-- #raw -->
labels = xr.Dataset({
    'scores': scores,
    'scores_p': (scores.dims, kit.ecdf(scores)),
})
(
    labels[['scores', 'scores_p']]
    .hvplot
#    .line(x='labels', y='labels_p', by='component')
#    .opts(hv.opts.Curve(interpolation='steps-pre'))
    .scatter(x='scores', y='scores_p', by='pc', xlabel='score', ylabel='probability')
    .opts(title='ECDF of phytoplankton PCA by component')
)
<!-- #endraw -->

<!-- #raw -->
(
    hvplot.scatter_matrix(
        scores.to_dataset(dim='pc').to_dataframe(),
        chart='hexbin',
        gridsize=16,
    )
    .opts(hv.opts.HexTiles(cmap='Viridis', tools=['hover']))
)
<!-- #endraw -->
