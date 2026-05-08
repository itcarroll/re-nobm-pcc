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

# Plots

```python
import json

from dvc.api import DVCFileSystem
import pandas as pd
import hvplot.xarray
import holoviews as hv
import xarray as xr

from re_nobm_pcc.kit import TAXA

hv.extension('bokeh', logo=False)
```

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

```python
experiments = [
    'exp-0d139', # cnn f/32/p > ? > ?, 7996 parameters, swish, weighted loss
    'exp-e7486', # mnn 32 > 64, 19236 parameters, swish, weighted loss
    'exp-d1c73', # linear regression, weighted loss
]

weights = xr.DataArray(
    data=[55.77425765991211, 49.406314849853516, 49.0872688293457, 10.0181884765625],
    coords=[('group', TAXA)],
)
weights_on = ['ME', 'MAE', 'RMSE']

metrics = {}
for item in experiments:
    fs = DVCFileSystem('.', rev=item)
    with fs.open('data/metrics.json') as stream:
        metrics[item] = json.load(stream)
metrics = (
    pd.DataFrame([
        pd.Series(v.values(), index=v, name=k) for k, v in metrics.items()
    ])
    .transpose()
    .drop('loss')
)
metrics.index = pd.MultiIndex.from_tuples(
    (
        tuple(i[1:]) for i in metrics.index.str.split('_')
    )
)
metrics.index = metrics.index.rename(('group', 'metric'))
metrics = metrics.to_xarray()
metrics.loc[{'metric': weights_on}] = metrics.loc[{'metric': weights_on}] * weights
metrics = metrics.loc[{'metric': weights_on + ['R2']}]
```

```python
(
    metrics
    .hvplot
    .bar('metric', groupby='group', xlabel='')
    .options('Bars', xrotation=45, frame_width=400, fontscale=1.2)
    .layout().cols(2)
    .options('NdLayout', shared_axes=False)
)
```

# Notes (New Orleans 2024)


The size of the labelled dataset is reduced (about 1% of available pixels), and it's now stored in a zarr.

- train,    4_194_304
- validate, 1_048_576
- test,       262_144


## leafy-tipi


Mainly a fix to ceric-guns that thresholds y so all have zero inflation (sets values less than 1e-10 to 0), but also a new way to get the tfd shape (should have no effect).

Network (changes from below)
- none


## ceric-guns


A first attempt with the independent inflated gamma.

Network
- hidden layers
  - dense, 64, sigmoid
- input, 381
- output, 6
- Independent Inflated(log_logits) Gamma(concentration=softmax, log_rate), 6 * 3
- adam, 3e-5
- negative log likelihood

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Notes (Liegè 2023)
<!-- #endregion -->

to process exp_id/jobid/tmpdir = (batch, arch, learning rate)
slurm 621639 is looking good, 621634-5 are duds, 621636 was good until nan (segfaulted?),

in-process
- tmpq_rtdiex/621909 = 64, [128] * 3, 3e-6
- tmpdip_1o4_ = 64, [64]  * 3, 3e-6 <- loss=-1e3
- tmpffedw8na = 64, [128] * 3, 3e-5 <- nan, looks blah
- tmpp6dcvgo5 = 64, [128] * 2, 3e-7 <- loss ballooned
- tmpu17bc_29 = 64, [64]  * 2, 3e-7 <- loss=-3e8

complete
- pseud-pail  = 64, [128]    , 3e-5
- nowed-suss  = 64, [128] * 4, 3e-5 <- why stop? cut out early-stopping since i have checkpoints
- local-duad  = 64, [128]    , 3e-6
- hardy-bids  = 64, [64]  * 4, 3e-5 <- good until nan
- third-prat  = 64, [64]  * 4, 3e-6 <- why stop?
- newsy-volt  = 64, [32]  * 4, 3e-6
- axile-flux  = 64, [128] * 4, 3e-5



## pseud-pail

<!-- #raw -->
Dimensions:   (epoch: 169)
Data variables:
    loss      (epoch) float64 2.287e+05 5.541e+06 ... -953.6 -1.142e+03
    val_loss  (epoch) float64 2.549e+04 3.975e+04 ... -1.541e+03 -1.55e+03
Data variables:
    loss      float64 -1.397e+03
    val_loss  float64 -1.568e+03
<!-- #endraw -->

## local-duad
spikes but recovers, with never great loss

<!-- #raw -->
Dimensions:   (epoch: 20)
Data variables:
    loss      (epoch) float64 4.168e+05 1.119e+05 ... 1.281e+05 1.142e+04
    val_loss  (epoch) float64 4.313e+04 1.091e+03 ... 4.045e+03 1.507e+04
Data variables:
    loss      float64 5.59e+03
    val_loss  float64 -223.3
<!-- #endraw -->

## nowed-suss
begins recovery from late spike, why stopped?

<!-- #raw -->
Dimensions:   (epoch: 12)
Data variables:
    loss      (epoch) float64 1.132e+03 -717.2 -84.36 ... 1.188e+06 4.796e+05
    val_loss  (epoch) float64 -679.4 -766.1 123.7 ... 1.643e+03 1.723e+04
Data variables:
    loss      float64 -717.2
    val_loss  float64 -766.1
<!-- #endraw -->

## axile-flux

<!-- #raw -->
Dimensions:   (epoch: 2)
Data variables:
    loss      (epoch) float64 -26.2 -184.2
    val_loss  (epoch) float64 -777.6 -921.7
<!-- #endraw -->

## third-prat

<!-- #raw -->
Dimensions:   (epoch: 13)
Data variables:
    loss      (epoch) float64 3.705e+03 2.32e+04 1.293e+03 ... 832.5 1.321e+03
    val_loss  (epoch) float64 -412.8 488.2 -575.1 ... 3.925e+05 2.647e+04
Data variables:
    loss      float64 -99.33
    val_loss  float64 -575.1
<!-- #endraw -->

## hardy-bids

<!-- #raw -->
Dimensions:   (epoch: 14)
Data variables:
    loss      (epoch) float64 1.528e+03 -472.9 -764.4 ... -587.4 3.475e+05 nan
    val_loss  (epoch) float64 -674.1 -576.8 -957.3 ... -589.0 2.332e+12 nan
Data variables:
    loss      float64 -1.058e+03
    val_loss  float64 -1.226e+03
<!-- #endraw -->

## newsy-volt

<!-- #raw -->
Dimensions:   (epoch: 12)
Data variables:
    loss      (epoch) float64 2.903e+03 4.463e+03 ... 2.359e+03 5.005e+03
    val_loss  (epoch) float64 161.0 -258.2 163.0 ... 1.254e+04 -113.5 5.363e+03
Data variables:
    loss      float64 979.7
    val_loss  float64 -258.2
<!-- #endraw -->

## 47d9c52f (merge dev into cnn)


### exp-19f79


- cnn with 3 layers (8, 16, 32 filters)
- swish
- 3,284 params

- cya stuck at zero

<!-- #raw -->
'Test loss: 2.906240701675415'
loss	AUC	ME	MAE	RMSE	R2
chl	0.012148	NaN	-0.004036	0.012148	0.042415	0.437520
coc	0.019029	NaN	-0.005862	0.019029	0.048158	0.101442
cya	0.020666	NaN	-0.020666	0.020666	0.036025	-0.490467
dia	0.027364	NaN	0.002353	0.027364	0.062000	0.814033
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-c9fa5
<!-- #endregion -->

- cnn with 3 layers (4, 8, 16 filters)
- 1,180 params
- swish

- chl and coc have noisy val_loss
- cya got stuck at zeros

<!-- #raw -->
'Test loss: 2.9124202728271484'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011597	NaN	-0.006417	0.011597	0.039975	0.500376
coc	0.019766	NaN	-0.003106	0.019766	0.048758	0.078908
cya	0.020665	NaN	-0.020663	0.020665	0.036022	-0.490211
dia	0.027414	NaN	-0.004076	0.027414	0.061976	0.814178
<!-- #endraw -->

## 60b305ad (loss by group)


### exp-09501


- mnn with 2 narrowing layers (56, 24)
- 30,980 params
- swish

looks nice

<!-- #raw -->
'Test loss: 2.213247299194336'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011035	NaN	-0.004069	0.011035	0.043677	0.403546
coc	0.017221	NaN	-0.003758	0.017221	0.045836	0.186014
cya	0.009940	NaN	-0.005198	0.009940	0.021895	0.449416
dia	0.025857	NaN	-0.002382	0.025857	0.060703	0.821731
<!-- #endraw -->

### exp-e7486


- mnn with 2 widening layers (32, 64)
- 19,236 params
- swish

<!-- #raw -->
'Test loss: 2.198610544204712'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010630	NaN	-0.004393	0.010630	0.037945	0.549840
coc	0.017528	NaN	-0.002499	0.017528	0.045617	0.193757
cya	0.009878	NaN	-0.004278	0.009878	0.021929	0.447715
dia	0.025439	NaN	-0.004534	0.025439	0.060447	0.823233
<!-- #endraw -->

### exp-e1687, exp-d1c73


- linear regression
- ran twice

<!-- #raw -->
'Test loss: 2.7233831882476807'
loss	AUC	ME	MAE	RMSE	R2
chl	0.014333	NaN	-0.004640	0.014333	0.052574	0.135829
coc	0.020903	NaN	-0.005055	0.020903	0.053047	-0.090282
cya	0.011180	NaN	-0.006010	0.011180	0.023711	0.354345
dia	0.034176	NaN	-0.003106	0.034176	0.072351	0.746753
<!-- #endraw -->

<!-- #raw -->
'Test loss: 2.642101287841797'
loss	AUC	ME	MAE	RMSE	R2
chl	0.013247	NaN	-0.008834	0.013247	0.045224	0.360559
coc	0.020142	NaN	-0.006644	0.020142	0.051897	-0.043503
cya	0.011472	NaN	-0.005772	0.011472	0.024190	0.327991
dia	0.034432	NaN	-0.006163	0.034432	0.072229	0.747604
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-6cc07
<!-- #endregion -->

- cnn with 3 layers (4, 16, 32 filters)
- 4,100 params
- swish

- cya stuck at zero

<!-- #raw -->
'Test loss: 3.0120980739593506'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011793	NaN	-0.006666	0.011793	0.041870	0.451877
coc	0.020625	NaN	-0.001592	0.020625	0.049482	0.051330
cya	0.020665	NaN	-0.020665	0.020665	0.036021	-0.490113
dia	0.032035	NaN	0.010783	0.032035	0.067541	0.779308
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-7f142
<!-- #endregion -->

- mnn with 2 narrowing layers (32, 16)
- 17,460 params

<!-- #raw -->
'Test loss: 2.2939674854278564'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010785	NaN	-0.005093	0.010785	0.037641	0.557021
coc	0.018536	NaN	-0.000353	0.018536	0.057101	-0.263286
cya	0.009981	NaN	-0.005283	0.009981	0.022170	0.435505
dia	0.028615	NaN	-0.006390	0.028615	0.067141	0.781912
<!-- #endraw -->

## 0a097ce8f (restore convolutions)


- weights MAE by mean (train) abundance


### exp-0d139


- cnn with 3 layers narrowing from 32 filter
- swish
- 7,996 parameters

<!-- #raw -->
'Test loss: 2.3489720821380615'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010822	NaN	-0.003861	0.010822	0.038912	0.526598
coc	0.019944	NaN	-0.000234	0.019944	0.048672	0.082169
cya	0.010213	NaN	-0.005858	0.010213	0.022291	0.429350
dia	0.025824	NaN	-0.001769	0.025824	0.074518	0.731353
<!-- #endraw -->

### exp-b0694


- cnn with 3 layers (8 outputs, window shrinking from 7)
- 5,364 trainable parameters
- swish activation
- coc looks better, but isn't by R2

<!-- #raw -->
'Test loss: 2.308502435684204'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010930	NaN	-0.004999	0.010930	0.036770	0.577287
coc	0.018837	NaN	-0.003468	0.018837	0.048391	0.092723
cya	0.010252	NaN	-0.005543	0.010252	0.022674	0.409574
dia	0.026448	NaN	-0.007138	0.026448	0.060378	0.823636
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-11c7f
<!-- #endregion -->

- cnn with 2 layers widenning from 8 filters
- swish
- ends with dense 64 layer
- 70,788 params

<!-- #raw -->
'Test loss: 2.409071207046509'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011292	NaN	-0.005758	0.011292	0.038708	0.531540
coc	0.019843	NaN	-0.003123	0.019843	0.163190	-9.318094
cya	0.010637	NaN	-0.006799	0.010637	0.024084	0.333817
dia	0.027631	NaN	-0.002807	0.027631	0.073848	0.736166
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-d1e2b
<!-- #endregion -->

- cnn with 1 layer, 8 filters
- then a dense 64 layer
- swish
- 89,444 params

<!-- #raw -->
'Test loss: 2.5086543560028076'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011539	NaN	-0.005268	0.011539	0.039064	0.522902
coc	0.020880	NaN	-0.003335	0.020880	0.157827	-8.651020
cya	0.010925	NaN	-0.007232	0.010925	0.024605	0.304697
dia	0.029665	NaN	0.002033	0.029665	0.064299	0.799986
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-ac07d
<!-- #endregion -->

- cnn with 1 conv layer having 8 outputs
- 1 dense layer with 32 nodes
- 44k trainable parameters
- terrible

<!-- #raw -->
'Test loss: 2.9868886470794678'
loss	AUC	ME	MAE	RMSE	R2
chl	0.012142	NaN	-0.001109	0.012142	0.041105	0.471733
coc	0.020217	NaN	-0.001239	0.020217	0.048685	0.081661
cya	0.020665	NaN	-0.020664	0.020665	0.036022	-0.490265
dia	0.029588	NaN	0.001345	0.029588	0.547218	-13.486889
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-33f53
<!-- #endregion -->

- cnn with 1 conv layer having 8 outputs
- 1 dense layer with 16 nodes
- 22k trainable parameters

<!-- #raw -->
'Test loss: 2.5226738452911377'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011403	NaN	-0.004773	0.011403	0.038800	0.529320
coc	0.021783	NaN	0.001537	0.021783	0.051808	-0.039930
cya	0.010736	NaN	-0.006954	0.010736	0.023949	0.341291
dia	0.028295	NaN	-0.008302	0.028295	0.063815	0.802986
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-b3f84
<!-- #endregion -->

- cnn with 3 layers decreasing in bands (from 16)
- 1 dense layer with 32 nodes
- 3,344 trainable parameters
- val_loss went bad, overfitting, but seems odd with this few parameters

<!-- #raw -->
'Test loss: 2.971609592437744'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010975	NaN	-0.004861	0.010975	0.038292	0.541571
coc	0.021676	NaN	0.002031	0.021676	0.050321	0.018906
cya	0.020666	NaN	-0.020666	0.020666	0.036025	-0.490510
dia	0.027362	NaN	-0.002998	0.027362	0.062903	0.808578
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-72218
<!-- #endregion -->

- widening cnn (only 9k params) with swish activations
- poor performance except for diatoms

<!-- #raw -->
'Test loss: 2.3822529315948486'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011127	NaN	-0.004222	0.011127	0.047965	0.280689
coc	0.019963	NaN	-0.001227	0.019963	0.100067	-2.879654
cya	0.010458	NaN	-0.004341	0.010458	0.049245	-1.785061
dia	0.026153	NaN	-0.006758	0.026153	0.060267	0.824282
<!-- #endraw -->

## 2dc27b89 (restore weights)


- weights on MAE


### exp-eab57


- no layers, just regression

<!-- #raw -->
'Test loss: 2.672224760055542'
loss	AUC	ME	MAE	RMSE	R2
chl	0.013627	NaN	-0.006233	0.013627	0.048942	0.251109
coc	0.020561	NaN	-0.005626	0.020561	0.052590	-0.071545
cya	0.011256	NaN	-0.004650	0.011256	0.023762	0.351516
dia	0.034319	NaN	-0.002828	0.034319	0.071172	0.754941
<!-- #endraw -->

```python
import numpy as np

# weights (inverse mean of training data)
weights = np.array(
    [55.77426528930664, 49.406314849853516, 49.087276458740234, 10.018187522888184]
)
print(1/weights)
weights.dot([0.013627, 0.020561, 0.011256, 0.034319])
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-c0ff8
<!-- #endregion -->

- mnn with 3 narrowing (from 128) layers (77,924)
- swish activation
- loss noisy at 200 epochs

<!-- #raw -->
'Test loss: 2.3015084266662598'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011108	NaN	-0.003682	0.011108	0.057716	-0.041500
coc	0.018151	NaN	-0.003074	0.018151	0.047728	0.117401
cya	0.010654	NaN	-0.001883	0.010654	0.097159	-9.841341
dia	0.026170	NaN	-0.001227	0.026170	0.072183	0.747928
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-b09b1
<!-- #endregion -->

- mnn with 3 widening (from 32) layers ()
- swish activation

<!-- #raw -->
'Test loss: 2.221733331680298'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011126	NaN	-0.002731	0.011126	0.041488	0.461840
coc	0.017779	NaN	-0.000806	0.017779	0.044947	0.217273
cya	0.009557	NaN	-0.002770	0.009557	0.031573	-0.144867
dia	0.025325	NaN	0.000862	0.025325	0.061010	0.819921
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-b6cb1
<!-- #endregion -->

- mnn with 3 wide (64 nodes) layers (42,308 trainable parameters)
- swish activate
- very early termination (~80 epochs)

<!-- #raw -->
'Test loss: 2.255746603012085'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010688	NaN	-0.004342	0.010688	0.039924	0.501658
coc	0.018306	NaN	-0.001531	0.018306	0.046519	0.161557
cya	0.010328	NaN	-0.003140	0.010328	0.022661	0.410216
dia	0.024776	NaN	-0.001365	0.024776	0.058526	0.834291
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-b9eff
<!-- #endregion -->

- mnn with 3 narrowing (from 32 nodes) layers (17,564 trainable parameters)
- swish activation

<!-- #raw -->
'Test loss: 2.241039752960205'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010943	NaN	-0.002994	0.010943	0.044897	0.369782
coc	0.017548	NaN	-0.002584	0.017548	0.046169	0.174145
cya	0.009973	NaN	-0.005313	0.009973	0.021891	0.449652
dia	0.027367	NaN	-0.000264	0.027367	0.063320	0.806029
<!-- #endraw -->

### exp-9f6bf


- mnn with 3 widening (from 8 nodes) layers (5,036 trainable parameters)
- swish activation

<!-- #raw -->
'Test loss: 2.1852617263793945'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010867	NaN	-0.004432	0.010867	0.037672	0.556296
coc	0.017330	NaN	-0.003577	0.017330	0.046128	0.175609
cya	0.009532	NaN	-0.003976	0.009532	0.021168	0.485385
dia	0.025457	NaN	-0.003357	0.025457	0.059085	0.831107
<!-- #endraw -->

### exp-60b2d


- mnn with 3, 32 node layers, relu activation

<!-- #raw -->
'Test loss: 2.1982078552246094'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010762	NaN	-0.004088	0.010762	0.040754	0.480730
coc	0.017248	NaN	-0.001879	0.017248	0.044910	0.218547
cya	0.009920	NaN	-0.003386	0.009920	0.021730	0.457688
dia	0.025840	NaN	-0.000791	0.025840	0.061541	0.816777
<!-- #endraw -->

## 594b900 (simple model with only abundance)


### exp-4781f

change from exp-c7418:
- it weights the MAE loss by inverse of group means
- a marginal improvement in R2

<!-- #raw -->
'Test loss: 2.260983467102051'
loss	AUC	ME	MAE	RMSE	R2
chl	0.010650	NaN	-0.003946	0.010650	0.037557	0.558996
coc	0.018073	NaN	-0.002085	0.018073	0.046591	0.158964
cya	0.010249	NaN	-0.005946	0.010249	0.022945	0.395388
dia	0.027045	NaN	-0.006145	0.027045	0.061881	0.814745
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-c7418
<!-- #endregion -->

change from exp-b7004:
- 'relu' activation
- minimal impact

<!-- #raw -->
'Test loss: 0.06551116704940796'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011420	NaN	-0.003401	0.011420	0.040104	0.497151
coc	0.017706	NaN	-0.004282	0.017706	0.047439	0.128076
cya	0.010313	NaN	-0.005339	0.010313	0.022703	0.408040
dia	0.026072	NaN	-0.003928	0.026072	0.060662	0.821974
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### exp-b7004
<!-- #endregion -->

this looks like an okay "naive" case:
- no weight
- has a single layer with 32 nodes and 'swish' activation
- performs fine on data set to 0.0 at or below 10e-something

<!-- #raw -->
'Test loss: 0.0653865784406662'
loss	AUC	ME	MAE	RMSE	R2
chl	0.011078	NaN	-0.004085	0.011078	0.040480	0.487676
coc	0.017908	NaN	-0.003272	0.017908	0.046911	0.147362
cya	0.010439	NaN	-0.006818	0.010439	0.023156	0.384194
dia	0.025962	NaN	-0.005958	0.025962	0.059386	0.829383
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 2e21a66 (loss weighted by abundance prob)
<!-- #endregion -->

### exp-9f291

I don't trust the AUC values shown, as the ROC curves were essentially diagonal. Everything else looked bad to worse.

```python
metrics = {
  "loss": 1.2280960083007812,
  "product_chl_ME": -0.004955189768224955,
  "product_chl_MAE": 0.022813349962234497,
  "product_chl_RMSE": 0.05957883968949318,
  "product_chl_R2": -0.10980522632598877,
  "product_coc_ME": -0.015996867790818214,
  "product_coc_MAE": 0.01973014324903488,
  "product_coc_RMSE": 0.06181754171848297,
  "product_coc_R2": -0.4805917739868164,
  "product_cya_ME": 0.0014727258821949363,
  "product_cya_MAE": 0.016090724617242813,
  "product_cya_RMSE": 0.02713635377585888,
  "product_cya_R2": 0.15429013967514038,
  "product_dia_ME": -0.02798730880022049,
  "product_dia_MAE": 0.06729260087013245,
  "product_dia_RMSE": 0.12573890388011932,
  "product_dia_R2": 0.23511826992034912,
  "presence_chl_loss": 0.3108878433704376,
  "abundance_chl_loss": 0.024213816970586777,
  "presence_coc_loss": 0.3642471730709076,
  "abundance_coc_loss": 0.01825951784849167,
  "presence_cya_loss": 0.38027915358543396,
  "abundance_cya_loss": 0.015522638335824013,
  "presence_dia_loss": 0.04745016619563103,
  "abundance_dia_loss": 0.06723576039075851,
  "presence_chl_AUC": 0.8756850957870483,
  "presence_coc_AUC": 0.8391563892364502,
  "presence_cya_AUC": 0.9311991333961487,
  "presence_dia_AUC": 0.4999741017818451
}
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## c106b09
<!-- #endregion -->

### exp-6852d


### exp-73a57


### exp-681ae

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 55a25bd
<!-- #endregion -->

### exp-fab79


### exp-4cddf


### exp-19308


### exp-302e1


### exp-cd4ba

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 52b6478
<!-- #endregion -->

### exp-52daa


### exp-96767

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 9960490
<!-- #endregion -->

### exp-38b0c
