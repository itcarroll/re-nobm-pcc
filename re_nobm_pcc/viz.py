import holoviews as hv
import hvplot.xarray
import numpy as np

from . import TAXA

PGIMENTS = {
    'chl': 'Greens',
    'cya': 'RdPu',
    'coc': 'Wistia',
    'dia': 'Reds',
}

def loss(ds):
    plt = (
        ds.hvplot.line(x='epoch', y=['loss', 'val_loss']).options('Curve', color='black')
        # + hv.Overlay(tuple(
        #     fit.hvplot.line(x='epoch', y=[f'abundance_{i}_loss', f'val_abundance_{i}_loss'], logy=True)
        #     for i in TAXA
        # )).options('Curve', color=hv.Cycle(
        #     ['blue', 'blue', 'orange', 'orange', 'red', 'red', 'green', 'green']
        # ))
    )
    return (
        plt
        .options(shared_axes=False)
        .options('Curve', line_dash=hv.Cycle(['solid', 'dashed']))
        # .cols(1)
    )

def hexbin(ds):
    hv.output(size=120)
    plots = {}
    for item in TAXA:
        plots[item] = (
            hv.HexTiles(
                data=(ds[item], ds[f'{item}_hat']),
                kdims=['y', 'y_hat'],
            ).options(
                logz=True,
                cmap=PGIMENTS[item],
                bgcolor='lightskyblue',
                tools=['hover'],
                padding=0.001,
                aspect='square',
                fontscale=1.4,
                colorbar=True,
            )
            *hv.Slope(1, 0).options(
                color='black',
                line_width=1.5,
            )
        )
    return (
        hv.HoloMap(plots, kdims='group').layout().cols(2).options(shared_axes=False)
    )


def roc(ds):
    n = ds.sizes['pxl']
    plots = {}
    for item in TAXA:
        order = ds[f'{item}_hat'].argsort()
        false_neg = np.insert(ds[item].cumsum(), 0, 0)
        pos = false_neg[-1]
        neg = n - pos
        true_neg = np.arange(n + 1) - false_neg
        true_pos_rate = 1 - false_neg / pos
        false_pos_rate = 1 - true_neg / neg
        plots[item] = hv.Curve(
            (false_pos_rate, true_pos_rate), 'False Positive Rate', 'True Positive Rate'
        )
    return hv.HoloMap(plots, kdims='group').overlay().opts(legend_position='bottom_right')
