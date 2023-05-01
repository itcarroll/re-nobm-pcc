# RE-NOBM-PCC

The project aims to Reverse Emulate results from the NASA Ocean Biogeochemical
Model on Phytoplankton Community Composition.
Greg and Rousseaux ([2017][1]) simulated PACE-like scenes of the NOBM product
using the Ocean Atmosphere Spectral Irradiance Model (OASIM).
One could imagine emulating OASIM with machine learning (although OASIM is
pretty light-weight).
Instead, we aim to emulate the NOBM variables as a proof-of-concept for
learning about PCC from PACE.

## Getting Started (WIP)

merge these into older instructions ...

1. `conda env create --file=cuda.yaml`
1. `conda activate cuda`
1. `source .env`
1. `mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice`
1. `cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/`
1. `poetry install`

Unlike the conda "base" environment, the "cuda" environment is a virtual environment
that poetry will use directly (no nested virtual environment). You only need poetry
to install/update dependencies and the current project, so `poetry run ...` or
`poetry shell` are not needed.

older instructions ...

1. Set up a Python virtual environment using [Poetry][2].
    1. If not available, you [could][3] install Poetry with
    `pipx install poetry` or, on macOS, with `brew install poetry`.
    1. Within the project root (the location of `poetry.lock`), execute
    `poetry install`.
    1. Activate the environment with
    `source $(poetry env info -p)/bin/activate`.
1. Pull project data from a public Google Drive using [DVC][4]. For good
performance without additional customization, contact @itcarroll with your
Google Account email to request permission to use an existing
[Google Cloud Project][5]. The rather expansive permissions you must grant
during the subsequent step are presently unavoidable (see
[iterative/dvc#447][6]).
    1. Within the activated python environment, execute `dvc pull` and follow
    the instructions.
1. Place associated Google Colab [notebooks][7] in Google Drive under
`PCC-ML/colab-nobm`. Generally, a notebook needs to connect to a local kernel to
run. Configure (one time only) and launch a Jupyter server to enable.
    1. `jupyter serverextension enable --py jupyter_http_over_ws`
    1. `jupyter lab --NotebookApp.allow_origin='https://colab.research.google.com'`

## Reproducing (WIP)


## Acknowledgements

- NASA ROSES Grant 80NSSC21K0431

[1]: https://doi.org/10.3389/fmars.2017.00060
[2]: https://python-poetry.org/
[3]: https://python-poetry.org/docs/#installing-with-pipx
[4]: https://dvc.org
[5]: https://dvc.org/doc/user-guide/setup-google-drive-remote#using-a-custom-google-cloud-project-recommended
[6]: https://github.com/iterative/dvc/issues/4477
[7]: https://drive.google.com/drive/folders/1fE1Ck_XPoHQ2OVBSGRfEGlY72kSSklkE
