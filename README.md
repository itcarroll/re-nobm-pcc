# RE-NOBM-PCC

The project aims to Reverse Emulate results from the NASA Ocean Biogeochemical Model on Phytoplankton Community Composition.
Greg and Rousseaux ([2017](https://doi.org/10.3389/fmars.2017.00060)) simulated PACE-like scenes of the NOBM product using the Ocean Atmosphere Spectral Irradiance Model (OASIM).
One could imagine emulating OASIM with machine learning (although OASIM is pretty light-weight).
Instead, we aim to emulate the NOBM variables as a proof-of-concept for learning about PCC from PACE.

## Getting Started

1. Set up a Python virtual environment using [Poetry](https://python-poetry.org/).
    1. If not available, you [could](https://python-poetry.org/docs/#installing-with-pipx) install Poetry with `pipx install poetry`.
    1. Within the project root (the location of `poetry.lock`), execute `poetry install`.
    1. Activate the environment with `source $(poetry env info -p)/bin/activate`.
1. Pull project data from a public Google Drive using [DVC](https://dvc.org).
   For good performance without additional customization, contact @itcarroll with your Google Account email to request permission to use an existing [Google Cloud Project](https://dvc.org/doc/user-guide/setup-google-drive-remote#using-a-custom-google-cloud-project-recommended)
   The rather expansive permissions you must grant during the subsequent step are, at present, unavoidable.
  1. Within the activated python environment, execute `dvc pull` and follow the instructions.

## Acknowledgements

- NASA ROSES Grant 80NSSC21K0431
