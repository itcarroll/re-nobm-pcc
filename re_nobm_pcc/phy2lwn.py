from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from multiprocessing import Pool
import subprocess
import os

import numpy as np
import xarray as xr

from . import DATADIR, EXTENSION


def main(argv=None):

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('year', help='"year" part of the data path')
    parser.add_argument('pool', help='size of multiprocessing pool', type=int)
    args = parser.parse_args(argv)

    # prepare filesystm
    data = DATADIR/'oasim'/args.year
    data.mkdir(parents=True, exist_ok=True)

    # run OASIM on the NOBM data in a tempdir and copy out results
    nobm = xr.open_mfdataset(str(DATADIR/'nobm'/args.year/f'*{EXTENSION}'))
    var = ['dia', 'chl', 'cya', 'coc', 'tot'] # FIXME missing 'dtc' 'pic' 'cdc' 's' 't'
    nobm = nobm[var]

    # run oasim on a single time step in parallel
    pool = Pool(args.pool)
    pool.starmap(
        oasim,
        product([data], nobm.groupby('time', squeeze=False)),
    )


def oasim(dst, nobm):
    # because multiprocessing ...
    _, nobm = nobm
    nan = 999.0e9
    # run in private directory
    with TemporaryDirectory() as path:
        path = Path(path)
        # make parameter files available to a subprocess at path
        for item in (DATADIR/'oasim_param').glob('*'): # FIXME remove fake 'dtc' 'pic' 'cdc' 's' 't' from DATA_DIR/oasim_param
            (path/item.name).symlink_to(item)
        # capture the roll index
        lonroll = int((nobm['lon'] == 0.0).argmax())
        # write each variable to a binary file readable by fortran
        for item in nobm:
            da = nobm[item]
            da = (
                da
                .where(da.notnull(), nan)
                .roll(lon=lonroll)
            )
            save_fortran(path/f'day{item}.dat', da)
        # run OASIM
        subprocess.run(['modlwn1nm'], cwd=path)
        subprocess.run(['rrs1nm'], cwd=path)
        # initialize dataset
        oasim = nobm[list(nobm.dims)]
        oasim.attrs = {}
        oasim['wavelength'] = (('wavelength',), np.arange(250, 776))
        shape = np.array(list(oasim.dims.values()))
        # read fortran output into dataset
        for item in ['lwn', 'rrs']:
            da = load_fortran(path/f'{item}.dat', oasim.dims, np.float32)
            da = da.roll(lon=-lonroll)
            oasim[item] = da
        # postprocess and write dataset
        oasim = oasim.where(oasim != nan)
        time = oasim["time"].dt.strftime("%Y%m%d").data[0]
        oasim.to_netcdf(dst/f'day{time}{EXTENSION}')


def save_fortran(path, array):
    buffer = array.compute().data.transpose().tobytes()
    offset = np.array(len(buffer), dtype='i4').tobytes()
    with path.open('wb') as stream:
        stream.write(offset)
        stream.write(buffer)
        stream.write(offset)


def load_fortran(path, dims, dtype):
    shape = np.array(list(dims.values()))
    return xr.DataArray(
        data=(
            np.fromfile(
                path,
                dtype=dtype,
                count=shape.prod(),
                offset=4,
            )
            .reshape(shape, order='F')
        ),
        dims=list(dims),
    )


if __name__ == '__main__':
    try:
        task = os.environ['SLURM_ARRAY_TASK_ID']
        cpus = os.environ['SLURM_CPUS_PER_TASK']
        main([str(1998 + int(task)), cpus])
    except KeyError:
        main()
