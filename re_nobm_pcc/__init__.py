from pathlib import Path

DATA_DIR = (Path(__file__).parents[1]/'data').absolute()
del Path

EXTENSION = '.R2017.nc4'