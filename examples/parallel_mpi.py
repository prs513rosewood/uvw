import sys
import numpy as np

from mpi4py import MPI

from uvw.parallel import PRectilinearGrid
from uvw import DataArray


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if comm.Get_size() != 4:
  if rank == 0:
    print('Please execute with 4 MPI tasks', file=sys.stderr)
  sys.exit(1)

N = 20

# Domain bounds
bounds = [
    {'x': (-2, 0), 'y': (-2, 0)},
    {'x': (-2, 0), 'y': (0,  2)},
    {'x': (0,  2), 'y': (-2, 2)},
    {'x': (-2, 2), 'y': (2,  3)},
]

# Domain sizes
sizes = [
    {'x': N, 'y': N},
    {'x': N, 'y': N},
    {'x': N, 'y': 2*N-1},  # account for overlap
    {'x': 2*N-1, 'y':N//2},
]

# Size offsets
offsets = [
    [0, 0],
    [0, N],
    [N, 0],
    [0, 2*N-1],
]

x = np.linspace(*bounds[rank]['x'], sizes[rank]['x'])
y = np.linspace(*bounds[rank]['y'], sizes[rank]['y'])

out_name = 'parallel_mpi.pvtr'

xx, yy = np.meshgrid(x, y, indexing='ij', sparse=True)
r = np.sqrt(xx**2 + yy**2)
data = np.exp(-r**2)
proc = np.ones_like(r) * rank

with PRectilinearGrid(out_name, (x, y), offsets[rank]) as rect:
    rect.addPointData(DataArray(data, range(2), 'gaussian'))
    rect.addPointData(DataArray(proc, range(2), 'proc'))
