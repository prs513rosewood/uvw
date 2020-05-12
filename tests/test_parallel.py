import numpy as np
import pytest

from mpi4py import MPI

from uvw.parallel import PRectilinearGrid
from uvw import DataArray


@pytest.mark.mpi(min_size=2)
def test_prectilinear_grid():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    N = 3

    bounds = [
        (0, 1),
        (1, 2)
    ]

    offsets = [
        [0, 0, 0],
        [2*N, 0, 0],
    ]

    x = np.linspace(*bounds[rank], N*2)
    y = np.linspace(0, 1, N+2)
    z = np.linspace(0, 1, N)

    out_name = 'test_prectilinear_grid.pvtr'

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    r = np.sqrt(xx**2 + yy**2 + zz**2)

    rect = PRectilinearGrid(out_name, (x, y, z), offsets[rank])
    rect.addPointData(DataArray(r, range(3), 'R'))
    rect.write()

    if rank == 0:
        output = open(out_name, 'r')
        reference = open('test_prectilinear_grid.ref', 'r')

        assert output.read() == reference.read()


if __name__ == '__main__':
    test_prectilinear_grid()
