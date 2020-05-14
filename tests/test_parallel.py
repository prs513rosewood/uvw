import numpy as np
import pytest
import vtk
import os

from mpi4py import MPI
from vtk.util.numpy_support import vtk_to_numpy

from uvw.parallel import PRectilinearGrid
from uvw import DataArray


@pytest.mark.mpi(min_size=2)
def test_prectilinear_grid(compression_fixture):
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

    compress = compression_fixture.param
    rect = PRectilinearGrid(out_name, (x, y, z), offsets[rank], 
                            compression=compress)
    rect.addPointData(DataArray(r, range(3), 'R'))
    rect.write()

    if rank == 0:
        reader = vtk.vtkXMLPRectilinearGridReader()
        reader.SetFileName(out_name)
        reader.Update()
        output = reader.GetOutput()
        vtk_r = vtk_to_numpy(output.GetPointData().GetArray('R'))
    else:
        vtk_r = None

    vtk_r = comm.bcast(vtk_r, root=0)
    vtk_r = vtk_r.reshape([4*N-1, N+2, N], order='F')

    i, j, k = [int(i) for i in offsets[rank]]

    # Adjusting for overlap
    if offsets[rank][0] != 0:
        i -= 1

    sub_vtk = vtk_r[i:i+x.size, j:j+y.size, k:k+z.size]
    assert np.all(sub_vtk == r)

    if rank == 0:
        try:
            os.remove(out_name)
            os.remove('test_prectilinear_grid_rank0.vtr')
            os.remove('test_prectilinear_grid_rank1.vtr')
        except:
            pass


if __name__ == '__main__':
    test_prectilinear_grid()
