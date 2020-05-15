import numpy as np
import pytest
import vtk
import os

from mpi4py import MPI
from vtk.util.numpy_support import vtk_to_numpy
from vtk import vtkXMLPRectilinearGridReader
from conftest import get_vtk_data, transp
from numpy import all

from uvw.parallel import PRectilinearGrid
from uvw import DataArray


def test_prectilinear_grid(field_data, compression_fixture, format_fixture):
    coords, r, e_r = field_data
    dim = r.ndim
    out_name = 'test_prectilinear_grid.pvtr'

    compress = compression_fixture.param
    format = format_fixture.param
    with PRectilinearGrid(out_name,
                          coords, dim * [0], compression=compress) as rect:
        rect.init_master(None)  # useless here: for coverage only
        rect.addPointData(DataArray(r, range(dim), 'point'), vtk_format=format)
        rect.addCellData(DataArray(e_r, range(dim), 'cell'), vtk_format=format)

    reader = vtkXMLPRectilinearGridReader()
    vtk_r, vtk_e_r = get_vtk_data(reader, out_name)

    vtk_r = vtk_r.reshape(r.shape, order='F')
    vtk_e_r = vtk_e_r.reshape(e_r.shape, order='F').transpose(transp(dim))

    assert all(vtk_r == r)
    assert all(vtk_e_r == e_r)

    os.remove(out_name)
    os.remove('test_prectilinear_grid_rank0.vtr')


@pytest.mark.mpi(min_size=2)
def test_prectilinear_grid_mpi(compression_fixture, format_fixture):
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

    out_name = 'test_prectilinear_grid_mpi.pvtr'

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    r = np.sqrt(xx**2 + yy**2 + zz**2)

    compress = compression_fixture.param
    format = format_fixture.param
    rect = PRectilinearGrid(out_name, (x, y, z), offsets[rank],
                            compression=compress)
    rect.addPointData(DataArray(r, range(3), 'R'), vtk_format=format)
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
            os.remove('test_prectilinear_grid_mpi_rank0.vtr')
            os.remove('test_prectilinear_grid_mpi_rank1.vtr')
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    test_prectilinear_grid()
