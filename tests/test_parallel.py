"Tests for parallel writers"
import numpy as np
import pytest
import vtk

from mpi4py import MPI
from vtk.util.numpy_support import vtk_to_numpy
from vtk import vtkXMLPRectilinearGridReader, vtkXMLPImageDataReader
from conftest import get_vtk_data, clean
from numpy import all

from uvw.parallel import PRectilinearGrid, PImageData
from uvw import DataArray


@pytest.mark.mpi_skip
def test_prectilinear_grid(field_data,
                           compression_fixture,
                           format_fixture,
                           ordering_fixture):
    coords, r, e_r, field, order = field_data
    dim = r.ndim
    out_name = 'test_prectilinear_grid.pvtr'

    compress = compression_fixture.param
    vtk_format = format_fixture.param
    rect = PRectilinearGrid(out_name,
                            coords, dim * [0],
                            compression=compress,
                            byte_order=order)
    rect.addPointData(
        DataArray(
            r, range(dim), 'point', components_order=ordering_fixture.param
        ),
        vtk_format=vtk_format,
    ).addCellData(
        DataArray(
            e_r, range(dim), 'cell', components_order=ordering_fixture.param
        ),
        vtk_format=vtk_format,
    ).addFieldData(
        DataArray(
            field, [0], 'field', components_order=ordering_fixture.param
        ),
        vtk_format=vtk_format,
    ).write()

    reader = vtkXMLPRectilinearGridReader()
    vtk_r, vtk_e_r, vtk_f = get_vtk_data(reader, out_name)

    vtk_r = vtk_r.reshape(r.shape, order='F')
    vtk_e_r = vtk_e_r.reshape(e_r.shape, order='F') \
                     .transpose(ordering_fixture.transp(dim))

    assert all(vtk_r == r)
    assert all(vtk_e_r == e_r)
    assert all(vtk_f == field)

    clean(rect)


@pytest.mark.mpi_skip
def test_pimagedata(field_data,
                    compression_fixture,
                    format_fixture,
                    ordering_fixture):
    coords, r, e_r, field, order = field_data
    dim = r.ndim
    out_name = 'test_pimagedata.pvti'

    compress = compression_fixture.param
    vtk_format = format_fixture.param
    rect = PImageData(out_name,
                      [(x.min(), x.max()) for x in coords],
                      [x.size for x in coords],
                      offsets=len(coords) * [0],
                      compression=compress,
                      byte_order=order)
    rect.addPointData(
        DataArray(
            r, range(dim), 'point', components_order=ordering_fixture.param
        ),
        vtk_format=vtk_format,
    ).addCellData(
        DataArray(
            e_r, range(dim), 'cell', components_order=ordering_fixture.param
        ),
        vtk_format=vtk_format,
    ).addFieldData(
        DataArray(
            field, [0], 'field', components_order=ordering_fixture.param
        ),
        vtk_format=vtk_format,
    ).write()

    reader = vtkXMLPImageDataReader()
    vtk_r, vtk_e_r, vtk_f = get_vtk_data(reader, out_name)

    vtk_r = vtk_r.reshape(r.shape, order='F')
    vtk_e_r = vtk_e_r.reshape(e_r.shape, order='F') \
                     .transpose(ordering_fixture.transp(dim))

    assert all(vtk_r == r)
    assert all(vtk_e_r == e_r)
    assert all(vtk_f == field)

    clean(rect)


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
    vtk_format = format_fixture.param
    rect = PRectilinearGrid(out_name, (x, y, z), offsets[rank],
                            compression=compress)
    rect.addPointData(DataArray(r, range(3), 'R'), vtk_format=vtk_format)
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

    clean(rect)


if __name__ == '__main__':
    test_prectilinear_grid_mpi()
