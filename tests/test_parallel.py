"Tests for parallel writers"

__copyright__ = "Copyright © 2018-2023 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from mpi4py import MPI
from vtk.util.numpy_support import vtk_to_numpy
from vtk import vtkXMLPRectilinearGridReader, vtkXMLPImageDataReader
from numpy import all

from conftest import get_vtk_data

from uvw.parallel import PRectilinearGrid, PImageData
from uvw import DataArray


def sequential_setup(uvw_type, vtk_type, filename, args, fixtures):
    (coords, r, e_r, field, order), compress, \
        vtk_format, comp_order, tmp_path = fixtures

    dim = r.ndim
    out_name = tmp_path / filename

    rect = uvw_type(out_name, *args(coords), dim * [0],
                    compression=compress,
                    byte_order=order)

    rect.addPointData(
        DataArray(
            r, range(dim), 'point', components_order=comp_order.param
        ), vtk_format=vtk_format) \
        .addCellData(DataArray(
            e_r, range(dim), 'cell', components_order=comp_order.param
        ), vtk_format=vtk_format) \
        .addFieldData(DataArray(
            field, [0], 'field', components_order=comp_order.param
        ), vtk_format=vtk_format).write()

    reader = vtk_type()
    vtk_r, vtk_e_r, vtk_f = get_vtk_data(reader, out_name)

    vtk_r = vtk_r.reshape(r.shape, order='F')
    vtk_e_r = vtk_e_r.reshape(e_r.shape, order='F') \
                     .transpose(comp_order.transp(dim))

    assert all(vtk_r == r)
    assert all(vtk_e_r == e_r)
    assert all(vtk_f == field)


@pytest.mark.mpi_skip
def test_prectilinear_grid(field_data,
                           compression_fixture,
                           format_fixture,
                           ordering_fixture,
                           tmp_path):
    sequential_setup(PRectilinearGrid,
                     vtkXMLPRectilinearGridReader,
                     'test_prectilinear_grid.pvtr',
                     lambda coords: [coords],
                     (field_data, compression_fixture.param,
                      format_fixture.param, ordering_fixture, tmp_path))


@pytest.mark.mpi_skip
def test_pimagedata(field_data,
                    compression_fixture,
                    format_fixture,
                    ordering_fixture,
                    tmp_path):
    sequential_setup(PImageData,
                     vtkXMLPImageDataReader,
                     'test_pimage_data.pvtr',
                     lambda coords: [[(x.min(), x.max()) for x in coords],
                                     [x.size for x in coords]],
                     (field_data, compression_fixture.param,
                      format_fixture.param, ordering_fixture, tmp_path))


def parallel_setup(uvw_type, vtk_type, filename, args, fixtures):
    compress, vtk_format, tmp_path = fixtures

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

    tmp_path = comm.bcast(tmp_path, root=0)
    out_name = tmp_path / filename
    print(out_name)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    r = np.sqrt(xx**2 + yy**2 + zz**2)

    rect = uvw_type(out_name, *args((x, y, z)), offsets[rank],
                    compression=compress)

    rect.addPointData(DataArray(r, range(3), 'R'), vtk_format=vtk_format)
    rect.write()

    if rank == 0:
        reader = vtk_type()
        reader.SetFileName(str(out_name))
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


@pytest.mark.mpi(min_size=2)
def test_prectilinear_grid_mpi(compression_fixture,
                               format_fixture,
                               tmp_path):
    parallel_setup(PRectilinearGrid, vtkXMLPRectilinearGridReader,
                   'test_prectilinear_grid_mpi.pvtr',
                   lambda coords: [coords],
                   (compression_fixture.param, format_fixture.param, tmp_path))


@pytest.mark.mpi(min_size=2)
def test_pimagedata_mpi(compression_fixture,
                        format_fixture,
                        tmp_path):
    parallel_setup(PImageData, vtkXMLPImageDataReader,
                   'test_pimage_data_mpi.pvti',
                   lambda coords: [[(x.min(), x.max()) for x in coords],
                                   [x.size for x in coords]],
                   (compression_fixture.param, format_fixture.param, tmp_path))


if __name__ == '__main__':
    from pathlib import Path

    class fixture_stub:
        pass

    compression = fixture_stub()
    format = fixture_stub()

    compression.param = None
    format.param = 'ascii'

    test_prectilinear_grid_mpi(compression, format, Path('.'))
