__copyright__ = "Copyright © 2018-2023 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

import io
import numpy as np

from numpy import all, min, max
from vtk import (
    vtkXMLRectilinearGridReader,
    vtkXMLImageDataReader,
    vtkXMLStructuredGridReader,
    vtkXMLUnstructuredGridReader,
)
from vtk.util.numpy_support import vtk_to_numpy
from conftest import get_vtk_data

from uvw import (
    ImageData,
    RectilinearGrid,
    StructuredGrid,
    UnstructuredGrid,
    ParaViewData,
    DataArray,
)

from uvw.unstructured import CellType


def test_rectilinear_grid(field_data,
                          compression_fixture,
                          format_fixture,
                          ordering_fixture):
    coords, r, e_r, field, order = field_data
    dim = r.ndim
    f = io.StringIO()

    compress = compression_fixture.param
    format = format_fixture.param
    rect = RectilinearGrid(f, coords, compression=compress, byte_order=order)
    rect.addPointData(
        DataArray(r, range(dim), 'point', ordering_fixture.param),
        vtk_format=format
    ).addCellData(
        DataArray(e_r, range(dim), 'cell', ordering_fixture.param),
        vtk_format=format
    ).addFieldData(
        DataArray(field, [0], 'field', ordering_fixture.param),
        vtk_format=format
    )

    rect.write()

    reader = vtkXMLRectilinearGridReader()

    # Testing the xml pretty print output as well
    pretty_sstream = io.StringIO(str(rect.writer))

    for ss in [f, pretty_sstream]:
        vtk_r, vtk_e_r, vtk_f = get_vtk_data(reader, ss)

        vtk_r = vtk_r.reshape(r.shape, order='F')
        vtk_e_r = vtk_e_r.reshape(e_r.shape, order='F') \
                         .transpose(ordering_fixture.transp(dim))

        assert all(vtk_r == r)
        assert all(vtk_e_r == e_r)
        assert all(vtk_f == field)


def test_image_data(field_data,
                    compression_fixture,
                    format_fixture,
                    ordering_fixture):
    coords, r, e_r, field, order = field_data
    dim = r.ndim
    f = io.StringIO()

    compress = compression_fixture.param
    format = format_fixture.param
    with ImageData(
            f,
            [(min(x), max(x)) for x in coords],
            [x.size for x in coords],
            compression=compress,
            byte_order=order) as fh:
        fh.addPointData(
            DataArray(r, range(dim), 'point', ordering_fixture.param),
            vtk_format=format
        ).addCellData(
            DataArray(e_r, range(dim), 'cell', ordering_fixture.param),
            vtk_format=format
        ).addFieldData(
            DataArray(field, [0], 'field', ordering_fixture.param),
            vtk_format=format
        )

    reader = vtkXMLImageDataReader()
    vtk_r, vtk_e_r, vtk_f = get_vtk_data(reader, f)

    vtk_r = vtk_r.reshape(r.shape, order='F')
    vtk_e_r = vtk_e_r.reshape(e_r.shape, order='F') \
                     .transpose(ordering_fixture.transp(dim))

    assert all(vtk_r == r)
    assert all(vtk_e_r == e_r)
    assert all(vtk_f == field)


def test_structured_grid(compression_fixture, format_fixture):
    f = io.StringIO()

    N = 5

    r = np.linspace(0, 1, N)
    theta = np.linspace(0, 2*np.pi, 5*N)

    theta, r = np.meshgrid(theta, r, indexing='ij')

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    points = np.vstack([x.ravel(), y.ravel()]).T

    compress = compression_fixture.param
    format = format_fixture.param
    grid = StructuredGrid(f, points, (N, 5*N), compression=compress)

    data = np.exp(-4*r**2)

    grid.addPointData(DataArray(data, reversed(range(2)), 'data'),
                      vtk_format=format)
    grid.write()

    reader = vtkXMLStructuredGridReader()
    reader.SetReadFromInputString(True)
    reader.SetInputString(f.getvalue())
    reader.Update()

    output = reader.GetOutput()
    vtk_data = vtk_to_numpy(output.GetPointData().GetArray('data'))
    vtk_data = vtk_data.reshape(data.shape, order='C')

    assert all(vtk_data == data)


def test_unstructured_grid(compression_fixture, format_fixture):
    f = io.StringIO()
    compress = compression_fixture.param
    format = format_fixture.param

    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])

    point_data = np.array([[0, 1], [1, 2], [2, 3]])
    cell_data = np.array([1, 2, 3, 4])

    connectivity = {
        CellType.TRIANGLE: np.array([range(3)], dtype=np.int32),
        6: np.array([[0, 1, 2]], dtype=np.int32),  # Testing true VTK type id
        CellType.POLY_LINE: np.array([  # Testing variable length cell type
            [0, 1],
            [1, 2, 0],
        ], dtype=object),
    }

    grid = UnstructuredGrid(f, nodes, connectivity, compression=compress)
    grid.addPointData(DataArray(point_data, [0], 'point'), vtk_format=format)
    grid.addCellData(DataArray(cell_data, [0], 'cell'), vtk_format=format)
    grid.write()

    reader = vtkXMLUnstructuredGridReader()
    reader.SetReadFromInputString(True)
    reader.SetInputString(f.getvalue())
    reader.Update()

    vtk_pdata = vtk_to_numpy(
        reader.GetOutput().GetPointData().GetArray('point'))
    vtk_cdata = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray('cell'))
    vtk_pdata.reshape(point_data.shape, order='C')
    vtk_cdata.reshape(cell_data.shape, order='C')
    assert all(vtk_pdata == point_data)
    assert all(vtk_cdata == cell_data)


def test_paraview_data(tmp_path):
    """
    NB: This is just testing writing. Since PVD is a ParaView related extension,
    it cannot be tested with vanilla VTK
    """
    x = np.linspace(0, 1, 10)
    y = x.copy()

    grid = RectilinearGrid(tmp_path / 'grid.vtr', [x, y])
    grid.write()

    group = ParaViewData(tmp_path / 'grid.pvd')
    group.addFile(grid)
    group.write()
