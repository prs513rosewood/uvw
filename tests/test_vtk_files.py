import io
import numpy as np

from numpy import all, min, max
from vtk import (
    vtkXMLRectilinearGridReader,
    vtkXMLImageDataReader,
    vtkXMLStructuredGridReader,
)
from vtk.util.numpy_support import vtk_to_numpy
from conftest import get_vtk_data

from uvw import (
    ImageData,
    RectilinearGrid,
    StructuredGrid,
    DataArray,
)


def test_rectilinear_grid(field_data,
                          compression_fixture,
                          format_fixture,
                          ordering_fixture):
    coords, r, e_r, field = field_data
    dim = r.ndim
    f = io.StringIO()

    compress = compression_fixture.param
    format = format_fixture.param
    rect = RectilinearGrid(f, coords, compression=compress)
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
    coords, r, e_r, field = field_data
    dim = r.ndim
    f = io.StringIO()

    compress = compression_fixture.param
    format = format_fixture.param
    with ImageData(
            f,
            [(min(x), max(x)) for x in coords],
            [x.size for x in coords],
            compression=compress) as fh:
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
