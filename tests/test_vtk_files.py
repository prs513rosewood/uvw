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


def test_rectilinear_grid(threeD_data, compression_fixture):
    coords, r, e_r = threeD_data
    f = io.StringIO()

    compress = compression_fixture.param
    with RectilinearGrid(f, coords, compression=compress) as rect:
        rect.addPointData(DataArray(r, range(3), 'point'))
        rect.addCellData(DataArray(e_r, range(3), 'cell'))

    reader = vtkXMLRectilinearGridReader()
    vtk_r, vtk_e_r = get_vtk_data(reader, f)

    vtk_r = vtk_r.reshape(r.shape, order='F')
    vtk_e_r = vtk_e_r.reshape(e_r.shape, order='F').transpose((0, 1, 2, 4, 3))

    assert all(vtk_r == r)
    assert all(vtk_e_r == e_r)


def test_image_data(threeD_data, compression_fixture):
    coords, r, e_r = threeD_data
    f = io.StringIO()

    compress = compression_fixture.param
    with ImageData(
            f,
            [(min(x), max(x)) for x in coords],
            [x.size for x in coords],
            compression=compress) as fh:
        fh.addPointData(DataArray(r, range(3), 'point'))
        fh.addCellData(DataArray(e_r, range(3), 'cell'))

    reader = vtkXMLImageDataReader()
    vtk_r, vtk_e_r = get_vtk_data(reader, f)

    vtk_r = vtk_r.reshape(r.shape, order='F')
    vtk_e_r = vtk_e_r.reshape(e_r.shape, order='F').transpose((0, 1, 2, 4, 3))

    assert all(vtk_r == r)
    assert all(vtk_e_r == e_r)


def test_structured_grid(compression_fixture):
    f = io.StringIO()

    N = 5

    r = np.linspace(0, 1, N)
    theta = np.linspace(0, 2*np.pi, 5*N)

    theta, r = np.meshgrid(theta, r, indexing='ij')

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    points = np.vstack([x.ravel(), y.ravel()]).T

    compress = compression_fixture.param
    grid = StructuredGrid(f, points, (N, 5*N), compression=compress)

    data = np.exp(-4*r**2)

    grid.addPointData(DataArray(data, reversed(range(2)), 'data'))
    grid.write()

    reader = vtkXMLStructuredGridReader()
    reader.SetReadFromInputString(True)
    reader.SetInputString(f.getvalue())
    reader.Update()

    output = reader.GetOutput()
    vtk_data = vtk_to_numpy(output.GetPointData().GetArray('data'))
    vtk_data = vtk_data.reshape(data.shape, order='C')

    assert all(vtk_data == data)
