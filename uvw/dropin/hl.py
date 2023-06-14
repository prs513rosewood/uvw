"""
Compatibility package with the high-level API of PyEVTK.

This package is meant as a drop-in replacement for high-level functions of
PyEVTK. However, it does not reproduce exactly the behavior of PyEVTK, most
notably:

- uvw.Writer keyword arguments (such as compression=True) can be passed to
  the functions
- Data fields with any number of components are accepted. In PyEVTK, data
  fields have to be either scalar or 3D by specifying a tuple of 3 Numpy
  arrays. This is inconvenient and inefficient. UVW's native DataArray object
  does not impose such requirements.
- The function imageToVTK() does not impose all dimensions to be of equal size
- Requirements on data are not enforced with assertions
- Data is written in base64 format instead of raw binary

Original version of PyEVTK by Paulo A. Herrera can be found at:

- https://github.com/paulo-herrera/PyEVTK
- https://github.com/pyscience-projects/pyevtk (corresponding to PyPI version)

Below is the copyright notice for PyEVTK:

MIT License

Copyright (c) 2010-2021 Paulo A. Herrera

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

__copyright__ = "Copyright © 2018-2023 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

from .. import vtk_files as vtk
from ..data_array import DataArray

import typing as ts
import numpy as np
from collections.abc import Sequence


DataType = ts.Mapping[
    str, ts.Union[np.ndarray,
                  ts.Tuple[np.ndarray, np.ndarray, np.ndarray]]
]


def _assemble(data):
    if isinstance(data, Sequence):
        assembled_data = np.zeros(list(data[0].shape) + [3],
                                  dtype=data[0].dtype)
        for i in range(3):
            assembled_data[..., i] = data[i]
        return assembled_data
    return data


def _add_data(cellData, pointData, fieldData, fh, dims=range(3)):
    valid_data = zip([cellData, pointData, fieldData],
                     ['addCellData', 'addPointData', 'addFieldData'])
    valid_data = filter(lambda d: d[0] is not None, valid_data)
    for data, m in valid_data:
        for k, v in data.items():
            v = _assemble(v)
            getattr(fh, m)(DataArray(v, dims, k))


def imageToVTK(
        path: str,
        origin: ts.Tuple[float, float, float] = (0.0, 0.0, 0.0),
        spacing: ts.Tuple[float, float, float] = (1.0, 1.0, 1.0),
        cellData: DataType = None,
        pointData: DataType = None,
        fieldData: DataType = None,
        **kwargs
) -> str:
    """
    Write data to an ImageData file.

    :param path: path to file without extension
    :param origin: 3-tuple giving the domain origin
    :param spacing: 3-tuple giving the point spacing
    :param cellData: dict of data defined on cells
    :param pointData: dict of data defined on points
    :param fieldData: dict of data with arbitrary support
    :param **kwargs: passed on to uvw.Writer (e.g. compression=True)
    :returns: full path of created file

    Notes:
    - domain shape is inferred from either cellData or pointData
    - unlike PyEVTK arrays do not have to have the same dimension in each
      direction
    """
    if cellData is None and pointData is None:
        raise ValueError("Cannot infer image size if cell "
                         "and point data are absent")

    def _extract_shape(value, offset, name):
        value = next(iter(value.values()))
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return [s + offset for s in value.shape]
        if isinstance(value, Sequence) and value[0].ndim == 3:
            return [s + offset for s in value[0].shape]
        raise RuntimeError(f"Invalid dimension of {name}")

    # Deducing image size
    if cellData:
        shape = _extract_shape(cellData, 1, 'cellData')
    elif pointData:
        shape = _extract_shape(pointData, 0, 'pointData')

    ranges = [(o, o + n * s) for o, n, s in zip(origin, shape, spacing)]
    filename = path + '.vti'

    with vtk.ImageData(filename, ranges, shape, **kwargs) as fh:
        _add_data(cellData, pointData, fieldData, fh)
    return filename


def rectilinearToVTK(
        path: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        cellData: DataType = None,
        pointData: DataType = None,
        fieldData: DataType = None,
        **kwargs
) -> str:
    """Write data to a RectilinearGrid.

    :param path: path to file without extension
    :param x: 1d array of x coordinates
    :param y: 1d array of y coordinates
    :param z: 1d array of z coordinates
    :param cellData: dict of data defined on cells
    :param pointData: dict of data defined on points
    :param fieldData: dict of data with arbitrary support
    :param **kwargs: passed on to uvw.Writer (e.g. compression=True)
    :returns: full path of created file

    """
    filename = path + '.vtr'
    with vtk.RectilinearGrid(filename, (x, y, z), **kwargs) as fh:
        _add_data(cellData, pointData, fieldData, fh)
    return filename


def structuredToVTK(
        path: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        cellData: DataType = None,
        pointData: DataType = None,
        fieldData: DataType = None,
        **kwargs
) -> str:
    """Write data to a RectilinearGrid.

    :param path: path to file without extension
    :param x: 3d array of x coordinates
    :param y: 3d array of y coordinates
    :param z: 3d array of z coordinates
    :param cellData: dict of data defined on cells
    :param pointData: dict of data defined on points
    :param fieldData: dict of data with arbitrary support
    :param **kwargs: passed on to uvw.Writer (e.g. compression=True)
    :returns: full path of created file

    """
    filename = path + '.vts'
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    shape = [x.shape[0], y.shape[0], z.shape[0]]

    with vtk.StructuredGrid(filename, points, shape, **kwargs) as fh:
        _add_data(cellData, pointData, fieldData, fh)
    return filename


def gridToVTK(
        path: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        cellData: DataType = None,
        pointData: DataType = None,
        fieldData: DataType = None,
        **kwargs
) -> str:
    """
    Write data to either a RectilinearGrid or StructuredGrid file.

    :param path: path to file without extension
    :param x: {1,3}-d array of x coordinates
    :param y: {1,3}-d array of y coordinates
    :param z: {1,3}-d array of z coordinates
    :param cellData: dict of data defined on cells
    :param pointData: dict of data defined on points
    :param fieldData: dict of data with arbitrary support
    :param **kwargs: passed on to uvw.Writer (e.g. compression=True)
    :returns: full path of created file

    Notes:
    - file type is infered from dimension of coordinate arrays:
      - 1D -> RectilinearGrid
      - 3D -> StructuredGrid
      3D grid coordinates can be obtained with numpy.meshgrid
    """
    dims = [x.ndim, y.ndim, z.ndim]

    if dims == [1] * 3:
        return rectilinearToVTK(path, x, y, z,
                                cellData, pointData, fieldData, **kwargs)
    if dims == [3] * 3:
        return structuredToVTK(path, x, y, z,
                               cellData, pointData, fieldData, **kwargs)

    raise ValueError(f"Wrong dimensions for x, y, z arrays: {dims}")


def pointsToVTK(
        path: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        data: DataType = None,
        fieldData: DataType = None,
        **kwargs
) -> str:
    """
    Write point set to UnstructuredGrid file.

    :param path: path to file without extension
    :param x: 1d array of x coordinates
    :param y: 1d array of y coordinates
    :param z: 1d array of z coordinates
    :param data: dict of data defined on points
    :param fieldData: dict of data with arbitrary support
    :param **kwargs: passed on to uvw.Writer (e.g. compression=True)
    :returns: full path of created file
    """
    filename = path + '.vts'

    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    connectivity = {1: np.arange(coords.shape[0], dtype=np.int)}

    with vtk.UnstructuredGrid(filename, coords, connectivity, **kwargs) as fh:
        _add_data(None, data, fieldData, fh, [0])
    return filename


def linesToVTK(
        path: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        cellData: DataType = None,
        pointData: DataType = None,
        fieldData: DataType = None,
        **kwargs
) -> str:
    """
    Write segment set to an UnstructuredGrid file.

    :param path: path to file without extension
    :param x: {1,3}-d array of x coordinates
    :param y: {1,3}-d array of y coordinates
    :param z: {1,3}-d array of z coordinates
    :param cellData: dict of data defined on cells
    :param pointData: dict of data defined on points
    :param fieldData: dict of data with arbitrary support
    :param **kwargs: passed on to uvw.Writer (e.g. compression=True)
    :returns: full path of created file

    Notes:
    - number of points should be even
    - in the final file, no point will appear twice in connectivity
      (i.e. the lines are all disconected)

    The above requirements are from pyevtk and are not restrictions of the file
    format. Use uvw.UnstructuredGrid for more flexibility.
    """
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    if points.shape[0] % 2 != 2:
        raise ValueError('Number of points should be even')

    connectivity = {
        3: np.arange(points.shape[0], dtype=np.int).reshape(points.shape[0], 2)
    }

    filename = path + '.vtu'
    with vtk.UnstructuredGrid(filename, points, connectivity, **kwargs) as fh:
        _add_data(cellData, pointData, fieldData, fh)
    return filename


def polyLinesToVTK(
        path: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        pointsPerLine: np.ndarray,
        cellData: DataType = None,
        pointData: DataType = None,
        fieldData: DataType = None,
        **kwargs
) -> str:
    """
    Write segmented lines set to an UnstructuredGrid file.

    :param path: path to file without extension
    :param x: {1,3}-d array of x coordinates
    :param y: {1,3}-d array of y coordinates
    :param z: {1,3}-d array of z coordinates
    :param pointsPerLine: number of points per segmented line
    :param cellData: dict of data defined on cells
    :param pointData: dict of data defined on points
    :param fieldData: dict of data with arbitrary support
    :param **kwargs: passed on to uvw.Writer (e.g. compression=True)
    :returns: full path of created file
    """
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    start_ids = np.add.accumulate(pointsPerLine, dtype=np.int)
    connectivity = {4: np.array([
        np.arange(N, dtype=np.int) + s
        for N, s in zip(pointsPerLine, start_ids)
    ])}

    filename = path + '.vtu'
    with vtk.UnstructuredGrid(filename, points, connectivity, **kwargs) as fh:
        _add_data(cellData, pointData, fieldData, fh)
    return filename


def unstructuredGridToVTK(
        path: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        connectivity: np.ndarray,
        offsets: np.ndarray,
        cell_types: np.ndarray,
        cellData: DataType = None,
        pointData: DataType = None,
        fieldData: DataType = None,
        **kwargs
) -> str:
    """Write UnstructuredGrid file.

    :param path: path to file without extension
    :param x: 1d array of x coordinates
    :param y: 1d array of y coordinates
    :param z: 1d array of z coordinates
    :param connectivity: 1d array of nodes for each element
    :param offsets: 1d array giving the position in connectivity of the first
                    element of each cell
    :param cell_types: 1d array giving the type of each cell
    :param cellData: dict of data defined on cells
    :param pointData: dict of data defined on points
    :param fieldData: dict of data with arbitrary support
    :param **kwargs: passed on to uvw.Writer (e.g. compression=True)
    :returns: full path of created file

    Note: this implementation does not use uvw.UnstructuredGrid because the
    function expects connectivity data in a format that uvw.UnstructuredGrid
    already converts to.

    """
    filename = path + '.vtu'
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    ncells = cell_types.size
    npoints = points.shape[0]

    with vtk.VTKFile(filename, 'UnstructuredGrid', **kwargs) as fh:
        fh.piece.settAttributes({
            'NumberOfPoints': str(npoints),
            'NumberOfCells': str(ncells),
        })

        fh.piece.register('Points').registerDataArray(
            DataArray(points, [0], 'points'),
        )

        cells_component = fh.register('Cells')
        connectivity_data = {
            "connectivity": connectivity,
            "offsets": offsets,
            "types": cell_types,
        }

        for label, array in connectivity_data.items():
            cells_component.registerDataArray(
                DataArray(array, [0], label),
            )

        _add_data(cellData, pointData, fieldData, fh)


def cylinderToVTK(
        path: str,
        x0: float,
        y0: float,
        z0: float,
        z1: float,
        radius: float,
        nlayers: int,
        npilars: int = 16,
        cellData: DataType = None,
        pointData: DataType = None,
        fieldData: DataType = None,
        **kwargs
):
    """Not yet implemented."""
    raise NotImplementedError()
