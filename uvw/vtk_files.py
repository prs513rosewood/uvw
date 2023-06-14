"""Module with classes corresponding to VTK file types.

Currently, the following types are implemented:
- ImageData
- RectilinearGrid
- StructuredGrid
- UnstructuredGrid

The ParaViewData file format, although not part of VTK, is also available.

All instances of classes in this file are context managers and can be used in
the idiomatic:

  with RectilinearGrid(...) as file:
      file.addPointData(...)

"""

__copyright__ = "Copyright © 2018-2023 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

import functools
import typing as ts
import numpy as np

from . import writer
from .data_array import DataArray
from .unstructured import CellType, check_connectivity


def _make_3darray(points):
    """Complete missing coordinates to get 3d points."""
    if points.shape[1] == 3:
        return points

    points_3d = np.zeros((points.shape[0], 3))
    for i in range(min(points.shape[1], 3)):
        points_3d[:, i] = points[:, i]
    return points_3d


def _fold_extent(extent, offsets, dimension):
    if offsets is None:
        offsets = [0] * len(extent)
    elif len(offsets) == dimension:
        offsets = list(offsets)
        offsets += [0] * (len(extent) - len(offsets))
    else:
        raise ValueError(
            'Size of offsets should '
            f'match domain dimension {dimension}')

    def fold_extent(acc, couple):
        offset, extent = couple
        offset -= offset != 0
        return acc + f"{offset} {offset+extent} "

    # Create extent string with offsets
    return functools.reduce(fold_extent, zip(offsets, extent), "")


class WriteManager:
    """Context manager that writes on exit."""

    def write(self):
        """Write file."""
        raise NotImplementedError()

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, err_type, *_):
        """Exit context and handle errors."""
        if err_type is None:
            # Only write if no error
            self.write()
        return True


class VTKFile(WriteManager):
    """Generic VTK file.

    Base class of all VTK-formated files. Sets up the common XML tree with
    PointData and CellData. Also supplies the functions to add point, cell and
    field data.

    """

    _FileDescriptor = writer.Writer.FileDescriptor

    def __init__(self, filename: _FileDescriptor, filetype: str, **kwargs):
        """
        Create a generic VTK file.

        :param filename: name of file or file handle
        :param filetype: VTK format of file
        :param **kwargs: parameters forwarded to Writer
        """
        self.filetype = filetype
        self.filename = filename
        self.writer = writer.Writer(filetype, **kwargs)

        # Center piece
        self.piece = self.writer.registerPiece()

        # Registering data elements
        self.point_data = self.piece.register('PointData')
        self.cell_data = self.piece.register('CellData')
        self.field_data = self.writer.registerComponent(
            'FieldData', self.writer.data_node
        )

    def addPointData(self, data_array: DataArray, vtk_format: str = 'binary'):
        """
        Add a DataArray instance to the PointData section of the file.

        :param data_array: DataArray instance
        :param vtk_format: data format. Can be:
            - ``'ascii'``: data is written with numpy.savetxt in-place
            - ``'binary'``: data is written in base64 (with possible
            compression) in-place
            - ``'append'``: data is written in base64 (with possible
            compression) in the ``AppendData`` section of the file
        """
        self.point_data.registerDataArray(data_array, vtk_format)
        return self

    def addCellData(self, data_array: DataArray, vtk_format: str = 'binary'):
        """
        Add a DataArray instance to the PointData section of the file.

        Arguments are identical to `addPointData`.
        """
        self.cell_data.registerDataArray(data_array, vtk_format)
        return self

    def addFieldData(self, data_array: DataArray, vtk_format: str = 'binary'):
        """
        Add a DataArray instance to the FieldData section of the file.

        Arguments are identical to `addPointData`.

        The FieldData section contains arrays that are not related to
        the domain geometry. Their shape is therefore freeform.
        """
        self.field_data.registerDataArray(data_array, vtk_format)
        return self

    def write(self):
        """Write XML to file."""
        self.writer.registerAppend()
        self.writer.write(self.filename)


class ImageData(VTKFile):
    """VTK Image data (coordinates given by a range and constant spacing)."""

    def __init__(self,
                 filename: VTKFile._FileDescriptor,
                 ranges: ts.List[ts.Tuple[float, float]],
                 points: ts.List[int],
                 offsets: ts.List[int] = None, **kwargs):
        """
        Init an ImageData file (regular orthogonal grid).

        :param filename: name of file or file handle
        :param ranges: list of pairs for coordinate ranges
        :param points: list of number of points
        """
        VTKFile.__init__(self, filename, 'ImageData', **kwargs)

        # Computing spacings
        spacings = [(x[1] - x[0]) / (n - 1) for x, n in zip(ranges, points)]

        if offsets is None:
            offsets = [0] * len(points)

        # Filling in missing coordinates
        points += [1] * max(3 - len(points), 0)
        offsets += [0] * max(3 - len(offsets), 0)

        # Setting extents, spacing and origin
        self.extent = _fold_extent([x - 1 for x in points],
                                   offsets, len(points))
        spacings = functools.reduce(lambda x, y: x + f"{y} ", spacings, "")
        origins = functools.reduce(lambda x, y: x + f"{y[0]} ", ranges, "")

        self.writer.setDataNodeAttributes({
            'WholeExtent': self.extent,
            'Spacing': spacings,
            'Origin': origins
        })

        self.piece.setAttributes({
            "Extent": self.extent
        })


class RectilinearGrid(VTKFile):
    """VTK Rectilinear grid (coordinates are given by 3 seperate arrays)."""

    def __init__(self,
                 filename: VTKFile._FileDescriptor,
                 coordinates: ts.Union[ts.Iterable[np.ndarray], np.ndarray],
                 offsets: ts.List[int] = None, **kwargs):
        """
        Init an RectilinearGrid file (irregular orthogonal grid).

        :param filename: name of file or file handle
        :param coordinates: list of coordinates for each direction
        """
        VTKFile.__init__(self, filename, 'RectilinearGrid', **kwargs)

        # Checking that we actually have a list or tuple
        if isinstance(coordinates, np.ndarray):
            coordinates = [coordinates]

        self.coordinates = list(coordinates)
        # Filling in missing coordinates
        self.coordinates += (
            [np.array([0.])] * max(0, 3 - len(self.coordinates))
        )

        # Setting data extent
        extent = []

        for coord in self.coordinates:
            if coord.ndim != 1:
                raise ValueError(
                    'Coordinate array should have only one dimension'
                    f' (has {coord.ndim})')
            extent.append(coord.size-1)

        # Create extent string with offsets
        self.extent = _fold_extent(extent, offsets, len(coordinates))

        self.writer.setDataNodeAttributes({
            "WholeExtent": self.extent
        })

        self.piece.setAttributes({
            "Extent": self.extent
        })

        # Registering coordinates
        coordinate_component = self.piece.register('Coordinates')

        for coord, prefix in zip(self.coordinates, ('x', 'y', 'z')):
            array = DataArray(coord, [0], prefix + '_coordinates')
            coordinate_component.registerDataArray(array, vtk_format='append')


class StructuredGrid(VTKFile):
    """VTK Structured grid (coordinates given by a single array of points)."""

    def __init__(self,
                 filename: VTKFile._FileDescriptor,
                 points: np.ndarray,
                 shape: ts.List[int], **kwargs):
        """
        Init a StructuredGrid file (ordered quadrangle/hexahedron cells).

        :param filename: name of file or file handle
        :param points: 2D numpy array of point coordinates
        :param shape: number of points in each spatial direction
        """
        VTKFile.__init__(self, filename, 'StructuredGrid', **kwargs)

        if points.ndim != 2:
            raise ValueError('Points should be a 2D array')

        # Completing the missing coordinates
        points = _make_3darray(points)

        extent = [n - 1 for n in shape]
        extent += [0] * max(3 - len(extent), 0)

        extent = functools.reduce(lambda x, y: x + f"0 {y} ", extent, "")
        self.writer.setDataNodeAttributes({
            "WholeExtent": extent
        })

        self.piece.setAttributes({
            "Extent": extent
        })

        points_component = self.piece.register('Points')
        points_component.registerDataArray(
            DataArray(points, [0], 'points'), vtk_format='append',
        )


class UnstructuredGrid(VTKFile):
    """VTK Unstructured grid (data on nodes + connectivity)."""

    def __init__(self,
                 filename: VTKFile._FileDescriptor,
                 nodes: np.ndarray,
                 connectivity: ts.Mapping[int, np.ndarray], **kwargs):
        """
        Init an UnstructuredGrid file (mesh with connectivity).

        :param filename: name of file or file handle
        :param nodes: 2D numpy array of node coordinates
        :param connectivity: dict with arrays for each cell type
        """
        VTKFile.__init__(self, filename, 'UnstructuredGrid', **kwargs)

        if nodes.ndim != 2:
            raise ValueError('Nodes should be a 2D array')

        # Completing the missing coordinates
        nodes = _make_3darray(nodes)

        if not check_connectivity(connectivity):
            raise ValueError('Connectivity is invalid')

        cells_component = self.piece.register('Cells')

        ncells = sum(map(len, connectivity.values()))

        self.piece.setAttributes({
            'NumberOfPoints': str(nodes.shape[0]),
            'NumberOfCells': str(ncells),
        })

        self.piece.register('Points').registerDataArray(
            DataArray(nodes, [0], 'points'), vtk_format='append'
        )

        int32 = np.dtype('i4')

        flat_connectivity = np.empty(
            sum(len(cell) for x in connectivity.values() for cell in x),
            dtype=int32,
        )

        # Flattening the connectivities for each element type
        offset = 0
        for conn in connectivity.values():
            for cell in conn:
                flat_connectivity[offset:offset+len(cell)] = cell
                offset += len(cell)

        offsets = np.empty(ncells, dtype=int32)

        offset, index = 0, 0
        for conn in connectivity.values():
            for cell in conn:
                offset += len(cell)
                offsets[index] = offset
                index += 1

        types = np.empty(ncells, dtype=int32)

        offset = 0
        for k, conn in connectivity.items():
            if isinstance(k, CellType):
                k = k.value
            types[offset:offset+len(conn)] = k
            offset += len(conn)

        # Register mesh description
        connectivity_data = {
            "connectivity": flat_connectivity,
            "offsets": offsets,
            "types": types,
        }

        for label, array in connectivity_data.items():
            cells_component.registerDataArray(
                DataArray(array, [0], label),
                vtk_format='append',
            )


class ParaViewData(WriteManager):
    """
    Groups VTK files into a single description.

    See: https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
    """

    def __init__(self, filename: str, **kwargs):
        """
        Initialize a PVD file.

        PVD files contain references to other data files, and are convenient
        to represent time series, for example.
        """
        self.filename = filename
        self.writer = writer.Writer('Collection', **kwargs)

    def addFile(self, file, timestep=0, group="", part=0):
        """
        Add a file to the group.

        :param file: filename or VTKFile instance
        :param timestep: real-time value of file
        :param group: group to add the file to
        :param part: sub-part of the domain represented by file
        """
        if isinstance(file, VTKFile):
            file = file.filename

        self.writer.registerComponent('DataSet', self.writer.data_node, dict(
            timestep=str(timestep), group=group, part=str(part), file=str(file)
        ))

    def write(self):
        """Write file."""
        self.writer.write(self.filename)
