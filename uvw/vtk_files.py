from . import writer
from . import data_array

import functools
import numpy as np


class VTKFile:
    """Generic VTK file"""
    def __init__(self, filename, filetype, rank=None):
        self.filename = filename
        self.rank = rank
        self.writer = writer.Writer(filetype)

    def addPointData(self, data_array):
        self.point_data.registerDataArray(data_array)

    def addCellData(self, data_array):
        self.cell_data.registerDataArray(data_array)

    def write(self):
        self.writer.registerAppend()
        self.writer.write(self.filename)


class ImageData(VTKFile):
    """VTK Image data (coordinates are given by a range and constant spacing)"""
    def __init__(self, filename, ranges, points, rank=None):
        VTKFile.__init__(self, filename, 'ImageData', rank)

        # Computing spacings
        spacings = [(x[1] - x[0]) / (n - 1) for x, n in zip(ranges, points)]

        # Filling in missing coordinates
        for _ in range(len(points), 3):
            points.append(1)

        # Setting extents, spacings and origin
        extent = functools.reduce(lambda x, y: x + "0 {} ".format(y-1), points, "")
        spacings = functools.reduce(lambda x, y: x + "{} ".format(y), spacings, "")
        origins = functools.reduce(lambda x, y: x + "{} ".format(y[0]), ranges, "")

        self.writer.setDataNodeAttributes({
            'WholeExtent': extent,
            'Spacing': spacings,
            'Origin': origins
        })

        self.piece = self.writer.registerPiece({
            "Extent": extent
        })

        # Registering data elements
        self.point_data = self.piece.register('PointData')
        self.cell_data = self.piece.register('CellData')


class RectilinearGrid(VTKFile):
    """VTK Rectilinear grid (coordinates are given by 3 seperate ranges)"""
    def __init__(self, filename, coordinates, rank=None):
        VTKFile.__init__(self, filename, 'RectilinearGrid', rank)

        # Checking that we actually have a list or tuple
        if type(coordinates).__name__ == 'ndarray':
            coordinates = [coordinates]

        self.coordinates = list(coordinates)

        # Filling in missing coordinates
        for _ in range(len(self.coordinates), 3):
            self.coordinates.append(np.array([0.]))

        # Setting data extent
        extent = []

        for coord in self.coordinates:
            if coord.ndim != 1:
                raise Exception('Coordinate array should have only one dimension')
            extent.append(coord.size-1)

        extent = functools.reduce(lambda x, y: x + "0 {} ".format(y), extent, "")
        self.writer.setDataNodeAttributes({
            "WholeExtent": extent
        })

        self.piece = self.writer.registerPiece({
            "Extent": extent
        })

        # Registering coordinates
        coordinate_component = self.piece.register('Coordinates')

        for coord, prefix in zip(self.coordinates, ('x', 'y', 'z')):
            array = data_array.DataArray(coord, [0], prefix + '_coordinates')
            coordinate_component.registerDataArray(array)

        # Registering data elements
        self.point_data = self.piece.register('PointData')
        self.cell_data = self.piece.register('CellData')
