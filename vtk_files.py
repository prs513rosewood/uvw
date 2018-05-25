from writer import *
from data_array import *

import functools

class VTKFile:
    """Generic VTK file"""
    def __init__(self, filename, filetype, rank=None):
        self.filename = filename
        self.rank = rank
        self.writer = Writer(filetype)

    def addPointData(self, data_array):
        self.point_data.registerDataArray(data_array)

    def addCellData(self, data_array):
        self.cell_data.registerDataArray(data_array)

    def write(self):
        self.writer.registerAppend()
        self.writer.write(self.filename)


class RectilinearGrid(VTKFile):
    """VTK Rectilinear grid (coordinates are given by 3 seperate ranges)"""
    def __init__(self, filename, coordinates, rank=None):
        VTKFile.__init__(self, filename, 'RectilinearGrid', rank)

        # Checking that we actually have a list or tuple
        if type(coordinates).__name__ not in ('tuple', 'list'):
            coordinates = [coordinates]

        self.coordinates = list(coordinates)

        # Filling in missing coordinates
        for _ in range(len(coordinates), 3):
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
            array = DataArray(coord, [0], prefix + '_coordinates')
            coordinate_component.registerDataArray(array)

        # Registering data elements
        self.point_data = self.piece.register('PointData')
        self.cell_data = self.piece.register('CellData')
