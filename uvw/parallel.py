"""
Module with MPI-empowered classes for parallel VTK file types.
"""
import functools

from os.path import splitext
from mpi4py import MPI

from . import writer
from . import vtk_files

from .data_array import DataArray


MASTER_RANK = 0


def MPIWrapper(cls):
    "Wrap VTKFile functions with MPI variants"
    orig_addPointData = cls.addPointData
    orig_addCellData = cls.addCellData
    orig_addFieldData = cls.addFieldData
    orig_write = cls.write

    def addPointData(self, array, *args, **kwargs):
        if self.rank == MASTER_RANK:
            self.ppoint_data.registerPDataArray(array, *args, **kwargs)
        return orig_addPointData(self, array, *args, **kwargs)

    def addCellData(self, array, *args, **kwargs):
        if self.rank == MASTER_RANK:
            self.pcell_data.registerPDataArray(array, *args, **kwargs)
        return orig_addCellData(self, array, *args, **kwargs)

    def addFieldData(self, array, *args, **kwargs):
        if self.rank == MASTER_RANK:
            self.pfield_data.registerPDataArray(array, *args, **kwargs)
        return orig_addFieldData(self, array, *args, **kwargs)

    def write(self):
        orig_write(self)
        if self.rank == MASTER_RANK:
            self.pwriter.write(self.pfilename)

    cls.addPointData = addPointData
    cls.addCellData = addCellData
    cls.addFieldData = addFieldData
    cls.write = write
    return cls


@MPIWrapper
class PRectilinearGrid(vtk_files.RectilinearGrid):
    """Rectilinear grid parallel writer"""

    parent = vtk_files.RectilinearGrid

    def __init__(self, filename, coordinates, offset, **kwargs):
        self.pfilename = filename
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Construct name of piece file
        extension = splitext(filename)[1]
        piece_extension = extension.replace('p', '')
        self.piece_name_template = (
            filename.replace(extension, '.rank{rank}') + piece_extension
        )

        # Name of piece file associated to rank
        self.parent.__init__(
            self,
            self.piece_name_template.format(rank=self.rank),
            coordinates,
            offset,
            **kwargs
        )

        # Gather local data extents
        extents = self.comm.gather(self.extent, root=MASTER_RANK)
        self.init_master(extents)

    def init_master(self, extents):
        if not extents:
            return

        self.pwriter = writer.Writer('P' + self.filetype)
        data_node = self.pwriter.data_node
        self.ppoint_data = self.pwriter.registerComponent('PPointData',
                                                          data_node)
        self.pcell_data = self.pwriter.registerComponent('PCellData',
                                                         data_node)
        self.pfield_data = self.pwriter.registerComponent('PFieldData',
                                                          data_node)
        # Register coordinates
        pcoordinates = self.pwriter.registerComponent('PCoordinates',
                                                      data_node)

        for rank, extent in enumerate(extents):
            self.pwriter.registerPiece({
                'Source': self.piece_name_template.format(rank=rank),
                'Extent': extent
            })

        # Convert string extents to list of integers
        extents = list(zip(*map(lambda x: list(map(int, x.split())), extents)))
        mins = list(map(min, extents[0::2]))
        maxs = list(map(max, extents[1::2]))

        extent = functools.reduce(
            lambda x, y: x + "{} {} ".format(y[0], y[0]+y[1]),
            zip(mins, maxs),
            ""
        )

        self.pwriter.setDataNodeAttributes({
            'WholeExtent': extent
        })

        for coord, prefix in zip(self.coordinates, ('x', 'y', 'z')):
            array = DataArray(coord, [0], prefix + '_coordinates')
            pcoordinates.registerPDataArray(array)
