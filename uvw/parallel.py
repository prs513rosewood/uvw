from . import writer
from . import data_array
from . import vtk_files

import functools

from os.path import splitext
from mpi4py import MPI

MASTER_RANK = 0


class PRectilinearGrid(vtk_files.RectilinearGrid):
    """Rectilinear grid parallel writer"""

    parent = vtk_files.RectilinearGrid

    def __init__(self, filename, coordinates, offset, compression=None):
        self.pfilename = filename
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Construct name of piece file
        fragmented_name = splitext(filename)
        self.piece_name_template = (fragmented_name[0] +
                                    '_rank{rank}.' + fragmented_name[1][2:])

        # Name of piece file associated to rank
        self.parent.__init__(
            self,
            self.piece_name_template.format(rank=self.rank),
            coordinates,
            offset,
            compression
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

        # Register coordinates
        pcoordinates = self.pwriter.registerComponent('PCoordinates',
                                                      data_node)

        for coord, prefix in zip(self.coordinates, ('x', 'y', 'z')):
            array = data_array.DataArray(coord, [0], prefix + '_coordinates')
            pcoordinates.registerPDataArray(array)

    def addPointData(self, array):
        self.parent.addPointData(self, array)
        if self.rank == MASTER_RANK:
            self.ppoint_data.registerPDataArray(array)

    def addCellData(self, array):
        self.parent.addCellData(self, array)
        if self.rank == MASTER_RANK:
            self.pcell_data.registerPDataArray(array)

    def write(self):
        self.parent.write(self)
        if self.rank == MASTER_RANK:
            self.pwriter.write(self.pfilename)
