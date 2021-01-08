"""
Module with MPI-empowered classes for parallel VTK file types.
"""
import functools

from os import PathLike
from os.path import splitext, basename
from mpi4py import MPI

from . import writer
from . import vtk_files

from .data_array import DataArray


MASTER_RANK = 0


def _check_file_descriptor(fd):
    if issubclass(type(fd), (str, PathLike)):
        return
    raise TypeError('Expected path, got {}'.format(fd))


class PVTKFile(vtk_files.VTKFile):
    "Generic parallel VTK file"

    def __init__(self, filename, pfiletype, **kwargs):
        """
        Create a generic VTK file

        :param filename: name of file (cannot be file handle)
        :param filetype: VTK format of file
        :param **kwargs: parameters forwarded to Writer
        """
        _check_file_descriptor(filename)
        filename = str(filename)  # Ensure we have a string
        self.pfilename = filename
        self.pfiletype = pfiletype
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Construct name of piece file
        extension = splitext(filename)[1]
        piece_extension = extension.replace('p', '')
        self.piece_name_template = (
            filename.replace(extension, '.rank{rank}') + piece_extension
        )

        # We deliberatly do not call VTKFile.__init__

        self.pwriter = writer.Writer(self.pfiletype)
        self.ppoint_data, self.pcell_data, self.pfield_data = [
            self.pwriter.registerComponent(label, self.pwriter.data_node)
            for label in "PPointData PCellData PFieldData".split()
        ]

    def addPointData(self, array, *args, **kwargs):
        if self.rank == MASTER_RANK:
            self.ppoint_data.registerPDataArray(array, *args, **kwargs)
        return vtk_files.VTKFile.addPointData(self, array, *args, **kwargs)

    def addCellData(self, array, *args, **kwargs):
        if self.rank == MASTER_RANK:
            self.pcell_data.registerPDataArray(array, *args, **kwargs)
        return vtk_files.VTKFile.addCellData(self, array, *args, **kwargs)

    def addFieldData(self, array, *args, **kwargs):
        if self.rank == MASTER_RANK:
            self.pfield_data.registerPDataArray(array, *args, **kwargs)
        return vtk_files.VTKFile.addFieldData(self, array, *args, **kwargs)

    def write(self):
        vtk_files.VTKFile.write(self)
        if self.rank == MASTER_RANK:
            self.pwriter.write(self.pfilename)


class PRectilinearGrid(PVTKFile, vtk_files.RectilinearGrid):
    """Rectilinear grid parallel writer"""

    parent = vtk_files.RectilinearGrid

    def __init__(self, filename, coordinates, offset, **kwargs):
        """
        Init the parallel counterpart of a RectilinearGrid file

        :param filename: name of file or file handle
        :param coordinates: list of coordinate arrays for this rank
        :param offset: offset in global dataset for this rank
        """
        PVTKFile.__init__(self, filename, 'PRectilinearGrid', **kwargs)

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

        # Ensure on master
        if not extents:
            return

        # Register coordinates
        pcoordinates = self.pwriter.registerComponent('PCoordinates',
                                                      self.pwriter.data_node)

        for rank, extent in enumerate(extents):
            self.pwriter.registerPiece({
                'Source': basename(self.piece_name_template.format(rank=rank)),
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
