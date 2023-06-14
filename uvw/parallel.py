"""Module with MPI-empowered classes for parallel VTK file types."""

__copyright__ = "Copyright © 2018-2023 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

import functools

from os import PathLike
from os.path import splitext, basename
from mpi4py import MPI

from . import writer
from . import vtk_files

from .data_array import DataArray


MASTER_RANK = 0


def _check_file_descriptor(fd):
    """Check path argument."""
    if not isinstance(fd, (str, PathLike)):
        raise TypeError(f'Expected path, got {fd}')


def _str_convert(ex):
    """Convert list of string extents to list of tuples w/ mins and maxs."""
    return list(zip(*((int(x) for x in s.split()) for s in ex)))


def _min_max_reduce(extents):
    """Reduce local extents to global extent by looking for mins and maxs."""
    mins = map(min, extents[0::2])
    maxs = map(max, extents[1::2])

    return functools.reduce(
        lambda x, y: x + f"{y[0]} {y[0]+y[1]} ",
        zip(mins, maxs),
        ""
    )


class PVTKFile(vtk_files.VTKFile):
    """Generic parallel VTK file."""

    def __init__(self, filename, pfiletype, comm=None, **kwargs):
        """
        Create a generic VTK file.

        :param filename: name of file (cannot be file handle)
        :param pfiletype: VTK format of file
        :param comm: MPI communicator (default: MPI.COMM_WORLD)
        :param **kwargs: parameters forwarded to Writer
        """
        _check_file_descriptor(filename)
        filename = str(filename)  # Ensure we have a string
        self.pfilename = filename
        self.pfiletype = pfiletype
        self.comm = MPI.COMM_WORLD if comm is None else comm
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
        """Add point data."""
        if self.rank == MASTER_RANK:
            self.ppoint_data.registerPDataArray(array, *args, **kwargs)
        return vtk_files.VTKFile.addPointData(self, array, *args, **kwargs)

    def addCellData(self, array, *args, **kwargs):
        """Add cell data."""
        if self.rank == MASTER_RANK:
            self.pcell_data.registerPDataArray(array, *args, **kwargs)
        return vtk_files.VTKFile.addCellData(self, array, *args, **kwargs)

    def addFieldData(self, array, *args, **kwargs):
        """Add field data."""
        if self.rank == MASTER_RANK:
            self.pfield_data.registerPDataArray(array, *args, **kwargs)
        return vtk_files.VTKFile.addFieldData(self, array, *args, **kwargs)

    def write(self):
        """Write file."""
        vtk_files.VTKFile.write(self)
        if self.rank == MASTER_RANK:
            self.pwriter.write(self.pfilename)


class PImageData(PVTKFile, vtk_files.ImageData):
    """Image data parallel writer."""

    parent = vtk_files.ImageData

    def __init__(self, filename, ranges, points, offsets, comm=None, **kwargs):
        """
        Init an ImageData file (regular orthogonal grid).

        :param filename: name of file or file handle
        :param ranges: list of pairs for coordinate ranges
        :param points: list of number of points
        :param comm: MPI communicator (default: MPI.COMM_WORLD)
        """
        PVTKFile.__init__(self, filename, 'PImageData', comm=comm, **kwargs)
        self.parent.__init__(
            self,
            self.piece_name_template.format(rank=self.rank),
            ranges,
            points,
            offsets,
            **kwargs
        )

        # Gather local data extents
        extents = self.comm.gather(self.extent, root=MASTER_RANK)
        spacings = [(x[1] - x[0]) / (n - 1) for x, n in zip(ranges, points)]
        ranges = self.comm.gather(ranges, root=MASTER_RANK)

        # Ensure on master
        if not extents:
            return

        for rank, extent in enumerate(extents):
            self.pwriter.registerPiece({
                'Source': basename(self.piece_name_template.format(rank=rank)),
                'Extent': extent
            })

        extents = _min_max_reduce(_str_convert(extents))

        # Reducing domain ranges
        mins, maxs = [], []
        for r in ranges:
            mins.append(list(map(min, r)))
            maxs.append(list(map(max, r)))
        mins = list(zip(*mins))
        maxs = list(zip(*maxs))
        ranges = [(min(rmin), max(rmax)) for rmin, rmax in zip(mins, maxs)]

        spacings = functools.reduce(lambda x, y: x + f"{y} ", spacings, "")
        origins = functools.reduce(lambda x, y: x + f"{y[0]} ", ranges, "")

        self.pwriter.setDataNodeAttributes({
            'WholeExtent': extents,
            'Spacing': spacings,
            'Origin': origins
        })


class PRectilinearGrid(PVTKFile, vtk_files.RectilinearGrid):
    """Rectilinear grid parallel writer."""

    parent = vtk_files.RectilinearGrid

    def __init__(self, filename, coordinates, offsets, comm=None, **kwargs):
        """
        Init the parallel counterpart of a RectilinearGrid file.

        :param filename: name of file or file handle
        :param coordinates: list of coordinate arrays for this rank
        :param offset: offset in global dataset for this rank
        :param comm: MPI communicator (default: MPI.COMM_WORLD)
        """
        PVTKFile.__init__(self, filename, 'PRectilinearGrid',
                          comm=comm, **kwargs)

        # Name of piece file associated to rank
        self.parent.__init__(
            self,
            self.piece_name_template.format(rank=self.rank),
            coordinates,
            offsets,
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
        extent = _min_max_reduce(_str_convert(extents))

        self.pwriter.setDataNodeAttributes({
            'WholeExtent': extent
        })

        for coord, prefix in zip(self.coordinates, ('x', 'y', 'z')):
            array = DataArray(coord, [0], prefix + '_coordinates')
            pcoordinates.registerPDataArray(array)
