"""UVW: Universal VTK Writer.

UVW is a small utility library to write XML VTK files from data contained in
Numpy arrays. It handles fully-fledged `ndarrays` defined over {1, 2, 3}-d
domains, with arbitrary number of components. There are no constraints on the
particular order of components, although copy of data can be avoided if the
array is Fortran contiguous, as VTK files are written in Fortran order. UVW
supports multi-process writing of VTK files, so that it can be used in an MPI
environment.

For documentation and examples, check out the various classes' docstrings and
the repository home on GitHub: https://github.com/prs513rosewood/uvw
"""

__author__ = "Lucas Frérot"
__maintainer__ = __author__
__copyright__ = "Copyright © 2018-2023 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

# flake8: noqa

from .vtk_files import (
    RectilinearGrid,
    ImageData,
    StructuredGrid,
    UnstructuredGrid,
    ParaViewData
)

from .data_array import DataArray

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
