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
