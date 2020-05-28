import numpy as np

from uvw import UnstructuredGrid
from uvw.unstructured import CellType

nodes = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
])

connectivity = {
    CellType.TRIANGLE_STRIP: np.array([
        [0, 3, 1, 2],
    ], dtype=np.int32)
}

f = UnstructuredGrid('ugrid.vtu', nodes, connectivity)
f.write()
