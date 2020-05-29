import numpy as np

from uvw import UnstructuredGrid
from uvw.unstructured import CellType

nodes = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [2, 0, 0],
    [0, 2, 0],
    [1, 2, 0],
])

connectivity = {
    CellType.QUAD: np.array([
        [0, 1, 2, 3], [2, 6, 5, 3],
    ]),
    5: np.array([[4, 2, 1]]),
}

f = UnstructuredGrid('ugrid.vtu', nodes, connectivity)
f.write()
