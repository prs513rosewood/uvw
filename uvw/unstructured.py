"""
Module with class for cell types in vtkUnstructuredGrid
"""

from enum import Enum, unique

import numpy as np


@unique
class CellType(Enum):
    """
    Enumerates the VTK cell types.

    See https://lorensen.github.io/VTKExamples/site/VTKFileFormats/
    """
    VERTEX = 1, 1
    POLY_VERTEX = 2, -1
    LINE = 3, 2
    POLY_LINE = 4, -1
    TRIANGLE = 5, 3
    TRIANGLE_STRIP = 6, -1
    POLYGON = 7, -1
    PIXEL = 8, 4
    QUAD = 9, 4
    TETRA = 10, 4
    VOXEL = 11, 8
    HEXAHEDRON = 12, 9
    WEDGE = 13, 6
    PYRAMID = 14, 5
    PENTAGONAL_PRISM = 15, 10
    HEXAGONAL_PRISM = 16, 12
    QUADRATIC_EDGE = 21, 3
    QUADRATIC_TRIANGLE = 22, 6
    QUADRATIC_QUAD = 23, 8
    QUADRATIC_TETRA = 24, 10
    QUADRATIC_HEXAHEDRON = 25, 20
    QUADRATIC_WEDGE = 26, 15
    QUADRATIC_PYRAMID = 27, 13
    BIQUADRATIC_QUAD = 28, 9
    TRIQUADRATIC_HEXAHEDRON = 29, 27
    QUADRATIC_LINEAR_QUAD = 30, 6
    QUADRATIC_LINEAR_WEDGE = 31, 12
    BIQUADRATIC_QUADRATIC_WEDGE = 32, 18
    BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33, 24
    BIQUADRATIC_TRIANGLE = 34, 7
    CUBIC_LINE = 35, 4
    QUADRATIC_POLYGON = 36, -1


def check_connectivity(connectivity):
    "Sanity check for number of nodes per cell"
    for cell_type, conn in connectivity.items():
        if not isinstance(cell_type, CellType):
            raise TypeError("Connectivity cell type is not of type CellType")
        if not isinstance(conn, np.ndarray):
            raise TypeError("Connectivity needs to be of type numpy.ndarray")
        if conn.dtype not in [np.int32]:
            raise TypeError("Connectivity dtype needs to be an integer type")
        nnodes = cell_type.value[1]

        if nnodes not in (-1, conn.shape[1]):
            return False
    return True
