"""Module with class for cell types in vtkUnstructuredGrid."""

__copyright__ = "Copyright © 2018-2024 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

from enum import Enum, unique

import numpy as np

from .data_array import DTYPE_TO_VTK


@unique
class CellType(Enum):
    """
    Enumerates the VTK cell types.

    See https://kitware.github.io/vtk-examples/site/VTKFileFormats/#legacy-file-examples
    """

    VERTEX = 1
    POLY_VERTEX = 2
    LINE = 3
    POLY_LINE = 4
    TRIANGLE = 5
    TRIANGLE_STRIP = 6
    POLYGON = 7
    PIXEL = 8
    QUAD = 9
    TETRA = 10
    VOXEL = 11
    HEXAHEDRON = 12
    WEDGE = 13
    PYRAMID = 14
    PENTAGONAL_PRISM = 15
    HEXAGONAL_PRISM = 16
    QUADRATIC_EDGE = 21
    QUADRATIC_TRIANGLE = 22
    QUADRATIC_QUAD = 23
    QUADRATIC_TETRA = 24
    QUADRATIC_HEXAHEDRON = 25
    QUADRATIC_WEDGE = 26
    QUADRATIC_PYRAMID = 27
    BIQUADRATIC_QUAD = 28
    TRIQUADRATIC_HEXAHEDRON = 29
    QUADRATIC_LINEAR_QUAD = 30
    QUADRATIC_LINEAR_WEDGE = 31
    BIQUADRATIC_QUADRATIC_WEDGE = 32
    BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33
    BIQUADRATIC_TRIANGLE = 34
    CUBIC_LINE = 35
    QUADRATIC_POLYGON = 36
    LAGRANGE_CURVE = 68
    LAGRANGE_TRIANGLE = 69
    LAGRANGE_QUADRILATERAL = 70
    LAGRANGE_TETRAHEDRON = 71
    LAGRANGE_HEXAHEDRON = 72
    LAGRANGE_WEDGE = 73
    LAGRANGE_PYRAMID = 74
    BEZIER_CURVE = 75
    BEZIER_TRIANGLE = 76
    BEZIER_QUADRILATERAL = 77
    BEZIER_TETRAHEDRON = 78
    BEZIER_HEXAHEDRON = 79
    BEZIER_WEDGE = 80
    BEZIER_PYRAMID = 81


NODES_PER_CELL = {
    CellType.VERTEX: 1,
    CellType.POLY_VERTEX: -1,
    CellType.LINE: 2,
    CellType.POLY_LINE: -1,
    CellType.TRIANGLE: 3,
    CellType.TRIANGLE_STRIP: -1,
    CellType.POLYGON: -1,
    CellType.PIXEL: 4,
    CellType.QUAD: 4,
    CellType.TETRA: 4,
    CellType.VOXEL: 8,
    CellType.HEXAHEDRON: 8,
    CellType.WEDGE: 6,
    CellType.PYRAMID: 5,
    CellType.PENTAGONAL_PRISM: 10,
    CellType.HEXAGONAL_PRISM: 12,
    CellType.QUADRATIC_EDGE: 3,
    CellType.QUADRATIC_TRIANGLE: 6,
    CellType.QUADRATIC_QUAD: 8,
    CellType.QUADRATIC_TETRA: 10,
    CellType.QUADRATIC_HEXAHEDRON: 20,
    CellType.QUADRATIC_WEDGE: 15,
    CellType.QUADRATIC_PYRAMID: 13,
    CellType.BIQUADRATIC_QUAD: 9,
    CellType.TRIQUADRATIC_HEXAHEDRON: 27,
    CellType.QUADRATIC_LINEAR_QUAD: 6,
    CellType.QUADRATIC_LINEAR_WEDGE: 12,
    CellType.BIQUADRATIC_QUADRATIC_WEDGE: 18,
    CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON: 24,
    CellType.BIQUADRATIC_TRIANGLE: 7,
    CellType.CUBIC_LINE: 4,
    CellType.QUADRATIC_POLYGON: -1,
    CellType.LAGRANGE_CURVE: -1,
    CellType.LAGRANGE_TRIANGLE: -1,
    CellType.LAGRANGE_QUADRILATERAL: -1,
    CellType.LAGRANGE_TETRAHEDRON: -1,
    CellType.LAGRANGE_HEXAHEDRON: -1,
    CellType.LAGRANGE_WEDGE: -1,
    CellType.LAGRANGE_PYRAMID: -1,
    CellType.BEZIER_CURVE: -1,
    CellType.BEZIER_TRIANGLE: -1,
    CellType.BEZIER_QUADRILATERAL: -1,
    CellType.BEZIER_TETRAHEDRON: -1,
    CellType.BEZIER_HEXAHEDRON: -1,
    CellType.BEZIER_WEDGE: -1,
    CellType.BEZIER_PYRAMID: -1,
}


def check_connectivity(connectivity):
    """Sanity check for number of nodes per cell."""
    for cell_type, conn in connectivity.items():
        if not isinstance(cell_type, CellType):
            cell_type = CellType(cell_type)
        if not isinstance(conn, np.ndarray):
            raise TypeError("Connectivity needs to be of type numpy.ndarray")

        int_types = {
            dtype for dtype, label in DTYPE_TO_VTK.items() if 'Int' in label
        }

        if conn.dtype not in int_types | {np.dtype(object)}:
            raise TypeError("Connectivity dtype needs to be an integer type or"
                            "an object type for variable size cells")
        nnodes = NODES_PER_CELL[cell_type]

        if nnodes != -1 and nnodes != conn.shape[1]:
            return False
    return True
