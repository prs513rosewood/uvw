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
    POLYHEDRON = 42 # Polyhedron cell (consisting of polygonal faces)
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
    CellType.POLYHEDRON: -1,
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
        if not (isinstance(conn, np.ndarray) or isinstance(conn,list)): # question : wouldn't a list of list be acceptable ?
            raise TypeError("Connectivity needs to be of type numpy.ndarray or a list of numpy.ndarray")

        # NB : here it is not clear to me how variable size cells are dealt with...
        
        # int_types = {
        #     dtype for dtype, label in DTYPE_TO_VTK.items() if 'Int' in label
        # }

        # if conn.dtype not in int_types | {np.dtype(object)}:
        #     raise TypeError("Connectivity dtype needs to be an integer type or"
        #                    "an object type for variable size cells")

        nnodes = NODES_PER_CELL[cell_type]

        if nnodes != -1 :
            if nnodes != np.array(conn).shape[1]: 
                return False
    return True


# utility function for polyhedra faces
def faces_list_to_polyhedra(cells_faces):
    """
        Returns connectivity and a list containing a flat representation of faces for each cell

        List representation of faces: cells_faces[cell_index][face_index]=[vertex_0, vertex_1, ...]
        
        Flat representation of faces for one cell : 
        flat_faces[cell_index] = [number of faces, number_of_vertex_first_face, vertex_0, vertex_1, ..., number_of_vertex_second_face, vertex_0, vertex_1,...]
        
        :param cells_faces: list(cells) of list(faces) of array of point_ids
        :return: tuple(connectivity, flat_faces)
            Where
            connectivity is a list of arrays with the point ids for each cell
            flat_faces is a list of arrays with the flat description of faces for each cell
    """

    def flatten_list_of_list(list_of_list):
        return [x for xs in list_of_list for x in xs]
    
    connectivity = []    
    faces = []

    for cell_faces in cells_faces:

        # vtk representation of a face : number_of_vertex, vertex_0, vertex_1, ...
        vtk_faces = [
            [
                len(face),  # Nombre de sommets 
                *(face) # Indices sommets
            ]
            for face in cell_faces
        ]
        # vtk representation of a cell : number_of_faces, face_0, face_1, ...
        vtk_cell = [len(cell_faces), *flatten_list_of_list(vtk_faces)]
        faces.append(np.asarray(vtk_cell))

        # unique vertices for the cell connectivity
        face_vertex = np.unique(np.asarray(flatten_list_of_list(cell_faces)))
        connectivity.append(face_vertex)

    return {CellType.POLYHEDRON:connectivity}, faces

            return False
    return True
