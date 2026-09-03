import numpy as np

from uvw import UnstructuredGrid
from uvw.unstructured import CellType

# test connectivity as a list instead of a numpy.array

nodes = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [2, 0, 0],
        [0, 2, 0],
        [1, 2, 0],
    ]
)

connectivity = {
    CellType.QUAD: [
        np.array([0, 1, 2, 3]),
        np.array([2, 6, 5, 3]),
    ],
}

f = UnstructuredGrid("ugrid.vtu", nodes, connectivity)
f.write()


# test : a cube defined as a polyhedron

nodes = np.array(
    [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [0, 1, 0],  # 2
        [0, 0, 1],  # 3
        [1, 1, 0],  # 4
        [0, 1, 1],  # 5
        [1, 0, 1],  # 6
        [1, 1, 1],  # 7
    ]
)

# faces is a list of list of np.arrays or list of lists
# data is likely to be supplied this way
faces = [
    [
        [0, 1, 4, 2],  # face z=0
        [3, 5, 7, 6],  # face z=1
        [0, 2, 5, 3],  # face x=0
        [1, 6, 7, 4],  # face x=1    ],
        [0, 3, 6, 1],  # face y=0
        [2, 4, 7, 5],  # face y=1
    ]
]

f = UnstructuredGrid("ugrid.vtu", nodes, connectivity={}, faces=faces)
f.write()


# test with a regular polyhedron (all faces with the same number of points)

# Vertices of a dodecahedron
phi = (1 + np.sqrt(5)) / 2
nodes = np.array(
    [
        # (±1, ±1, ±1)
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
        # (0, ±1/φ, ±φ)
        (0, -1 / phi, -phi),
        (0, -1 / phi, phi),
        (0, 1 / phi, -phi),
        (0, 1 / phi, phi),
        # (±1/φ, ±φ, 0)
        (-1 / phi, -phi, 0),
        (-1 / phi, phi, 0),
        (1 / phi, -phi, 0),
        (1 / phi, phi, 0),
        # (±φ, 0, ±1/φ)
        (-phi, 0, -1 / phi),
        (phi, 0, -1 / phi),
        (-phi, 0, 1 / phi),
        (phi, 0, 1 / phi),
    ]
)

# Faces of the dodecahedron
faces = [
    [
        [0, 16, 2, 10, 8],
        [0, 12, 14, 4, 8],
        [0, 16, 18, 1, 12],
        [1, 9, 11, 3, 18],
        [1, 12, 14, 5, 9],
        [2, 10, 6, 15, 13],
        [2, 13, 3, 18, 16],
        [3, 11, 7, 15, 13],
        [4, 14, 5, 19, 17],
        [4, 17, 6, 10, 8],
        [6, 17, 19, 7, 15],
        [7, 19, 5, 9, 11],
    ]
]

f = UnstructuredGrid("ugrid.vtu", nodes, connectivity={}, faces=faces)
f.write()

# test : combine two types of cells including a polyhedron in the same file
# icosidodecahedron + cube (hexahedron) in that order (for nodes)

# Vertices of the icosidodecaedron
phi = (1 + np.sqrt(5)) / 2
nodes_icosi = np.array(
    [
        [0, 0, phi],
        [0, phi, 0],
        [phi, 0, 0],
        [0, 0, -phi],
        [0, -phi, 0],
        [-phi, 0, 0],
        [0.5, phi / 2, (1 + phi) / 2],
        [phi / 2, (1 + phi) / 2, 0.5],
        [(1 + phi) / 2, 0.5, phi / 2],
        [-0.5, phi / 2, (1 + phi) / 2],
        [phi / 2, (1 + phi) / 2, -0.5],
        [(1 + phi) / 2, -0.5, phi / 2],
        [0.5, -phi / 2, (1 + phi) / 2],
        [-phi / 2, (1 + phi) / 2, 0.5],
        [(1 + phi) / 2, 0.5, -phi / 2],
        [-0.5, -phi / 2, (1 + phi) / 2],
        [-phi / 2, (1 + phi) / 2, -0.5],
        [(1 + phi) / 2, -0.5, -phi / 2],
        [0.5, phi / 2, -(1 + phi) / 2],
        [phi / 2, -(1 + phi) / 2, 0.5],
        [-(1 + phi) / 2, 0.5, phi / 2],
        [-0.5, phi / 2, -(1 + phi) / 2],
        [phi / 2, -(1 + phi) / 2, -0.5],
        [-(1 + phi) / 2, -0.5, phi / 2],
        [0.5, -phi / 2, -(1 + phi) / 2],
        [-phi / 2, -(1 + phi) / 2, 0.5],
        [-(1 + phi) / 2, 0.5, -phi / 2],
        [-0.5, -phi / 2, -(1 + phi) / 2],
        [-phi / 2, -(1 + phi) / 2, -0.5],
        [-(1 + phi) / 2, -0.5, -phi / 2],
    ]
)

# Faces
triangles = [
    [0, 6, 9],
    [0, 12, 15],
    [1, 7, 10],
    [1, 13, 16],
    [2, 8, 11],
    [2, 14, 17],
    [3, 18, 21],
    [3, 24, 27],
    [4, 19, 22],
    [4, 25, 28],
    [5, 20, 23],
    [5, 26, 29],
    [6, 7, 8],
    [9, 13, 20],
    [10, 14, 18],
    [11, 12, 19],
    [15, 23, 25],
    [16, 21, 26],
    [17, 22, 24],
    [27, 28, 29],
]
pentagons = [
    [0, 6, 8, 11, 12],
    [0, 9, 20, 23, 15],
    [1, 13, 9, 6, 7],
    [1, 10, 18, 21, 16],
    [2, 11, 19, 22, 17],
    [2, 14, 10, 7, 8],
    [3, 27, 29, 26, 21],
    [3, 18, 14, 17, 24],
    [4, 28, 27, 24, 22],
    [4, 19, 12, 15, 25],
    [5, 23, 25, 28, 29],
    [5, 20, 13, 16, 26],
]
faces_icosi = [triangles + pentagons]

# Vertices
nodes_cube = np.array(
    [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [0, 1, 0],  # 2
        [0, 0, 1],  # 3
        [1, 1, 0],  # 4
        [0, 1, 1],  # 5
        [1, 0, 1],  # 6
        [1, 1, 1],  # 7
    ]
) + np.array([2, 0, 0])

all_nodes = np.vstack((nodes_icosi, nodes_cube))
offset_nodes_cube = len(nodes_icosi)

connectivity_cube = {
    CellType.HEXAHEDRON: [
        np.array([0, 1, 4, 2, 3, 6, 7, 5]) + offset_nodes_cube,
    ],
}

f = UnstructuredGrid("ugrid.vtu", all_nodes, connectivity_cube, faces=faces_icosi)
f.write()


