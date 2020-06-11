"Rectilinear grid example with reordering of components"

import numpy as np
from uvw import RectilinearGrid, DataArray

# Creating coordinates
x = np.linspace(-0.5, 0.5, 10)
y = np.linspace(-0.5, 0.5, 20)
z = np.linspace(-0.9, 0.9, 30)

# Creating the file
grid = RectilinearGrid('grid.vtr', (x, y, z), compression=True)

# A centered ball
z, x, y = np.meshgrid(z, y, x, indexing='ij')
r = np.sqrt(x**2 + y**2 + z**2)
ball = r < 0.3

# Some multi-component multi-dimensional data (components order z y x)
data = np.zeros([30, 20, 10, 3, 3])
data[ball, ...] = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 1, 1]])


# Adding the point data (see help(DataArray) for more info)
grid.addPointData(DataArray(data, [2, 1, 0], 'data'))
grid.write()
