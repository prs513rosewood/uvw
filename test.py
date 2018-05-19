from writer import *
from data_array import *
from vtk_files import *
import numpy as np

N = 10

x = np.linspace(0, 1, 2*N)
y = np.linspace(0, 1, N)
z = np.linspace(0, 1, N)

rect = RectilinearGrid('test.vtr', (x, y, z))

x, y, z = np.meshgrid(x, y, z, indexing='ij')
r = np.sqrt(x**2 + y**2 + z**2)
e_r = np.zeros(r.shape + (3,3))
e_r[0, 0, 0, :, :] = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]])
e_r[-1, 0, 0, :, :] = np.eye(3)
print(e_r.shape)
rect.addPointData(DataArray(e_r, range(3), 'e_r'))
rect.addPointData(DataArray(r, range(3), 'R'))
rect.write()
