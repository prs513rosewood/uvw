from uvw import RectilinearGrid, DataArray
import numpy as np

def test_rectilinear_grid():
    N = 10

    x = np.linspace(0, 1, N*2)
    y = np.linspace(0, 1, N+2)
    z = np.linspace(0, 1, N)

    rect = RectilinearGrid('test.vtr', (x, y, z))

    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)
    e_r = np.zeros(r.shape + (3,3))
    e_r[0, 0, 0, :, :] = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]])
    e_r[-1, 0, 0, :, :] = np.eye(3)
    print(e_r.shape)
    rect.addPointData(DataArray(r, range(3), 'R'))
    rect.addPointData(DataArray(r, range(3), 'R2'))
    rect.addPointData(DataArray(e_r, range(3), 'e_r'))
    rect.write()
