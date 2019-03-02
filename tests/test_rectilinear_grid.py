from uvw import RectilinearGrid, DataArray
import numpy as np


def test_rectilinear_grid():
    N = 3

    x = np.linspace(0, 1, N*2)
    y = np.linspace(0, 1, N+2)
    z = np.linspace(0, 1, N)

    out_name = 'test_rectilinear_grid.vtr'

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    e_r = np.zeros(r.shape + (3, 3))
    e_r[0, 0, 0, :, :] = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]])
    e_r[-1, 0, 0, :, :] = np.eye(3)

    with RectilinearGrid(out_name, (x, y, z)) as rect:
        rect.addPointData(DataArray(r, range(3), 'R'))
        rect.addPointData(DataArray(r, range(3), 'R2'))
        rect.addPointData(DataArray(e_r, range(3), 'e_r'))

    output = open(out_name, 'r')
    reference = open('test_rectilinear_grid.ref', 'r')

    assert output.read() == reference.read()
