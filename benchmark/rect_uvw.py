import timeit
import time

from uvw import RectilinearGrid, DataArray

import numpy as np

x = np.linspace(0, 1, 10**3)
y = np.linspace(0, 1, 5*10**2)
z = np.linspace(0, 1, 10**2)

xx, yy, zz = np.meshgrid(x, y, z, indexing='xy', sparse=True)

r = np.sqrt(xx**2 + yy**2 + zz**2)


def write_uvw():
    f = RectilinearGrid('uvw.vtr', (x, y, z))
    f.addPointData(DataArray(r, range(r.ndim), ''), vtk_format='append')
    f.write()


tik = time.time()
write_uvw()
print("uvw:", time.time() - tik)
