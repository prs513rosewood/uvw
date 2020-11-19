import timeit
import time

from pyevtk.hl import gridToVTK

import numpy as np

x = np.linspace(0, 1, 10**3)
y = np.linspace(0, 1, 5*10**2)
z = np.linspace(0, 1, 10**2)

xx, yy, zz = np.meshgrid(x, y, z, indexing='xy', sparse=True)

r = np.sqrt(xx**2 + yy**2 + zz**2)


def write_pyevtk():
    gridToVTK('pyevtk', x, y, z, pointData={'': r})


tok = time.time()
write_pyevtk()
print("pyevtk:", time.time() - tok)
