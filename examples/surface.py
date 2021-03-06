import sys
import numpy as np
from uvw import RectilinearGrid, DataArray

# Creating coordinates
x = np.linspace(-0.5, 0.5, 10)
y = np.linspace(-0.5, 0.5, 20)

# A centered disk
xx, yy = np.meshgrid(x, y, indexing='ij')
r = np.sqrt(xx**2 + yy**2)
R = 0.3
disk = r < R

data = np.zeros([10, 20])
data[disk] = np.sqrt(1-(r[disk]/R)**2)

cell_data = np.zeros([9, 19])
cell_data[0::2, 0::2] = 1
cell_data[1::2, 1::2] = 1

# File object can be used as a context manager
# and you can write to stdout!
with RectilinearGrid(sys.stdout, (x, y)) as grid:
  grid.addPointData(DataArray(data, range(2), 'data'))
  grid.addCellData(DataArray(cell_data, range(2), 'chess_board'))
