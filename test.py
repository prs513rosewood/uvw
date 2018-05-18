from writer import *
from data_array import *
import numpy as np

writer = Writer('RectilinearGrid')
writer.setDataNodeAttributes({
    "WholeExtent": "0 1 0 1 0 1"
})

piece = writer.registerPiece({
    "Extent": "0 1 0 1 0 1"
})

coordinates = piece.register('Coordinates')

array = np.array([[0, 0, 0], [1, 1, 1]], np.float64)
data_array = DataArray(array, [1])

coordinates.registerDataArray(data_array)
coordinates.registerDataArray(data_array)

print(writer)
