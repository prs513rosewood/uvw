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

array = np.array([[0, 0, 0], [1, 1, 1]])
print(type(array.dtype))
data_array = DataArray(array, [ 1])
print(data_array)

print(writer)
