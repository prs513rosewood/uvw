import numpy as np
import functools
import operator

class DataArray:
    """Class holding information on ndarray"""
    def __init__(self, data, spatial_axes, components_order='C'):
        self.data = data
        axes = list(range(data.ndim))
        for ax in spatial_axes:
            axes.remove(ax)

        nb_components = functools.reduce(lambda x, y: x * data.shape[y], axes, 1)

        if components_order == 'C':
            axes.reverse()
        else:
            raise Exception('Unrecognized components order')

        axes += spatial_axes

        self.flat_data = np.ravel(np.transpose(self.data, axes=axes), order='F')
        self.attributes = {
            "type": str(self.flat_data.dtype).capitalize(),
            "NumberOfComponents": nb_components
        }

    def __str__(self):
        return self.attributes.__str__()
