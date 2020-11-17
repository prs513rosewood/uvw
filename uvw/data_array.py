"""
Module defining class (`DataArray`) used to represent Numpy arrays in XML
model.
"""
import functools
import numpy as np

DTYPE_TO_VTK = {
    np.dtype(np.float32): 'Float32',
    np.dtype(np.float64): 'Float64',
    np.dtype(np.int8): 'Int8',
    np.dtype(np.int16): 'Int16',
    np.dtype(np.int32): 'Int32',
    np.dtype(np.int64): 'Int64',
    np.dtype(np.uint8): 'UInt8',
    np.dtype(np.uint16): 'UInt16',
    np.dtype(np.uint32): 'UInt32',
    np.dtype(np.uint64): 'UInt64',

    # Big-Endian variants (>i1 == <i1 and >u1 == <u1)
    np.dtype('>f4'): 'Float32',
    np.dtype('>f8'): 'Float64',
    np.dtype('>i2'): 'Int16',
    np.dtype('>i4'): 'Int32',
    np.dtype('>i8'): 'Int64',
    np.dtype('>u2'): 'UInt16',
    np.dtype('>u4'): 'UInt32',
    np.dtype('>u8'): 'UInt64',
}


class DataArray:
    """Class holding information on ndarray"""

    def __init__(self, data, spatial_axes, name='', components_order='C'):
        """
        Data array constructor

        :param data: the numpy array containing the data (possibly a view)
        :param spatial_axes: a container of ints that indicate which axes of
        the array correspond to space dimensions (in order)
        :param name: the name of the data
        :param components_order: the order of the non-spatial axes of the array
        """
        self.data = data
        axes = list(range(data.ndim))
        spatial_axes = list(spatial_axes)

        if data.ndim < len(spatial_axes):
            raise ValueError(
                'Dimensions of data smaller than space dimensions')

        for ax in spatial_axes:
            axes.remove(ax)

        nb_components = functools.reduce(
            lambda x, y: x * data.shape[y], axes, 1)

        if components_order == 'C':
            axes.reverse()
        elif components_order == 'F':
            pass
        else:
            raise ValueError('Unrecognized components order')

        axes += spatial_axes

        # Hopefully this is a view
        self.flat_data = self.data.transpose(*axes).reshape(-1, order='F')

        try:
            data_type = DTYPE_TO_VTK[self.flat_data.dtype]
        except KeyError:
            raise TypeError(
                'Array dtype {} is not supported by VTK'.format(
                    self.flat_data.dtype))

        self.attributes = {
            "Name": name,
            "type": data_type,
            "NumberOfComponents": str(nb_components),
            "NumberOfTuples": str(self.flat_data.size),
        }

        if 'Int' in data_type:
            self.format_str = '%d'
        else:
            self.format_str = '%.18e'

    def __str__(self):  # pragma: no cover
        return self.attributes.__str__()
