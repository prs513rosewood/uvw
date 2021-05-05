"""
Module defining class (`DataArray`) used to represent Numpy arrays in XML
model.
"""

__copyright__ = "Copyright © 2018-2021 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

import functools
import typing as ts
import numpy as np

DTYPE_TO_VTK = {
    np.dtype('<f4'): 'Float32',
    np.dtype('<f8'): 'Float64',
    np.dtype('<i1'): 'Int8',
    np.dtype('<i2'): 'Int16',
    np.dtype('<i4'): 'Int32',
    np.dtype('<i8'): 'Int64',
    np.dtype('<u1'): 'UInt8',
    np.dtype('<u2'): 'UInt16',
    np.dtype('<u4'): 'UInt32',
    np.dtype('<u8'): 'UInt64',

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

    def __init__(self,
                 data: ts.Sequence,
                 spatial_axes: ts.Iterable[int],
                 name: str = '',
                 components_order: str = 'C'):
        """
        Data array constructor

        :param data: the numpy array containing the data (possibly a view)
        :param spatial_axes: a sequence of ints that indicate which axes of the
                             array correspond to space dimensions (in order)
        :param name: the name of the data
        :param components_order: the order of the non-spatial axes of the array
        """
        self.data = np.asanyarray(data)
        axes = list(range(self.data.ndim))
        spatial_axes = list(spatial_axes)

        if self.data.ndim < len(spatial_axes):
            raise ValueError(
                'Dimensions of data smaller than space dimensions')

        for ax in spatial_axes:
            axes.remove(ax)

        nb_components = functools.reduce(
            lambda x, y: x * self.data.shape[y], axes, 1)

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
                f'Array dtype {self.flat_data.dtype} is not supported by VTK')

        self.attributes = {
            "Name": name,
            "type": data_type,
            "NumberOfComponents": str(nb_components),
            "NumberOfTuples": str(self.flat_data.size),
        }

        self.format_str = '%d' if 'Int' in data_type else '%.18e'

    def __str__(self) -> str:  # pragma: no cover
        return self.attributes.__str__()
