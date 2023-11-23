"""Defining class (`DataArray`) used to represent Numpy arrays in XML model."""

__copyright__ = "Copyright © 2018-2023 Lucas Frérot"
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
    """Class holding information on array data.

    This class allows ndarrays with arbitrary shape, including the number of
    components, to be used by the sub-classes of VTKFile. This is done by
    supplying the set of axis of the array that represent data in physical
    directions (e.g. x, y, z). This means that arrays defined on a {1,2,3}-d
    grid with N number of components are natively handled without having to
    reshape the array prior to writing the file.

    The VTK XML format assumes arrays are stored in Fortran order. By default,
    ndarrays are stored in C order, so a copy of the data may be necessary when
    the object is created. DataArray objects can also be told that components
    may be stored in C or Fortan order (e.g. if the data stored per point is a
    matrix).

    """

    def __init__(self,
                 data: ts.Sequence,
                 spatial_axes: ts.Iterable[int],
                 name: str = '',
                 components_order: str = 'C'):
        """
        Construct a data array.

        :param data: the numpy array containing the data (possibly a view)
        :param spatial_axes: a sequence of ints that indicate which axes of the
                             array correspond to space dimensions (in order)
        :param name: the name of the data
        :param components_order: the order of the non-spatial axes of the array
        """
        self.data = np.asanyarray(data)
        self.axes = list(range(self.data.ndim))
        spatial_axes = list(spatial_axes)

        if self.data.ndim < len(spatial_axes):
            raise ValueError(
                'Dimensions of data smaller than space dimensions')

        for ax in spatial_axes:
            self.axes.remove(ax)

        nb_components = functools.reduce(
            lambda x, y: x * self.data.shape[y], self.axes, 1)

        if components_order == 'C':
            self.axes.reverse()
        elif components_order == 'F':
            pass
        else:
            raise ValueError('Unrecognized components order')

        self.axes += spatial_axes

        flat_data = self.flat_data

        try:
            data_type = DTYPE_TO_VTK[flat_data.dtype]
        except KeyError:
            raise TypeError(
                f'Array dtype {flat_data.dtype} is not supported by VTK')

        self.attributes = {
            "Name": name,
            "type": data_type,
            "NumberOfComponents": str(nb_components),
            "NumberOfTuples": str(flat_data.size),
        }

        self.format_str = '%d' if 'Int' in data_type else '%.18e'

    @property
    def flat_data(self):
        """Lazy evaluation of flat array"""
        flat_data = self.data.transpose(*self.axes).reshape(-1, order='F')

        # Consistency check
        attributes = getattr(self, "attributes", None)
        if attributes is not None:
            assert attributes["type"] == DTYPE_TO_VTK[flat_data.dtype]
            assert attributes["NumberOfTuples"] == str(flat_data.size)

        return flat_data

    def __str__(self) -> str:  # pragma: no cover
        """Produce string representation."""
        return self.attributes.__str__()
