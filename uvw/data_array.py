"""
Module defining class (`DataArray`) used to represent Numpy arrays in XML
model.
"""
import functools


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

        data_type = str(self.flat_data.dtype).capitalize()

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
