import xml.dom.minidom as dom
import io
import zlib

import numpy as np

from functools import reduce
from base64 import b64encode


def setAttributes(node, attributes):
    """Set attributes of a node"""
    for item in attributes.items():
        node.setAttribute(*item)


def encodeArray(array, level):
    """Encode array data and header in base64."""
    def compress(array):
        """Compress array with zlib. Returns header and compressed data."""
        raw_data = memoryview(array)  # avoid a copy
        data_size = raw_data.nbytes

        max_block_size = 2**15

        # Enough blocks to span whole data
        nblocks = data_size // max_block_size + 1
        last_block_size = data_size % max_block_size

        # Compress regular blocks
        compressed_data = [
            zlib.compress(raw_data[i*max_block_size:(i+1)*max_block_size],
                          level)
            for i in range(nblocks-1)
        ]

        # Compress last (smaller) block
        compressed_data.append(
            zlib.compress(raw_data[-last_block_size:], level)
        )

        # Header data (cf https://vtk.org/Wiki/VTK_XML_Formats#Compressed_Data)
        usize = max_block_size
        psize = last_block_size
        csize = [len(x) for x in compressed_data]
        header = np.array([nblocks, usize, psize] + csize, dtype=np.uint32)
        return header.tobytes(), b"".join(compressed_data)

    def raw(array):
        """Returns header and array data in bytes."""
        header = np.array([array.nbytes], dtype=np.uint32)
        return header.tobytes(), memoryview(array)

    if level is not None:
        data = compress(array)
    else:
        data = raw(array)
    return "".join(map(lambda x: b64encode(x).decode(), data))


class Component:
    """Generic component class capable of registering sub-components"""

    def __init__(self, name, parent_node, writer):
        self.writer = writer
        self.document = writer.document
        self.node = self.document.createElement(name)
        parent_node.appendChild(self.node)

    def setAttributes(self, attributes):
        setAttributes(self.node, attributes)

    def register(self, name, attributes={}):
        """Register a sub-component"""
        if type(attributes) != dict:
            raise Exception(
                'Cannot register attributes of type ' + str(type(attributes)))
        sub_component = Component(name, self.node, self.writer)
        setAttributes(sub_component.node, attributes)
        return sub_component

    def _addArrayNodeData(self, data_array, node, vtk_format):
        if vtk_format == 'ascii':
            data_as_str = reduce(
                lambda x, y: x + str(y) + ' ', data_array.flat_data, "")
        elif vtk_format == 'binary':
            data_as_str = encodeArray(data_array.flat_data,
                                      self.writer.compression)
        else:
            raise Exception('Unsupported VTK Format "{}"'.format(vtk_format))

        node.appendChild(self.document.createTextNode(data_as_str))

    def _registerArrayComponent(self, array, name, vtk_format):
        attributes = array.attributes
        attributes['format'] = vtk_format
        return self.register(name, attributes)

    def registerDataArray(self, data_array, vtk_format='binary'):
        """Register a DataArray object"""
        component = self._registerArrayComponent(data_array,
                                                 'DataArray',
                                                 vtk_format)
        self._addArrayNodeData(data_array, component.node, vtk_format)

    def registerPDataArray(self, data_array, vtk_format='binary'):
        """Register a DataArray object in p-file"""
        self._registerArrayComponent(data_array, 'PDataArray', vtk_format)


class Writer:
    """Generic XML handler for VTK files"""

    def __init__(self, vtk_format,
                 compression=None,
                 vtk_version='0.1',
                 byte_order='LittleEndian'):
        self.document = dom.getDOMImplementation()  \
                           .createDocument(None, 'VTKFile', None)
        self.root = self.document.documentElement
        self.root.setAttribute('type', vtk_format)
        self.root.setAttribute('version', vtk_version)
        self.root.setAttribute('byte_order', byte_order)
        self.data_node = self.document.createElement(vtk_format)
        self.root.appendChild(self.data_node)
        self.size_indicator_bytes = np.dtype(np.uint32).itemsize
        self.append_data_arrays = []

        if compression is not None and compression != False:
            self.root.setAttribute('compressor', 'vtkZLibDataCompressor')

            if type(compression) is not int:
                compression = -1
            else:
                if compression not in list(range(-1, 10)):
                    raise Exception(('compression level {} is not '
                                    'recognized by zlib').format(compression))
        elif not compression:
            compression = None

        self.compression = compression

    def setDataNodeAttributes(self, attributes):
        """Set attributes for the entire dataset"""
        setAttributes(self.data_node, attributes)

    def registerPiece(self, attributes={}):
        """Register a piece element"""
        return self.registerComponent('Piece', self.data_node,
                                      attributes)

    def registerComponent(self, name, parent, attributes={}):
        comp = Component(name, parent, self)
        setAttributes(comp.node, attributes)
        return comp

    def registerAppend(self):
        append_node = Component('AppendedData', self.root, self)
        setAttributes(append_node.node, {'format': 'base64'})
        self.root.appendChild(append_node.node)
        data_str = b"_"

        for data_array in self.append_data_arrays:
            data_str += encodeArray(data_array.flat_data)

        text = self.document.createTextNode(data_str.decode('ascii'))
        append_node.node.appendChild(text)

    def write(self, fd):
        if type(fd) == str:
            with open(fd, 'w') as file:
                self.write(file)
        elif issubclass(type(fd), io.TextIOBase):
            self.document.writexml(fd, indent="\n  ", addindent="  ")
        else:
            raise RuntimeError("Expected a path or "
                               + "file handle, got {}".format(type(fd)))

    def __str__(self):
        """Print XML to string"""
        return self.document.toprettyxml()
