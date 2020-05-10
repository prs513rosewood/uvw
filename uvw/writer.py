import xml.dom.minidom as dom
import functools
import io

import base64
import numpy as np


def setAttributes(node, attributes):
    """Set attributes of a node"""
    for item in attributes.items():
        node.setAttribute(*item)


def encodeArray(array):
    # Mandatory number of bytes encoded as uint32
    nbytes = array.nbytes
    bytes = base64.b64encode(np.array([nbytes], dtype=np.uint32))
    bytes += base64.b64encode(array)
    return bytes


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
            data_as_str = functools.reduce(
                lambda x, y: x + str(y) + ' ', data_array.flat_data, "")
            node.appendChild(self.document.createTextNode(data_as_str))

        elif vtk_format == 'binary':
            node.appendChild(self.document.createTextNode(
                    encodeArray(data_array.flat_data).decode('ascii')))
        else:
            raise Exception('Unsupported VTK Format "{}"'.format(vtk_format))

    def registerDataArray(self, data_array, vtk_format='binary'):
        """Register a DataArray object"""
        array_component = Component('DataArray', self.node, self.writer)
        attributes = data_array.attributes

        attributes['format'] = vtk_format

        # Write array data
        self._addArrayNodeData(data_array, array_component.node, vtk_format)

        setAttributes(array_component.node, attributes)

    def registerPDataArray(self, data_array, vtk_format='binary'):
        """Register a DataArray object in p-file"""
        array_component = Component('PDataArray', self.node, self.writer)
        attributes = data_array.attributes
        attributes['format'] = vtk_format
        setAttributes(array_component.node, attributes)


class Writer:
    """Generic XML handler for VTK files"""

    def __init__(self, vtk_format,
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
