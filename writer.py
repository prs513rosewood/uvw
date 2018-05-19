import xml.dom
import xml.dom.minidom as dom
import functools

import base64

def setAttributes(node, attributes):
    """Set attributes of a node"""
    for item in attributes.items():
        node.setAttribute(*item)

class Component:
    """Generic component class capable of registering sub-components"""
    def __init__(self, name, parent_node, writer):
        self.writer = writer
        self.document = writer.document
        self.node = self.document.createElement(name)
        parent_node.appendChild(self.node)


    def register(self, name, attributes=dict()):
        """Register a sub-component"""
        sub_component = Component(name, self.node, self.writer)
        setAttributes(sub_component.node, attributes)
        return sub_component

    def registerDataArray(self, data_array, vtk_format='append'):
        array_component = Component('DataArray', self.node, self.writer)
        attributes = data_array.attributes

        attributes['format'] = vtk_format
        if vtk_format == 'append':
            attributes['offset'] = str(self.writer.offset)
            self.writer.offset += data_array.flat_data.size
            self.writer.append_data_arrays.append(data_array)

        elif vtk_format == 'ascii':
            data_as_str = functools.reduce(lambda x, y: x + str(y) + ' ', data_array.flat_data, "")
            array_component.node.appendChild(self.document.createTextNode(data_as_str))
            

        setAttributes(array_component.node, attributes)


class Writer:
    """Generic XML handler for VTK files"""
    def __init__(self, vtk_format, vtk_version='0.1', byte_order='LittleEndian'):
        self.document = dom.getDOMImplementation().createDocument(None, 'VTKFile', None)
        self.root = self.document.documentElement
        self.root.setAttribute('type', vtk_format)
        self.root.setAttribute('version', vtk_version)
        self.root.setAttribute('byte_order', byte_order)
        self.data_node = self.document.createElement(vtk_format)
        self.root.appendChild(self.data_node)
        self.offset = 0 # Global offset
        self.append_data_arrays = []

    def setDataNodeAttributes(self, attributes):
        """Set attributes for the entire dataset"""
        setAttributes(self.data_node, attributes)

    def registerPiece(self, attributes):
        """Register a piece element"""
        piece = Component('Piece', self.data_node, self)
        setAttributes(piece.node, attributes)
        return piece

    def registerAppend(self):
        append_node = Component('AppendedData', self.root, self)
        setAttributes(append_node.node, {'format': 'base64'})
        self.root.appendChild(append_node.node)
        data_str = b"_"

        for data_array in self.append_data_arrays:
            data_str += base64.b64encode(data_array.flat_data)

        text = self.document.createTextNode(data_str.decode('ascii'))
        append_node.node.appendChild(text)

    def write(self, filename):
        with open(filename, 'w') as file:
            file.write(str(self))

    def __str__(self):
        """Print XML to string"""
        return self.document.toprettyxml()

