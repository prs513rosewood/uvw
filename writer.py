import xml.dom
import xml.dom.minidom as dom

import sys

def setAttributes(node, attributes):
    """Set attributes of a node"""
    map(lambda x: node.setAttribute(*x), attributes.items())

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

    def registerDataArray(self, data_array, append=True):
        array_component = Component('DataArray', self.node, self.writer)
        attributes = data_array.attributes

        if append:
            attributes['format'] = 'append'
            attributes['offset'] = str(self.writer.offset)
            self.writer.offset += data_array.flat_data.size
        else:
            attributes['format'] = 'binary'

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

    def setDataNodeAttributes(self, attributes):
        """Set attributes for the entire dataset"""
        setAttributes(self.data_node, attributes)

    def registerPiece(self, attributes):
        """Register a piece element"""
        piece = Component('Piece', self.data_node, self)
        setAttributes(piece.node, attributes)
        return piece

    def __str__(self):
        """Print XML to string"""
        return self.document.toprettyxml()

