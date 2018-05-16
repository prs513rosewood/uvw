import xml.dom
import xml.dom.minidom as dom

import sys

def setAttributes(node, attributes):
    """Set attributes of a node"""
    map(lambda x: node.setAttribute(*x), attributes.items())

class Component:
    """Generic component class capable of registering sub-components"""
    def __init__(self, name, parent_node, document):
        self.node = document.createElement(name)
        self.document = document
        parent_node.appendChild(self.node)


    def register(self, name, attributes=dict()):
        """Register a sub-component"""
        sub_component = Component(name, self.node, self.document)
        setAttributes(sub_component.node, attributes)
        return sub_component


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

    def setDataNodeAttributes(self, attributes):
        """Set attributes for the entire dataset"""
        setAttributes(self.data_node, attributes)

    def registerPiece(self, attributes):
        """Register a piece element"""
        piece = Component('Piece', self.data_node, self.document)
        setAttributes(piece.node, attributes)
        return piece

    def __str__(self):
        """Print XML to string"""
        return self.document.toprettyxml()

