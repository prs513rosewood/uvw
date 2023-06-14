"""
Module with classes for interacting with the XML model underlying VTK files.

See https://kitware.github.io/vtk-examples/site/VTKFileFormats/ for format
description.
"""

__copyright__ = "Copyright © 2018-2023 Lucas Frérot"
__license__ = "SPDX-License-Identifier: MIT"

from .data_array import DataArray

import xml.dom.minidom as dom
import io
import zlib
import typing as ts

import numpy as np

from base64 import b64encode
from os import PathLike
from collections.abc import Mapping


def setAttributes(node: dom.Node, attributes: ts.Mapping[str, ts.Any]):
    """Set attributes of a node."""
    for item in attributes.items():
        node.setAttribute(*item)


def encodeArray(array: np.ndarray, level: int) -> str:
    """Encode array data and header in base64."""
    def compress(array):
        """Compress array with zlib. Returns header and compressed data."""
        raw_data = memoryview(array.tobytes())
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
        header_dtype = np.dtype(array.dtype.byteorder + 'u4')
        usize = max_block_size
        psize = last_block_size
        csize = [len(x) for x in compressed_data]
        header = np.array([nblocks, usize, psize] + csize, dtype=header_dtype)
        return header.tobytes(), b"".join(compressed_data)

    def raw(array):
        """Return header and array data in bytes."""
        header_dtype = np.dtype(array.dtype.byteorder + 'u4')
        header = np.array([array.nbytes], dtype=header_dtype)
        return header.tobytes(), memoryview(array)

    data = raw(array) if level == 0 else compress(array)
    return "".join(b64encode(x).decode() for x in data)


class Component:
    """Generic component class capable of registering sub-components."""

    def __init__(self, name: str, parent_node: dom.Node, writer):
        """Construct from name, parent node and writer object."""
        self.writer = writer
        self.document = writer.document
        self.node = self.document.createElement(name)
        parent_node.appendChild(self.node)

    def setAttributes(self, attributes: ts.Mapping[str, ts.Any]):
        """Set the node attributes from dictionary."""
        setAttributes(self.node, attributes)

    def register(
            self,
            name: str,
            attributes: ts.Optional[ts.Mapping[str, ts.Any]] = None
    ) -> 'Component':
        """Register a sub-component."""
        if attributes is None:
            attributes = {}

        if not isinstance(attributes, Mapping):
            raise ValueError(
                f'Cannot register attributes of type {type(attributes)}')

        sub_component = Component(name, self.node, self.writer)
        setAttributes(sub_component.node, attributes)
        return sub_component

    def _addArrayNodeData(self,
                          data_array: DataArray,
                          component: 'Component',
                          vtk_format: str):
        if vtk_format == 'ascii':
            sstream = io.StringIO()
            np.savetxt(
                sstream,
                data_array.flat_data,
                newline=' ',
                fmt=data_array.format_str,
            )
            data_as_str = sstream.getvalue()
            # reduce(lambda x, y: x + str(y) + ' ', data_array.flat_data, "")
        elif vtk_format == 'binary':
            data_as_str = encodeArray(data_array.flat_data,
                                      self.writer.compression)
        elif vtk_format == 'append':
            self.writer.append_data_arrays[component] = data_array
            return
        else:
            raise ValueError(f'Unsupported VTK Format "{vtk_format}"')

        component.node.appendChild(self.document.createTextNode(data_as_str))

    def _registerArrayComponent(self,
                                array: DataArray,
                                name: str,
                                vtk_format: str):
        attributes = array.attributes
        attributes['format'] = vtk_format
        return self.register(name, attributes)

    def registerDataArray(self,
                          data_array: DataArray,
                          vtk_format: str = 'binary'):
        """Register a DataArray object."""
        component = self._registerArrayComponent(data_array,
                                                 'DataArray',
                                                 vtk_format)
        self._addArrayNodeData(data_array, component, vtk_format)

    def registerPDataArray(self,
                           data_array: DataArray,
                           vtk_format: str = 'binary'):
        """Register a DataArray object in p-file."""
        self._registerArrayComponent(data_array, 'PDataArray', vtk_format)


class Writer:
    """Generic XML handler for VTK files."""

    FileDescriptor = ts.Union[str, PathLike, io.TextIOBase, io.BufferedIOBase]

    def __init__(self, vtk_format: str,
                 compression: ts.Optional[ts.Union[bool, int]] = None,
                 vtk_version: str = '0.1',
                 byte_order: str = 'LittleEndian'):
        """
        Create an XML writer.

        :param vtk_format: format of VTK file
        :param compression: compression level (see zlib), True, False or None
        :param vtk_version: version number of VTK file
        :param byte_order: byte order of binary data
        """
        valid_orders = {"LittleEndian", "BigEndian"}
        if byte_order not in valid_orders:
            raise ValueError(
                f"Byte order '{byte_order}' invalid, "
                f"should be in {valid_orders}")

        self.document = dom.getDOMImplementation()  \
                           .createDocument(None, 'VTKFile', None)
        self.root = self.document.documentElement
        self.root.setAttribute('type', vtk_format)
        self.root.setAttribute('version', vtk_version)
        self.root.setAttribute('byte_order', byte_order)
        self.data_node = self.document.createElement(vtk_format)
        self.root.appendChild(self.data_node)
        self.size_indicator_bytes = np.dtype('u4').itemsize
        self.append_data_arrays = {}

        if compression is None or compression is False or compression == 0:
            self.compression = 0
        elif compression is True:
            self.compression = -1
        elif compression in list(range(-1, 10)):
            self.compression = compression
        else:
            raise ValueError(f'compression level {compression} is not '
                             'recognized by zlib')

        if self.compression != 0:
            self.root.setAttribute('compressor', 'vtkZLibDataCompressor')

    def setDataNodeAttributes(self, attributes: ts.Mapping[str, ts.Any]):
        """Set attributes for the entire dataset."""
        setAttributes(self.data_node, attributes)

    def registerPiece(self, attributes: ts.Mapping[str, ts.Any] = {}):
        """Register a piece element."""
        return self.registerComponent('Piece', self.data_node,
                                      attributes)

    def registerComponent(
            self,
            name: str,
            parent: dom.Node,
            attributes: ts.Mapping[str, ts.Any] = {}
    ) -> Component:
        """Register a Component to a parent Component with set attributes."""
        comp = Component(name, parent, self)
        setAttributes(comp.node, attributes)
        return comp

    def registerAppend(self):
        """Register AppendedData node (should only be called once)."""
        append_node = Component('AppendedData', self.root, self)
        append_node.setAttributes({'format': 'base64'})
        self.root.appendChild(append_node.node)

        data_str = "_"
        offset = 0
        for component, data in self.append_data_arrays.items():
            component.setAttributes({'offset': str(offset)})
            data_b64 = encodeArray(data.flat_data, self.compression)
            data_str += data_b64
            offset += len(data_b64)

        text = self.document.createTextNode(data_str)
        append_node.node.appendChild(text)

    def write(self, fd: FileDescriptor):
        """Write to file descriptor."""
        if isinstance(fd, (str, PathLike)):
            with open(fd, 'wb') as fh:
                self.write(fh)
        elif isinstance(fd, io.TextIOBase):
            self.document.writexml(fd, indent="\n  ", addindent="  ")
        elif isinstance(fd, io.BufferedIOBase):
            fd.write(self.document.toxml(encoding='UTF-8'))
        else:
            raise TypeError(f"Expected a path or file descriptor, got {fd}")

    def __str__(self) -> str:
        """Print XML to string."""
        return self.document.toprettyxml()
