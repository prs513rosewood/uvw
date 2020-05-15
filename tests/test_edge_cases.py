import pytest
import numpy as np

from uvw import (
    DataArray,
    RectilinearGrid,
    StructuredGrid,
)


def test_malformed_attributes():
    x = np.array([0, 1])

    rect = RectilinearGrid('', x)

    with pytest.raises(ValueError):
        rect.piece.register('', [])


def test_unsupported_format():
    x = np.array([0, 1])

    rect = RectilinearGrid('', x)
    with pytest.raises(ValueError):
        rect.addPointData(DataArray(x, [0]), '#dummy')


def test_invalid_compression():
    x = np.array([0, 1])
    with pytest.raises(ValueError):
        RectilinearGrid('', x, compression=100)


def test_invalid_file():
    x = np.array([0, 1])
    grid = RectilinearGrid('', x)
    with pytest.raises(ValueError):
        grid.writer.write(1)


def test_array_dimensions():
    x = np.array([0, 1])
    with pytest.raises(ValueError):
        DataArray(x, range(2))

    with pytest.raises(ValueError):
        DataArray(x, range(1), components_order=2)


def test_rectilinear_fails():
    x = np.array([[0, 1], [2, 3]])
    with pytest.raises(ValueError):
        RectilinearGrid('', x)

    x = np.array([0, 1])
    with pytest.raises(ValueError):
        RectilinearGrid('', x, [])


def test_structured_fails():
    x = np.array([1, 2])
    with pytest.raises(ValueError):
        StructuredGrid('', x, (1, 2))
