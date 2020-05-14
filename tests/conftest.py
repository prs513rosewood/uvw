import pytest
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as v2n

@pytest.fixture
def threeD_data():
    N = 4

    x = np.linspace(0, 1, N*2)
    y = np.linspace(0, 1, N+2)
    z = np.linspace(0, 1, N)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    e_r = np.zeros([s - 1 for s in r.shape] + [3, 3])
    e_r[0, 0, 0, :, :] = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]])
    e_r[-1, 0, 0, :, :] = np.eye(3)

    return [x, y, z], r, e_r


@pytest.fixture(params=[False, True])
def compression_fixture(request):
    return request


def get_vtk_data(reader, sstream):
    reader.SetReadFromInputString(True)
    reader.SetInputString(sstream.getvalue())
    reader.Update()
    output = reader.GetOutput()
    return v2n(output.GetPointData().GetArray('point')), \
        v2n(output.GetCellData().GetArray('cell'))
