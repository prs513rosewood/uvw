import pytest
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as v2n

from uvw.data_array import DTYPE_TO_VTK

@pytest.fixture(params=DTYPE_TO_VTK.keys())
def dtype_fixture(request):
    return request

@pytest.fixture(params=[1, 2, 3])
def field_data(request, dtype_fixture):
    N = 4

    dim = request.param

    if dim == 1:
        x = np.linspace(0, 1, N*2)
        coords = [x]
        r = np.abs(x)

    if dim == 2:
        x = np.linspace(0, 1, N*2)
        y = np.linspace(0, 1, N+2)
        coords = [x, y]

        xx, yy = np.meshgrid(x, y, indexing='ij', sparse=True)
        r = np.sqrt(xx**2 + yy**2)

    if dim == 3:
        x = np.linspace(0, 1, N*2)
        y = np.linspace(0, 1, N+2)
        z = np.linspace(0, 1, N)
        coords = [x, y, z]

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        r = np.sqrt(xx**2 + yy**2 + zz**2)

    e_r = np.zeros([s - 1 for s in r.shape] + [3, 3])
    e_r[..., :, :] = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]])
    f = np.arange(dim, dtype=dtype_fixture.param)  # field array

    return coords, r, e_r, f


@pytest.fixture(params=[False, True])
def compression_fixture(request):
    return request


@pytest.fixture(params=['ascii', 'binary', 'append'])
def format_fixture(request):
    return request


def get_vtk_data(reader, sstream):
    if isinstance(sstream, str):
        reader.SetFileName(sstream)
    else:
        reader.SetReadFromInputString(True)
        reader.SetInputString(sstream.getvalue())
    reader.Update()
    output = reader.GetOutput()
    return v2n(output.GetPointData().GetArray('point')), \
        v2n(output.GetCellData().GetArray('cell')), \
        v2n(output.GetFieldData().GetArray('field'))


@pytest.fixture(params=['C', 'F'])
def ordering_fixture(request):
    def transp(dim):
        trans = list(range(dim+2))
        if request.param == 'C':
            trans[-2], trans[-1] = trans[-1], trans[-2]
        return trans
    request.transp = transp
    return request
