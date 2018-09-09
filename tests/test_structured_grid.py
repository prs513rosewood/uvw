from uvw import StructuredGrid, DataArray
import numpy as np

def test_structured_grid():
    N = 5

    r = np.linspace(0, 1, N)
    theta = np.linspace(0, 2*np.pi, 5*N)

    theta, r = np.meshgrid(theta, r, indexing='ij')

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    points = np.vstack([x.ravel(), y.ravel()]).T

    out_name = 'test_structured_grid.vts'
    grid = StructuredGrid(out_name, points, (N, 5*N))

    data = np.exp(-4*r**2)

    grid.addPointData(DataArray(data, reversed(range(2)), 'data'))
    grid.write()

    output = open(out_name, 'r')
    reference = open('test_structured_grid.ref', 'r')

    assert output.read() == reference.read()
