from uvw import ImageData, DataArray
import numpy as np

def test_image_data():
    N = 4

    x = np.linspace(0, 1, N*2)
    y = np.linspace(1, 3, N+2)
    z = np.linspace(0, 2, N)

    out_name = 'test_image_data.vti'
    rect = ImageData(out_name,
                     [(0, 1), (1, 3), (0, 2)],
                     [N*2, N+2, N])

    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)

    rect.addPointData(DataArray(r, range(3), 'R'))
    rect.write()

    output = open(out_name, 'r')
    reference = open('test_image_data.ref', 'r')

    assert output.read() == reference.read()
