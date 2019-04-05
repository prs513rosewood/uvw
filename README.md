UVW - Universal VTK Writer
==========================

UVW is a small utility library to write VTK files from data contained in Numpy arrays. It handles fully-fledged `ndarrays` defined over {1, 2, 3}-d domains, with arbitrary number of components. There are no constraints on the particular order of components, although copy of data can be avoided if the array is Fortran contiguous, as VTK files are written in Fortran order. Future developments will include multi-process write support.

## Getting Started

Here is how to install and use `uvw`.

### Prerequisites

* Python 3. It may work with python 2, but it hasn't been tested.
* [Numpy](http://www.numpy.org/). This code has been tested with Numpy version 1.14.3.

### Installing

This library can be installed with `pip`:

```
pip install --user uvw
```

### Writing Numpy arrays

As a first example, let us write a multi-component numpy array into a rectilinear grid:

```python
import numpy as np
from uvw import RectilinearGrid, DataArray

# Creating coordinates
x = np.linspace(-0.5, 0.5, 10)
y = np.linspace(-0.5, 0.5, 20)
z = np.linspace(-0.9, 0.9, 30)

# Creating the file
grid = RectilinearGrid('grid.vtr', (x, y, z))

# A centered ball
x, y, z = np.meshgrid(x, y, z, indexing='ij')
r = np.sqrt(x**2 + y**2 + z**2)
ball = r < 0.3

# Some multi-component multi-dimensional data
data = np.zeros([10, 20, 30, 3, 3])
data[ball, ...] = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 1, 1]])


# Adding the point data (see help(DataArray) for more info)
grid.addPointData(DataArray(data, range(3), 'data'))
grid.write()
```

UVW also supports writing data on 2D and 1D physical domains, for example:

```python
import numpy as np
from uvw import RectilinearGrid, DataArray

# Creating coordinates
x = np.linspace(-0.5, 0.5, 10)
y = np.linspace(-0.5, 0.5, 20)

# A centered disk
xx, yy = np.meshgrid(x, y, indexing='ij')
r = np.sqrt(xx**2 + yy**2)
R = 0.3
disk = r < R

data = np.zeros([10, 20])
data[disk] = np.sqrt(1-(r[disk]/R)**2)

# File object can be used as a context manager
with RectilinearGrid('grid.vtr', (x, y)) as grid:
  grid.addPointData(DataArray(data, range(2), 'data'))
```

## List of features

Here is a list of what is available in UVW:

### VTK file formats

- Image data (`.vti`)
- Rectilinear grid (`.vtr`)
- Structured grid (`.vts`)

### Data representation

- ASCII
- Base64 (uncompressed)

### Planned developments

Here is a list of future developments:

- [x] Image data
- [ ] Unstructured grid
- [x] Structured grid
- [ ] Parallel writing (multi-process)
- [ ] Benchmarking + performance comparison with [pyevtk](https://bitbucket.org/pauloh/pyevtk)


## Developing

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Git repository

First clone the git repository:

```
git clone https://c4science.ch/source/uvw.git
```

Then you can use pip in development mode (possibly in [virtualenv](https://virtualenv.pypa.io/en/stable/)):

```
pip install --user -e .
```

## Running the tests

The tests can be run using [pytest](https://docs.pytest.org/en/latest/):

```
cd tests; pytest
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

* [@PurpleBooth](https://github.com/PurpleBooth)'s [README-Template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
