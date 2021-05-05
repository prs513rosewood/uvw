UVW - Universal VTK Writer
==========================
[![Build Status](https://travis-ci.com/prs513rosewood/uvw.svg?branch=master)](https://travis-ci.com/prs513rosewood/uvw)
[![Coverage
Status](https://coveralls.io/repos/github/prs513rosewood/uvw/badge.svg?branch=master)](https://coveralls.io/github/prs513rosewood/uvw?branch=master)
[![PyPI Version](https://img.shields.io/pypi/v/uvw.svg)](https://pypi.org/project/uvw/)

UVW is a small utility library to write [XML VTK
files](https://kitware.github.io/vtk-examples/site/VTKFileFormats/#xml-file-formats)
from data contained in Numpy arrays. It handles fully-fledged `ndarrays` defined
over {1, 2, 3}-d domains, with arbitrary number of components. There are no
constraints on the particular order of components, although copy of data can be
avoided if the array is Fortran contiguous, as VTK files are written in Fortran
order. UVW supports multi-process writing of VTK files, so that it can be used
in an MPI environment.

## Getting Started

Here is how to install and use `uvw`.

### Prerequisites

* Python 3. It may work with python 2, but it hasn't been tested.
* [Numpy](http://www.numpy.org/). This code has been tested with Numpy version
  1.14.3.
* (Optional) [mpi4py](https://mpi4py.readthedocs.io/en/stable/) only if you wish to use the
  parallel classes of UVW (i.e. the submodule `uvw.parallel`)

### Installing

This library can be installed with `pip`:

```
pip install --user uvw
```

If you want to activate parallel capabilities, run:

```
pip install --user uvw[mpi]
```

which will automatically pull `mpi4py` as a dependency.

### Writing Numpy arrays

As a first example, let us write a multi-component numpy array into a
rectilinear grid:

```python
import numpy as np
from uvw import RectilinearGrid, DataArray

# Creating coordinates
x = np.linspace(-0.5, 0.5, 10)
y = np.linspace(-0.5, 0.5, 20)
z = np.linspace(-0.9, 0.9, 30)

# Creating the file (with possible data compression)
grid = RectilinearGrid('grid.vtr', (x, y, z), compression=True)

# A centered ball
x, y, z = np.meshgrid(x, y, z, indexing='ij')
r = np.sqrt(x**2 + y**2 + z**2)
ball = r < 0.3

# Some multi-component multi-dimensional data
data = np.zeros([10, 20, 30, 3, 3])
data[ball, ...] = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 1, 1]])

# Some cell data
cell_data = np.zeros([9, 19, 29])
cell_data[0::2, 0::2, 0::2] = 1

# Adding the point data (see help(DataArray) for more info)
grid.addPointData(DataArray(data, range(3), 'ball'))
# Adding the cell data
grid.addCellData(DataArray(cell_data, range(3), 'checkers'))
grid.write()
```

UVW also supports writing data on 2D and 1D physical domains, for example:

```python
import sys
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
# and you can write to stdout!
with RectilinearGrid(sys.stdout, (x, y)) as grid:
  grid.addPointData(DataArray(data, range(2), 'data'))
```

## Writing in parallel with `mpi4py`

The classes contained in the `uvw.parallel` submodule support multi-process
writing using `mpi4py`. Here is a code example:

```python
import numpy as np

from mpi4py import MPI

from uvw.parallel import PRectilinearGrid
from uvw import DataArray

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N = 20

# Domain bounds per rank
bounds = [
    {'x': (-2, 0), 'y': (-2, 0)},
    {'x': (-2, 0), 'y': (0,  2)},
    {'x': (0,  2), 'y': (-2, 2)},
]

# Domain sizes per rank
sizes = [
    {'x': N, 'y': N},
    {'x': N, 'y': N},
    {'x': N, 'y': 2*N-1},  # account for overlap
]

# Size offsets per rank
offsets = [
    [0, 0],
    [0, N],
    [N, 0],
]

x = np.linspace(*bounds[rank]['x'], sizes[rank]['x'])
y = np.linspace(*bounds[rank]['y'], sizes[rank]['y'])

xx, yy = np.meshgrid(x, y, indexing='ij', sparse=True)
r = np.sqrt(xx**2 + yy**2)
data = np.exp(-r**2)

# Indicating rank info with a cell array
proc = np.ones((x.size-1, y.size-1)) * rank

with PRectilinearGrid('pgrid.pvtr', (x, y), offsets[rank]) as rect:
    rect.addPointData(DataArray(data, range(2), 'gaussian'))
    rect.addCellData(DataArray(proc, range(2), 'proc'))
```

As you can see, using `PRectilinearGrid` feels just like using
`RectilinearGrid`, except that you need to supply the position of the local grid
in the global grid numbering (the `offsets[rank]` in the above example). Note
that RecilinearGrid VTK files need an overlap in point data, hence why the
global grid size ends up being `(2*N-1, 2*N-1)`. If you forget that overlap,
Paraview (or another VTK-based software) may complain that some parts in the
global grid (aka "extents" in VTK) are missing data.

## Writing unstructured data

UVW supports VTK's UnstructuredGrid, where the geometry is given with a list of
nodes and a connectivity. The `UnstructuredGrid` class expects connectivity to
be a dictionnary enumerating the different connectivity types and the cells
associated to each type. For example:

```python
import numpy as np

from uvw import UnstructuredGrid
from uvw.unstructured import CellType

nodes = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [2, 0, 0],
    [0, 2, 0],
    [1, 2, 0],
])

connectivity = {
    CellType.QUAD: np.array([
        [0, 1, 2, 3], [2, 6, 5, 3],
    ]),
    5: np.array([[4, 2, 1]]),
}

f = UnstructuredGrid('ugrid.vtu', nodes, connectivity)
f.write()
```

As you can see, cell types can be specified with the `unstructured.CellType`
enumeration or with the underlying integer value (see
[VTKFileFormats](https://kitware.github.io/vtk-examples/site/VTKFileFormats/)
for more info). `UnstructuredGrid` performs a sanity check of the connectivity
to see if the number of nodes matches the cell type.

If you work with large amounts of unstructured data, consider checking out
[meshio](https://github.com/nschloe/meshio) which provides many different
read/write capabilities for various unstructured formats, some of which are
supported by VTK and are better than VTK's simple XML format.

## List of features

Here is a list of what is available in UVW:

### VTK file formats

- Image data (`.vti`)
- Rectilinear grid (`.vtr`)
- Structured grid (`.vts`)
- Unstructured grid (`.vtu`)
- Parallel Rectilinear grid (`.pvtr`)
- Parallel Image data (`.pvti`)
- ParaView Data (`.pvd`)

### Data representation

- ASCII
- Base64 (raw and compressed: the `compression` argument of file constructors
  can be `True`, `False`, or an integer in `[-1, 9]` for compression levels)

Note that raw binary data, while more space efficient and supported by VTK,
is not valid XML, and therefore not supported by UVW, which uses minidom for XML
writing.

### PyEVTK high-level API implementation

To facilitate transition from
[PyEVTK](https://github.com/pyscience-projects/pyevtk), UVW implements a part of
its API, without imposing restrictions on data (such as the number of components
per array) and allowing data compression. Simply replace `import pyevtk.hl` by
`import uvw.dropin.hl`. To enable compression, provide `compression=True` to any
of the functions in `uvw.dropin.hl`. *Note*: the drop-in is not automatically
tested, do not hesitate to report problems.

### Planned developments

Here is a list of future developments:

- [x] Image data
- [x] Unstructured grid
- [x] Structured grid
- [x] Parallel writing (`mpi4py`-enabled `PRectilinearGrid` and `PImageData`
      *are now available!*)
- [ ] Benchmarking + performance comparison with
      [pyevtk](https://github.com/pyscience-projects/pyevtk)


## Developing

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

### Git repository

First clone the git repository:

```
git clone https://github.com/prs513rosewood/uvw.git
```

Then you can use pip in development mode (possibly in
[virtualenv](https://virtualenv.pypa.io/en/stable/)):

```
pip install --user -e .[mpi,tests]
```
Installing with the `tests` extra pulls `vtk` as a dependency. This is because
reading files with VTK in tests is the most reliable way to check file
integrity.

## Running the tests

The tests can be run using [pytest](https://docs.pytest.org/en/latest/):

```
pytest
# or for tests with mpi
mpiexec -n 2 pytest --with-mpi
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for
details.

## Acknowledgments

* [@PurpleBooth](https://github.com/PurpleBooth)'s
  [README-Template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
