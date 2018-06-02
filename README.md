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
pip install --user git+https://c4science.ch/source/uvw.git
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


## Developing

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Git repository

First clone the git repository:

```
git clone https://c4science.ch/source/uvw.git
```

Then you can use pip in development mode (most convenient in virtualenv):

```
pip install --user -e .
```

## Running the tests

The tests can be run using [pytest](https://docs.pytest.org/en/latest/):

```
pytest tests
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

* [@PurpleBooth](https://github.com/PurpleBooth)'s [README-Template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
