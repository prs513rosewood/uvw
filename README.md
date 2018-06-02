UVW - Universal VTK Writer
==========================

UVW is a small utility library to write VTK files from data contained in Numpy arrays. It handles fully-fledged `ndarrays` defined over {1, 2, 3}-d domains, with arbitrary number of components. There are no constraints on the particular order of components, although copy of data can be avoided if the array is Fortran contiguous, as VTK files are written in Fortran order. Future developments will include multi-process write support.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* [Numpy](http://www.numpy.org/). This code has been tested with Numpy version 1.14.3.
* [pytest](https://docs.pytest.org/en/latest/) for testing.

### Installing

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

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* [@PurpleBooth](https://github.com/PurpleBooth)'s [README-Template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
