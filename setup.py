"Install script"
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read().replace('lang=', '')

setuptools.setup(
    name="uvw",
    version="0.3.2",
    author="Lucas Frérot",
    author_email="lucas.frerot@protonmail.com",
    description="Universal VTK Writer for Numpy Arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prs513rosewood/uvw",
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    extras_require={
      "mpi": ['mpi4py'],
      "tests": ['pytest', 'pytest-mpi', 'pytest-cov', 'mpi4py', 'vtk'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Intended Audience :: Science/Research",
    ],
)
