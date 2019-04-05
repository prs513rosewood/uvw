import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read().replace('lang=', '')

setuptools.setup(
    name="uvw",
    version="0.0.7",
    author="Lucas Frérot",
    author_email="lucas.frerot@epfl.ch",
    description="Universal VTK Writer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prs513rosewood/uvw",
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Intended Audience :: Science/Research",
    )
)
