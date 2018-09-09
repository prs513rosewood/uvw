import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uvw",
    version="0.0.3",
    author="Lucas Frérot",
    author_email="lucas.frerot@epfl.ch",
    description="Universal VTK Writer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://c4science.ch/source/uvw.git",
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
