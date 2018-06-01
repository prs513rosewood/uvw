import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uvw",
    version="0.0.1",
    author="Lucas Frérot",
    author_email="lucas.frerot@epfl.ch",
    description="Universal VTK Writer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
)
