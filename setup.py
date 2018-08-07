import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="marquant",
    version="0.2",
    author="Pedro Serrano Drozdowskyj, Manuel Pasieka",
    author_email="pedro.serrano@vbcf.ac.at, manuel.pasieka@protonmail.ch",
    description="Automatic Object Recognition and quantification of micro array (and other) images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CSF-BioComp/marquant.git",
    packages=setuptools.find_packages(),
    scripts=['marquantCLI.py'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
