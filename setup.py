import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="microor",
    version="0.1",
    author="Pedro Serrano Drozdowskyj, Manuel Pasieka",
    author_email="pedro.serrano@vbcf.ac.at, manuel.pasieka@protonmail.ch",
    description="Automatic Object Recognition and quantification of micro array images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://mapa17@bitbucket.org/csf_biocomp/microor.git",
    packages=setuptools.find_packages(),
    scripts=['microorCLI.py'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
