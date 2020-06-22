from setuptools import setup
import subprocess
import os


with open("README.md", "r") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setup(name="mlmicrophysics",
          version="0.1.1",
          description="Machine learning emulator testbed for microphysics.",
          long_description=long_description,
          long_description_content_type="text/markdown",
          author="David John Gagne and Gabrielle Gantos",
          author_email="dgagne@ucar.edu",
          license="MIT",
          url="https://github.com/NCAR/mlmicrophysics",
          packages=["mlmicrophysics"],
          install_requires=["numpy",
                            "scipy",
                            "pandas",
                            "matplotlib",
                            "xarray",
                            "tensorflow",
                            "netcdf4",
                            "scikit-learn",
                            "pyyaml",
                            "pyarrow"],
          )
