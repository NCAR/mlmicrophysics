from setuptools import setup

if __name__ == "__main__":
    setup(name="mlmicrophysics",
          version="0.1",
          description="Machine learning emulator testbed for microphysics.",
          author="David John Gagne and Negin Sobhani",
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
                            "keras",
                            "netcdf4",
                            "scikit-learn",
                            "pyyaml"])
