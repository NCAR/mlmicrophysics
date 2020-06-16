from numpy.distutils.core import setup, Extension
import subprocess
import os

ext_call_collect = Extension(name="mlmicrophysics.call_collect",
                             sources=["mlmicrophysics/call_collect.f90"],
                             extra_objects=["mlmicrophysics/stochastic_collect_tau_cam.o"])

if __name__ == "__main__":
    #fortran_compiler = "gfortran"
    #os.chdir("mlmicrophysics")
    #subprocess.call([fortran_compiler, "-c", "stochastic_collect_tau_cam.f90"])
    #os.chdir("../")
    setup(name="mlmicrophysics",
          version="0.1",
          description="Machine learning emulator testbed for microphysics.",
          author="David John Gagne and Negin Sobhani",
          author_email="dgagne@ucar.edu",
          license="MIT",
          url="https://github.com/NCAR/mlmicrophysics",
          packages=["mlmicrophysics"],
          #data_files=[("mlmicrophysics", ["mlmicrophysics/KBARF"])],
          install_requires=["numpy",
                            "scipy",
                            "pandas",
                            "matplotlib",
                            "xarray",
                            #"tensorflow",
                            #"keras",
                            "netcdf4",
                            "scikit-learn",
                            "pyyaml",
                            "pyarrow"],
          #ext_modules=[ext_call_collect]
          )
