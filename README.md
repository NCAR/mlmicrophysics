# mlmicrophysics
Machine learning emulators for microphysical processes in CESM.

## Requirements

The library has been tested with Python 3.6 and 3.7.
The mlmicrophysics library requires the following Python libraries:
* numpy
* scipy
* matplotlib
* scikit-learn
* tensorflow
* keras
* pandas
* xarray
* pyyaml
* netcdf4

You can install the dependencies using conda or pip depending on your local
Python installation. In order to compile the fortran code within the library,
you will need gfortran on your system.

## Installation
To install and compile the library, run the following command:
```bash
git clone https://github.com/NCAR/mlmicrophysics.git
cd mlmicrophysics
pip install .
```

## Running
To train a new microphysics neural network emulator, you will first need to process
the CESM CAM output files using the `scripts/process_cesm_output.py` script. The
process script converts the CAM netCDF files to a set of csv files and filters
out non-cloud grid cells. The script requires a yaml config file. See `config/cesm_tau_run5_full_process.yml` for
an example. To run the processing script:
```bash
cd ~/mlmicrophysics/scripts
python -u process_cesm_output.py ../config/cesm_tau_run5_full_process.yml -p 5 >& tau_run5_process.log
```

Once the data are processed, you can train a set of neural network emulators with `scripts/train_mp_neural_nets.py`.
This script pre-processes the training and validation data, trains a set of neural networks
and saves them and their verification statistics to an output directory.

## CAM modifications
The code used to run the neural network emulator and TAU in CAM 6 can be found in the `cam_code` and `cam_tau_code` directories. These can be added to the CESM cam6_0_000 code as a SourceMod if you want to run your own simulations.

## Machine Learning the Warm Rain Process
Some of the figures and tables for the Machine Learning the Warm Rain Process paper were generated using the code in `notebooks/nn_train_evaluation_20190512.ipynb`. 

## Contact
If you have issues with the library, please create an issue on the github page.
General questions can be sent to David John Gagne at dgagne@ucar.edu.
