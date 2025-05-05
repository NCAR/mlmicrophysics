#!/bin/bash -l
#PBS -N phys_proc
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=36:ngpus=0:mem=1480GB
#PBS -A P93300606 
#PBS -q casper 

### Merge output and error files
#PBS -j oe
#PBS -k eod

module load conda
conda activate mlmicro
conda install -c conda-forge xarray=2022.12.0
echo `which python`
echo `conda list`
cd ~/mlmicrophysics/scripts
python -u process_e3sm_output.py ../config/e3sm_tau_run1_process.yml -p 36 >& e3sm_tau_run4_process.log
