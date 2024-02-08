#!/bin/bash -l
#PBS -N phys_proc
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=36:ngpus=0:mem=500GB
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
python -u process_cesm_output.py ../config/cesm_tau_run9_process.yml -p 36 >& tau_run9_lim_process.log
