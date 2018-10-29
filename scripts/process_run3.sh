#!/bin/bash -l
#PBS -N tau_run3
#PBS -A NAML0001
#PBS -l walltime=02:00:00
#PBS -q regular
#PBS -j oe
#PBS -m abe
#PBS -M dgagne@ucar.edu
#PBS -l select=1:ncpus=36:mpiprocs=36
module purge
source ~/.bash_profile
export PATH="/glade/u/home/dgagne/miniconda3/envs/deep/bin:$PATH"
cd ~/mlmicrophysics/scripts
python -u process_cesm_output.py ../config/cesm_tau_run3_process.yml -p 5 >& tau_run3_process.log
