#!/bin/bash -l
#PBS -N phys_proc
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=36:ngpus=0:mem=383GB
#PBS -A NAML0001
#PBS -q casper

### Merge output and error files
#PBS -j oe
#PBS -k eod
conda activate mlmicro
echo `which python`
cd ~/mlmicrophysics/scripts
python -u process_cesm_output.py ../config/cesm_tau_run5_phys_process.yml -p 36 >& tau_phys_process.log
python -u process_cesm_output.py ../config/cesm_sd_phys_process.yml -p 36 >& tau_phys_process.log