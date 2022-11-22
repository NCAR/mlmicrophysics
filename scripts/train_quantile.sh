#!/bin/bash -l
#PBS -N quantile_train
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:ngpus=0:mem=128GB
#PBS -A NAML0001
#PBS -q casper

### Merge output and error files
#PBS -j oe
#PBS -k eod
conda activate mlmicro
echo `which python`
cd ~/mlmicrophysics/scripts
python -u train_quantile_neural_nets.py ../config/cesm_tau_run6_train_quantile_nn.yml
