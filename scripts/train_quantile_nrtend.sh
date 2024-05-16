#!/bin/bash -l
#PBS -N quantile_train
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=16:ngpus=1:mem=256GB
#PBS -l gpu_type=v100
#PBS -A P93300606 
#PBS -q casper

### Merge output and error files
#PBS -j oe
#PBS -k eod
module load conda
conda activate mlmicro
echo `which python`
export LD_PRELOAD=/glade/work/wchuang/conda-envs/mlmicro/lib/libstdc++.so
cd ~/mlmicrophysics/scripts
python -u train_quantile_neural_nets.py ../config/training_tau_run12_nrtend.yml &> training_tau_run12_nrtend_add_CLD_lev_FREQR_overfit_attempt_3.log
