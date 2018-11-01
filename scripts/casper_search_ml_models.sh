#!/bin/bash -l
#SBATCH -J search_ml
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --mem=382G
#SBATCH -t 6:00:00
#SBATCH -A NAML0001
#SBATCH -p dav
#SBATCH -C casper
#SBATCH -o search_ml.log
module purge
export HOME="/glade/u/home/dgagne"
export PATH="/glade/u/home/dgagne/miniconda3/envs/ml/bin:$PATH"
cd $HOME/mlmicrophysics/scripts
echo `which python`
python -u search_ml_model_params.py ../config/cesm_tau_run2_ml_models.yml -p 2 >& ml_search_casper.log
