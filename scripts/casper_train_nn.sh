#!/bin/bash -l 
#SBATCH --job-name=nn_run5
#SBATCH --account=NAML0001
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --output=train_nn.%j.out
#SBATCH --mem=400G
module load gnu/7.3.0 openmpi python/3.6.8 cuda/10.0
source /glade/work/dgagne/ncar_pylib_dl_10/bin/activate
cd ~/mlmicrophysics/
python setup.py install
cd ~/mlmicrophysics/scripts
python -u train_mp_neural_nets.py ../config/cesm_tau_run5_full_train_nn.yml
