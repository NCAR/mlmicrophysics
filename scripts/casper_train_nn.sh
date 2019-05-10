#!/usr/bin/env bash
#SBATCH --job-name=nn_run2
#SBATCH --account=NAML0001
#SBATCH --time=02:00:00
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --output=train_nn.%j.out
#SBATCH --mem=256G
module purge
module load gnu/7.3.0 openmpi-x/3.1.0 python/3.6.4 cuda/9.2 netcdf
source /glade/work/dgagne/ncar_pylib_dl/bin/activate
cd ~/mlmicrophysics/
python setup.py install
cd ~/mlmicrophysics/scripts
python train_mp_neural_nets.py ../config/cesm_tau_run_2_train_nn.yml