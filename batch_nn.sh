#!/bin/bash -l
#SBATCH -J micro_nn
#SBATCH --account=NAML0001
#SBATCH -t 23:00:00
#SBATCH --mem=128G
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH -o sd_nn_half.o
#SBATCH -e sd_nn_half.o
module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib ncar_20200417
export PATH="/glade/u/home/ggantos/ncar_20200417/bin:$PATH"

pip install /glade/work/ggantos/mlmicrophysics/.
python scripts/train_mp_neural_nets.py config/cesm_sd_full_train_nn.yml
