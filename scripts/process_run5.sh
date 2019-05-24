#!/bin/bash -l
#SBATCH --job-name=tau_run5
#SBATCH --account=NAML0001
#SBATCH --time=02:00:00
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=dav
#SBATCH --output=tau_run2.out.%j
#SBATCH --mem=128G
module purge
module load gnu/7.3.0 openmpi-x python/3.6.8
source /glade/work/dgagne/ncar_pylib_dl_10/bin/activate
cd ~/mlmicrophysics
python setup.py install
cd ~/mlmicrophysics/scripts
python -u process_cesm_output.py ../config/cesm_tau_run5_process.yml -p 5 >& tau_run5_process.log
