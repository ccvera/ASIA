#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -q normal
#SBATCH -n 16
#SBATCH -t 1-00:00:00
#SBATCH -D /home/fcsc/ccalvo/ML_meteo/utils/
#SBATCH -J FilterNC
#SBATCH -o FilterNC.%j.out
#SBATCH --mem=0

#SBATCH --mail-user=carmen.calvo@scayle.es
#SBATCH --mail-type=END,FAIL

source /soft/calendula2/intel/ipsxe_2018_u4/parallel_studio_xe_2018/psxevars.sh
export PATH=/home/fcsc/ccalvo/test_GPUs/utils/python/2.7.12/bin:$PATH
export LD_LIBRARY_PATH=/home/fcsc/ccalvo/test_GPUs/utils/python/2.7.12/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/fcsc/ccalvo/test_GPUs/utils/python/2.7.12/lib/python2.7/site-packages

# Train
python filter_nc_variables.py -D ~/preproces/dataset_train/raw -O /home/fcsc/ccalvo/METEO/nc_train
# Validation
#python filter_nc_variables.py -D ~/preproces/dataset_validation/raw -O /home/fcsc/ccalvo/METEO/nc_val
