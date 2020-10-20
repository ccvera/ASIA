#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -q normal
#SBATCH -n 16
#SBATCH -t 1-00:00:00
#SBATCH -D /home/fcsc/ccalvo/ML_meteo/utils/
#SBATCH -J CreatingDataset
#SBATCH -o CreatingDataset.%j.out
#SBATCH --mem=85GB

#SBATCH --mail-user=carmen.calvo@scayle.es
#SBATCH --mail-type=END,FAIL

source /soft/calendula2/intel/ipsxe_2018_u4/parallel_studio_xe_2018/psxevars.sh
export PATH=/home/fcsc/ccalvo/test_GPUs/utils/python/2.7.12/bin:$PATH
export LD_LIBRARY_PATH=/home/fcsc/ccalvo/test_GPUs/utils/python/2.7.12/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/fcsc/ccalvo/test_GPUs/utils/python/2.7.12/lib/python2.7/site-packages

# Train
#python create_csv_trihorario.py -D /home/fcsc/ccalvo/METEO/nc_train -O /home/fcsc/ccalvo/METEO/csv_trihorario_train_rangosEduardo -f /home/fcsc/ccalvo/METEO/datos_interpolados.nc
python create_csv_trihorario.py -D /home/fcsc/ccalvo/METEO/preproces/dataset_train/new_nc_train -O /home/fcsc/ccalvo/METEO/csv_trihorario_train_rangosEduardo_tmp -f /home/fcsc/ccalvo/METEO/datos_interpolados.nc

# Validation
#python create_csv_trihorario.py -D /home/fcsc/ccalvo/METEO/nc_val -O /home/fcsc/ccalvo/METEO/csv_trihorario_val_rangosEduardo -f /home/fcsc/ccalvo/METEO/datos_interpolados.nc
