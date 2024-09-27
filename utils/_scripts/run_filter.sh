#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -q normal
#SBATCH -n 16
#SBATCH -t 1-00:00:00
#SBATCH -D $work_dir
#SBATCH -J FilterNC
#SBATCH -o FilterNC.%j.out
#SBATCH --mem=0


module load intel_18
module load python_2.7

python preprocess_daily_data.py -D $work_dir/preproces/dataset_train/new_raw -O $work_dir/preproces/dataset_train/new_nc_train
