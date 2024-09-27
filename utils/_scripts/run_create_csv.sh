#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -q normal
#SBATCH -n 16
#SBATCH -t 1-00:00:00
#SBATCH -D $work_dir
#SBATCH -J CreatingDataset
#SBATCH -o CreatingDataset.%j.out
#SBATCH --mem=85GB

module load intel_18
module load python_2.7

# Train
python create_csv_trihorario.py -D $work_dir/preproces/dataset_train/nc_train -O $work_dir/csv_trihorario_train -f $work_dir/ground_truth.nc

# Validation
#python create_csv_trihorario.py -D $work_dir/preproces/dataset_val/nc_val -O $work_dir/csv_trihorario_val -f $work_dir/ground_truth.nc
