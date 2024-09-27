#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -q normal
#SBATCH -n 16
#SBATCH -t 1-00:00:00
#SBATCH -D $work_dir
#SBATCH -J Merge_train_dataset
#SBATCH -o Merge_train_dataset.%j.out
#SBATCH --mem=0

module load intel_18
module load python_2.7

python merge_csv.py -t $work_dir/csv_trihorario_train -v $work_dir/csv_trihorario_val
