#!/bin/bash
#SBATCH -p cascadelakegpu
#SBATCH -q normal
#SBATCH -n 16
#SBATCH -t 5-00:00:00
#SBATCH -D .
#SBATCH -J diario
#SBATCH -o semana_deChachis.%j.out
#SBATCH --mem=0

#SBATCH --mail-user=mcalo@unileon.es
#SBATCH --mail-type=END,FAIL

export PATH=/home/fcsc/ccalvo/METEO/preproces/SCIKIT-LEARN/final_beta/miniconda3/bin:$PATH
source /home/fcsc/ccalvo/METEO/preproces/SCIKIT-LEARN/final_beta/miniconda3/etc/profile.d/conda.sh
conda activate env

python svm.py

conda deactivate

