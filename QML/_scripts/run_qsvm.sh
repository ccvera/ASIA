#!/bin/bash
#SBATCH -p cascadelakegpu
#SBATCH -q normal
#SBATCH -n 16
#SBATCH -t 5-00:00:00
#SBATCH -D .
#SBATCH -J 500_qs_optimizado
#SBATCH -o qsvm_2009-01-03_solo500optimizado.%j.out
#SBATCH --mem=0

#SBATCH --mail-user=mcalo@unileon.es
#SBATCH --mail-type=END,FAIL

export PATH=/home/fcsc/ccalvo/METEO/preproces/SCIKIT-LEARN/final_beta/miniconda3/bin:$PATH
source /home/fcsc/ccalvo/METEO/preproces/SCIKIT-LEARN/final_beta/miniconda3/etc/profile.d/conda.sh
conda activate /home/fcsc/ccalvo/QUANTUMSPAIN/envs/env_qiskit_aer/

#python qsvm.py
python qsvm_optimizado.py

conda deactivate

