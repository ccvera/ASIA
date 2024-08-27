#!/bin/bash
#SBATCH -p cascadelakegpu
#SBATCH -q normal
#SBATCH -n 36
#SBATCH -t 5-00:00:00
#SBATCH -D $WORKDIR/DL
#SBATCH -J gru_completo
#SBATCH -o $WORKDIR/DL/_logs/gru_completo.%j.out
#SBATCH --mem=0
#SBATCH --gpus=1

module load conda
conda activate $WORKDIR/envs/env_tf

python tf_gru.py

conda deactivate
