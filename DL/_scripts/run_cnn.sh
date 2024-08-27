#!/bin/bash
#SBATCH -p cascadelakegpu
#SBATCH -q normal
#SBATCH -n 36
#SBATCH -t 5-00:00:00
#SBATCH -D $WORKDIR/DL
#SBATCH -J cnn_completo
#SBATCH -o $WORKDIR/DL/_logs/cnn_completo.%j.out
#SBATCH --mem=0
#SBATCH --gpus=1

module load conda
conda activate $WORKDIR/envs/env_tf

python tf_cnn.py

conda deactivate
