#!/bin/bash
#SBATCH -p icelakegpu1tb
#SBATCH -q normal
#SBATCH -n 64
#SBATCH -t 5-00:00:00
#SBATCH -D $WORKDIR/DL
#SBATCH -J lstm_completo
#SBATCH -o $WORKDIR/DL/_logs/lstm_completo.%j.out
#SBATCH --mem=0
##SBATCH --gpus=1

module load conda
conda activate $WORKDIR/envs/env_tf

python tf_lstm.py

conda deactivate
