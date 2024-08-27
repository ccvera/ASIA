#!/bin/bash
#SBATCH -p icelakegpu1tb
#SBATCH -q normal
#SBATCH -n 64
#SBATCH -t 5-00:00:00
#SBATCH -D $WORKDIR/DL
#SBATCH -J SA-LSTM_completo_torch
#SBATCH -o $WORKDIR/DL/_logs/SA-LSTM_torch_completo.%j.out
#SBATCH --mem=0
#SBATCH --gpus=1

module load conda
conda activate $WORKDIR/envs/env_torch_nuevo

python torch_SA-LSTM.py

conda deactivate
