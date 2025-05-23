#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J train_gqformer.py
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH --gres:gpu=8
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
