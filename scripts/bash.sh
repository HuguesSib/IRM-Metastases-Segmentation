#!/bin/bash
#SBATCH  --output=logs/%j.out	   
#SBATCH  --gres=gpu:1		  
#SBATCH  --mem=50G

source /itet-stor/hsibille/net_scratch/conda/etc/profile.d/conda.sh
conda activate project
python -u inference.py "$@"


