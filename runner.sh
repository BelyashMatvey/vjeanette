#!/bin/bash -l

#SBATCH --job-name=Task_train        # Job name
#SBATCH --output=Task_train.%j.log   # Standard output and error log 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=belyakov06mat@mail.ru
#SBATCH --cpus-per-task=32         # Run on a single CPU
#SBATCH --mem=128gb   
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

cd ~/vidjil/seq2vdj
conda activate py_env
python run.py