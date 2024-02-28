#!/bin/bash
#SBATCH --mail-user=jyin97@uwo.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:t4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=20:0
#SBATCH --account=def-ling
#SBATCH --output=../out/%j.out   

module load gcc arrow python
source ~/ENV/bin/activate

export HF_HOME=/home/rzyjch/projects/def-ling/rzyjch/22summer/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd ..
python 18adamW.py