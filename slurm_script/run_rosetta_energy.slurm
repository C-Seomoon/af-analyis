#!/bin/bash
#SBATCH --job-name=bindenergy_calc
#SBATCH --partition=g4090_short
#SBATCH --nodelist=gpu7
#SBATCH --output=/home/cseomoon/appl/af_analysis-0.1.4/log/log_bindenergy_%A_%a.out
#SBATCH --error=/home/cseomoon/appl/af_analysis-0.1.4/log/log_bindenergy_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

roj /home/cseomoon/miniconda3/envs/Abnb/bin/python /home/cseomoon/appl/af_analysis-0.1.4/bindenergy_analysis.py /home/cseomoon/pect/ABAG/AbNb_benchmark/datastructure/new_dockq/final_data_20250408.csv /home/cseomoon/project/ABAG/AbNb_benchmark/datastructure/new_dockq/rosetta_20250417.csv --num_workers 24
