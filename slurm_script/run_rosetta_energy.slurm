#!/bin/bash
#SBATCH --job-name=bindenergy_calc
#SBATCH --partition=rome_short
#SBATCH --output=/home/cseomoon/appl/af_analysis-0.1.4/slurm_script/log/log_rosetta_250522_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

/home/cseomoon/miniconda3/envs/Abnb/bin/python /home/cseomoon/appl/af_analysis-0.1.4/data/train_set/merge_rosetta_metrics.py
