#!/bin/bash
#SBATCH --job-name=h3_plddt_calc
#SBATCH --partition=milan_normal
#SBATCH --nodelist=cpu10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -o /home/cseomoon/appl/af_analysis-0.1.4/slurm_script/log/h3_plddt_calc-%A.log


echo "============================================="
echo "[INFO] Job Information"
echo "---------------------------------------------"
echo "[INFO] Job Name       : $SLURM_JOB_NAME"
echo "[INFO] Job ID         : $SLURM_JOB_ID"
echo "[INFO] Submit Dir     : $SLURM_SUBMIT_DIR"
echo "[INFO] Partition      : $SLURM_JOB_PARTITION"
echo "[INFO] Node List      : $SLURM_JOB_NODELIST"
echo "[INFO] Nodes          : $SLURM_JOB_NUM_NODES"
echo "[INFO] CPUs per Task  : $SLURM_CPUS_PER_TASK"
echo "[INFO] Memory (MB)    : $SLURM_MEM_PER_NODE"
echo "[INFO] User           : $SLURM_JOB_USER"
echo "[INFO] Work Dir       : $SLURM_SUBMIT_DIR"
echo "[INFO] Tasks per Node : $SLURM_TASKS_PER_NODE"
echo "[INFO] Dependency     : $SLURM_JOB_DEPENDENCY"
echo "[INFO] GPU(s)         : $SLURM_JOB_GPUS"
echo "============================================="
echo
echo "[INFO] Job started at : $(date)"
echo "[INFO] Job allocated on node(s): $SLURM_NODELIST"
echo "[INFO] Job running on: $(hostname)"
echo

/home/cseomoon/miniconda3/envs/Abnb/bin/python  /home/cseomoon/appl/af_analysis-0.1.4/src/af_analysis/calc_h3_plddt.py --input /home/cseomoon/appl/af_analysis-0.1.4/data/test_set/ABAG_final_test_dataset_20250512.csv --output /home/cseomoon/appl/af_analysis-0.1.4/data/test_set/ABAG_final_h3_plddt_20250522.csv --cpu 32

echo "[INFO] Job completed at: $(date)"
