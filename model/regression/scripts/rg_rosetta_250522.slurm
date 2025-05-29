#!/bin/bash
#SBATCH --job-name=rg_with_rosetta
#SBATCH --partition=rome_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -o /home/cseomoon/appl/af_analysis-0.1.4/model/regression/log/rg_250522-%A.out
#SBATCH -e /home/cseomoon/appl/af_analysis-0.1.4/model/regression/log/rg_250522-%A.err

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=48	   # 또는 필요한 스레드 수
export MKL_THREADING_LAYER=GNU # Intel MKL만 쓰도록 강제

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

# 작업 디렉토리로 이동
cd /home/cseomoon/appl/af_analysis-0.1.4
# 스크립트 실행

TS=$(date +%Y%m%d_%H%M%S)
OUT_BASE="/home/cseomoon/appl/af_analysis-0.1.4/model/regression/results/rg_rosetta"
OUTPUT_DIR="${OUT_BASE}_${TS}"

echo "[INFO] Saving results to ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Python 실행
/home/cseomoon/miniconda3/envs/Abnb/bin/python -m model.regression.regression_nested_cv \
    --input_file /home/cseomoon/appl/af_analysis-0.1.4/model/regression/data/train/final_data_with_new_rosetta_metrics_20250521.csv \
    --n_jobs $SLURM_CPUS_PER_TASK \
    --outer_folds 5 \
    --inner_folds 3 \
    --random_iter 50 \
    --output_dir "${OUTPUT_DIR}"
echo "작업 완료"
echo "[INFO] Job completed at: $(date)"
