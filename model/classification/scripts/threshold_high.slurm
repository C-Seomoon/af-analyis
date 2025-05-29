#!/bin/bash
#SBATCH --job-name=cls_framework
#SBATCH --partition=milan_short
##SBATCH --nodelist=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -o /home/cseomoon/appl/af_analysis-0.1.4/model/classification/log/threshold_high-%A.out
#SBATCH -e /home/cseomoon/appl/af_analysis-0.1.4/model/classification/log/threshold_high-%A.err


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

# 타임스탬프 생성 (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 결과 디렉토리 설정
OUTPUT_DIR="/home/cseomoon/appl/af_analysis-0.1.4/model/classification/results/threshold_high_${TIMESTAMP}"
echo "[INFO] Results will be saved to: $OUTPUT_DIR"

# 작업 디렉토리로 이동 (필요시)
cd /home/cseomoon/appl/af_analysis-0.1.4/model/classification

# 스크립트 실행
/home/cseomoon/miniconda3/envs/Abnb/bin/python /home/cseomoon/appl/af_analysis-0.1.4/model/classification/train_models.py \
    --input_file /home/cseomoon/appl/af_analysis-0.1.4/model/classification/data/final_data_with_rosetta_scaledRMSD_20250423.csv \
    --n_jobs $SLURM_CPUS_PER_TASK \
    --outer_folds 5 \
    --inner_folds 3 \
    --random_iter 100 \
	--threshold 0.8
    --output_dir $OUTPUT_DIR

echo "작업 완료"
echo "[INFO] Job completed at: $(date)"


