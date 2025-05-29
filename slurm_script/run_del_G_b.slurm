#!/bin/bash
#SBATCH --job-name=energy_calc
#SBATCH --output=./log/energy_calc_%A_%a.log
#SBATCH --error=./log/energy_calc_%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --array=0-80     # 81개의 작업으로 분할
#SBATCH --partition=rome_veryshort

# 입력 파일 경로
INPUT_CSV="/home/cseomoon/appl/af_analysis-0.1.4/data/sequence_classification/train_set_AbNb/decoy/AbNb_decoy_data_250523.csv"
OUTPUT_DIR="/home/cseomoon/appl/af_analysis-0.1.4/results/energy_calc"
SCRIPT_PATH="/home/cseomoon/appl/af_analysis-0.1.4/data/train_set/energy_calculator.py"
MERGE_SCRIPT="/home/cseomoon/appl/af_analysis-0.1.4/slurm_script/merge_script.sh"

# 결과 및 로그 디렉토리 생성
mkdir -p $OUTPUT_DIR
mkdir -p ./log

# 총 계산 수와 작업 분할
TOTAL=18250
ARRAY_SIZE=81  # 0-80까지 총 81개 작업
CHUNK=$((TOTAL / ARRAY_SIZE))
START=$((SLURM_ARRAY_TASK_ID * CHUNK))

# 마지막 작업은 나머지 모두 처리
if [ $SLURM_ARRAY_TASK_ID -eq $((ARRAY_SIZE-1)) ]; then
    END=$TOTAL
else
    END=$((START + CHUNK))
fi

# 출력 파일명 설정
OUTPUT_FILE="${OUTPUT_DIR}/energy_result_${SLURM_ARRAY_TASK_ID}.csv"

echo "Starting calculation for range $START-$END (Task ID: ${SLURM_ARRAY_TASK_ID})"
echo "Total items: $TOTAL, Total tasks: $ARRAY_SIZE, Chunk size: $CHUNK"

# 에너지 계산 실행
/home/cseomoon/miniconda3/envs/Abnb/bin/python $SCRIPT_PATH $INPUT_CSV \
    --path_column pdb \
    --cpu $SLURM_CPUS_PER_TASK \
    --start $START \
    --end $END \
    --output $OUTPUT_FILE

echo "Calculation completed for task ${SLURM_ARRAY_TASK_ID}"

# 배열 작업의 마지막 작업이 완료되면 결과 병합 작업 제출
if [ $SLURM_ARRAY_TASK_ID -eq $((ARRAY_SIZE-1)) ]; then
    echo "Submitting merge job..."
    # 병합 스크립트가 존재하는지 확인
    if [ -f "$MERGE_SCRIPT" ]; then
        sbatch $MERGE_SCRIPT $SLURM_ARRAY_JOB_ID $OUTPUT_DIR
    else
        echo "Error: Merge script not found at $MERGE_SCRIPT"
    fi
fi
