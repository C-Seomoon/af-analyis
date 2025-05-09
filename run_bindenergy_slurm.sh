#!/bin/bash
#SBATCH --job-name=bindenergy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-9

# 타임스탬프 생성 (YYYYMMDD_HHMMSS 형식)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 출력 및 에러 파일 설정 (타임스탬프 포함)
#SBATCH --output=log_bindenergy_%A_%a_${TIMESTAMP}.out
#SBATCH --error=log_bindenergy_%A_%a_${TIMESTAMP}.err

# 작업 디렉토리로 이동
cd $SLURM_SUBMIT_DIR

# 시작 시간과 타임스탬프 기록
echo "[INFO] Job started at: $(date)"
echo "[INFO] Timestamp: $TIMESTAMP"

# 도움말 표시 함수
show_help() {
    echo "Usage: $0 <input_csv> <output_csv> [options]"
    echo
    echo "Options:"
    echo "  --cpus NUM       CPU 코어 수 (기본값: 8)"
    echo "  --mem MEM        메모리 크기 (기본값: 16G)"
    echo "  --time HH:MM:SS  최대 실행 시간 (기본값: 24:00:00)"
    echo "  --array START-END  SLURM 배열 작업 범위 (기본값: 0-9)"
    echo "  --num_workers N  작업당 병렬 프로세스 수 (기본값: CPU 코어 수)"
    echo "  --chunk_size N   작업자당 청크 크기 (기본값: 1)"
    echo
    echo "Examples:"
    echo "  $0 data/input.csv results/output.csv"
    echo "  $0 data/input.csv results/output.csv --cpus 16 --mem 32G --array 0-19"
    exit 1
}

# 인자 확인
if [ "$#" -lt 2 ]; then
    show_help
fi

# 기본 설정값
INPUT_CSV=$1
OUTPUT_CSV=$2
shift 2

CPUS=8
MEM="16G"
TIME="24:00:00"
ARRAY="0-9"
NUM_WORKERS=0
CHUNK_SIZE=1

# 추가 옵션 처리
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpus)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                CPUS=$2
                NUM_WORKERS=$CPUS  # 기본적으로 워커 수는 CPU 수와 동일하게 설정
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --mem)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                MEM=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --time)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                TIME=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --array)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                ARRAY=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --num_workers)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                NUM_WORKERS=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --chunk_size)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                CHUNK_SIZE=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# 타임스탬프 생성 (YYYYMMDD_HHMMSS 형식)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 출력 파일에 타임스탬프 추가
OUTPUT_BASE=$(basename "$OUTPUT_CSV" .csv)
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_BASE}_${TIMESTAMP}.csv"

# 로그 디렉토리 확인 및 생성
LOG_DIR="./log"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo "Created log directory: $LOG_DIR"
fi

# 로그 파일 이름 설정 (타임스탬프 포함)
LOG_OUTPUT="${LOG_DIR}/bindenergy_%A_%a_${TIMESTAMP}.out"
LOG_ERROR="${LOG_DIR}/bindenergy_%A_%a_${TIMESTAMP}.err"

echo "준비 설정:"
echo "- 입력 CSV: $INPUT_CSV"
echo "- 출력 CSV: $OUTPUT_CSV"
echo "- CPU 코어 수: $CPUS"
echo "- 메모리: $MEM"
echo "- 최대 실행 시간: $TIME"
echo "- 작업 배열: $ARRAY"
echo "- 작업당 워커 수: $NUM_WORKERS"
echo "- 청크 크기: $CHUNK_SIZE"
echo "- 타임스탬프: $TIMESTAMP"
echo "- 로그 출력: $LOG_OUTPUT"
echo "- 로그 에러: $LOG_ERROR"

# SLURM 작업 제출
echo "SLURM 작업 제출 중..."
sbatch \
    --job-name="bindenergy_${TIMESTAMP}" \
    --output="$LOG_OUTPUT" \
    --error="$LOG_ERROR" \
    --ntasks=1 \
    --cpus-per-task=$CPUS \
    --mem=$MEM \
    --time=$TIME \
    --array=$ARRAY \
    --wrap="python bindenergy_analysis.py $INPUT_CSV $OUTPUT_CSV --num_workers $NUM_WORKERS --chunk_size $CHUNK_SIZE --slurm_array_task_id \$SLURM_ARRAY_TASK_ID --slurm_array_task_count \$SLURM_ARRAY_TASK_COUNT"

echo "작업이 제출되었습니다. 'squeue' 명령으로 상태를 확인하세요." 