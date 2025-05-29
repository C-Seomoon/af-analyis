#!/bin/bash
#SBATCH --job-name=merge_energy
#SBATCH --output=./log/merge_energy.log
#SBATCH --error=./log/merge_energy.err
#SBATCH --cpus-per-task=2
#SBATCH --dependency=afterany:$1   # 첫 번째 인자로 받은 작업 ID에 의존

# 결과 디렉토리
OUTPUT_DIR=$2
FINAL_OUTPUT="${OUTPUT_DIR}/combined_energy_results.csv"

echo "Starting to merge results from $OUTPUT_DIR"

# 병합 스크립트 실행
python - << EOF
import pandas as pd
import glob
import os

# 결과 파일 찾기
result_files = glob.glob("${OUTPUT_DIR}/energy_result_*.csv")
print(f"Found {len(result_files)} result files to merge")

if not result_files:
    print("No result files found!")
    exit(1)

# 모든 결과 읽어서 하나로 합치기
dfs = []
for f in result_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"Loaded {f}: {len(df)} rows")
    except Exception as e:
        print(f"Error loading {f}: {e}")

if not dfs:
    print("No valid data frames to merge!")
    exit(1)

# 합치기
all_results = pd.concat(dfs)
print(f"Combined data frame has {len(all_results)} rows")

# 합친 결과 저장
all_results.to_csv("${FINAL_OUTPUT}", index=False)
print(f"Results saved to ${FINAL_OUTPUT}")
EOF

echo "Merge completed"
