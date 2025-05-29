#!/bin/bash
#SBATCH --job-name=rosetta_h3_calc
#SBATCH --output=/home/cseomoon/appl/af_analysis-0.1.4/slurm_script/log/rosetta_h3_calc_%j.out
#SBATCH --partition=milan_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

# 작업 디렉토리
WORKDIR="/home/cseomoon/appl/af_analysis-0.1.4"
cd $WORKDIR

# 입력 및 출력 파일 경로
INPUT_CSV="$WORKDIR/data/sequence_classification/metric_calculation/AbNb_negative_naive_20250521_224325.csv"
TEMP_OUTPUT="$WORKDIR/data/sequence_classification/metric_calculation/AbNb_negative_naive_with_rosetta.csv"
FINAL_OUTPUT_DIR="$WORKDIR/data/sequence_classification/train_set_AbNb/decoy/AbNb_final_data_250522.csv"

# 1단계: Rosetta 에너지 계산
echo "Step 1: Running Rosetta energy calculation..."
/home/cseomoon/miniconda3/envs/Abnb/bin/python /home/cseomoon/appl/af_analysis-0.1.4/src/af_analysis/calc_rosetta_energy.py --input $INPUT_CSV --output $TEMP_OUTPUT --cpu 48

# 종료 상태 확인
if [ $? -ne 0 ]; then
    echo "Error: Rosetta energy calculation failed!"
    exit 1
fi
echo "Rosetta energy calculation completed successfully."

# 2단계: H3 pLDDT 계산
echo "Step 2: Running H3 pLDDT calculation..."
 /home/cseomoon/miniconda3/envs/Abnb/bin/python /home/cseomoon/appl/af_analysis-0.1.4/src/af_analysis/calc_h3_plddt.py --input $TEMP_OUTPUT --output $FINAL_OUTPUT_DIR --cpu 48

# 종료 상태 확인
if [ $? -ne 0 ]; then
    echo "Error: H3 pLDDT calculation failed!"
    exit 1
fi
echo "H3 pLDDT calculation completed successfully."

echo "All processing completed!"
