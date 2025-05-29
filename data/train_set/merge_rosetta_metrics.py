#!/usr/bin/env python3
import pandas as pd
import os
from af_analysis.rosetta_energy import batch_extract

# 1. 원본 CSV 파일 로드
input_csv = "/home/cseomoon/appl/af_analysis-0.1.4/data/sequence_classification/train_set_AbNb/native/ABAG_final_h3_plddt_20250522.csv"
output_csv = "/home/cseomoon/appl/af_analysis-0.1.4/data/sequence_classification/train_set_AbNb/native/ABAG_native_with_rosetta_metrics_20250523.csv"

df_original = pd.read_csv(input_csv)

# 2. PDB 파일 경로 목록 추출
pdb_files = df_original['pdb'].tolist()

# 3. 존재하는 파일만 필터링
valid_pdb_files = [path for path in pdb_files if os.path.exists(path)]
print(f"Found {len(valid_pdb_files)} valid PDB files out of {len(pdb_files)} entries")

# 4. Rosetta 에너지 계산 (병렬 처리)
df_rosetta = batch_extract(
    valid_pdb_files,
    n_jobs=48,  # CPU 코어 수에 맞게 조정
    verbose=True,
    antibody_mode=True  # 항체 모드 여부에 맞게 설정
)

# 5. 원본 데이터프레임과 Rosetta 결과 병합
df_merged = pd.merge(df_original, df_rosetta, on='pdb', how='left')

# 6. 결과 저장
df_merged.to_csv(output_csv, index=False)
print(f"[✓] 병합된 데이터 ({len(df_merged)} 행) → {output_csv} 저장 완료")
