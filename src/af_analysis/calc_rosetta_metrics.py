#!/usr/bin/env python3
import pandas as pd
import os
import argparse
from af_analysis.rosetta_energy import batch_extract

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Calculate Rosetta energy metrics and merge with existing CSV")
    parser.add_argument("--input", required=True, help="Input CSV file path containing PDB references")
    parser.add_argument("--output", required=True, help="Output CSV file path for merged results")
    parser.add_argument("--cpu", type=int, default=4, help="Number of CPU cores to use for parallel processing")
    args = parser.parse_args()

    # 1. 원본 CSV 파일 로드
    df_original = pd.read_csv(args.input)
    print(f"Loaded original data with {len(df_original)} entries")

    # 2. PDB 파일 경로 목록 추출
    pdb_files = df_original['pdb'].tolist()

    # 3. 존재하는 파일만 필터링
    valid_pdb_files = [path for path in pdb_files if os.path.exists(path)]
    print(f"Found {len(valid_pdb_files)} valid PDB files out of {len(pdb_files)} entries")

    # 4. Rosetta 에너지 계산 (병렬 처리)
    df_rosetta = batch_extract(
        valid_pdb_files,
        n_jobs=args.cpu,
        verbose=True,
        antibody_mode=True
    )

    # 5. 원본 데이터프레임과 Rosetta 결과 병합
    df_merged = pd.merge(df_original, df_rosetta, on='pdb', how='left')

    # 6. 결과 저장
    df_merged.to_csv(args.output, index=False)
    print(f"[✓] 병합된 데이터 ({len(df_merged)} 행) → {args.output} 저장 완료")

if __name__ == "__main__":
    main()