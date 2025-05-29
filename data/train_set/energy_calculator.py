#!/usr/bin/env python
# energy_calculator.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import argparse
from af_analysis.binding_energy import initialize_pyrosetta, compute_single_binding_energy
import gc
import time

def process_file(file_path):
    """단일 파일의 결합 에너지 계산"""
    # CIF 파일 자동 변환은 compute_single_binding_energy 내부에서 처리됨
    energy, error = compute_single_binding_energy(file_path, antibody_mode=True)
    return file_path, energy, error

def main():
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description="Calculate binding energies for protein structures")
    parser.add_argument("csv_file", help="CSV file with model paths")
    parser.add_argument("--path_column", default="pdb", help="Column name with file paths")
    parser.add_argument("--cpu", type=int, default=mp.cpu_count()-2, help="Number of CPU cores to use")
    parser.add_argument("--output", help="Output CSV file path (default: input_energy.csv)")
    parser.add_argument("--start", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end", type=int, default=None, help="End index for processing")
    args = parser.parse_args()
    
    # 출력 파일명 설정
    if not args.output:
        args.output = os.path.splitext(args.csv_file)[0] + "_energy.csv"
    
    # CSV 파일 로드
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} rows from {args.csv_file}")
    
    # 'del_G_B' 컬럼이 없으면 추가
    if 'del_G_B' not in df.columns:
        df['del_G_B'] = np.nan
    
    # 경로 컬럼 확인
    if args.path_column not in df.columns:
        print(f"Error: Column '{args.path_column}' not found in the CSV file")
        return
    
    # 유효한 파일 경로만 추출
    valid_paths = df[pd.notna(df[args.path_column])][args.path_column].tolist()
    files_to_process = [path for path in valid_paths if os.path.exists(path)]
    print(f"Found {len(files_to_process)} valid file paths")
    
    # 이미 계산된 항목 제외
    files_to_calculate = []
    for file_path in files_to_process:
        idx = df[df[args.path_column] == file_path].index
        if len(idx) > 0 and pd.isna(df.loc[idx[0], 'del_G_B']):
            files_to_calculate.append(file_path)

    # 작업 범위 제한 - 이 부분을 계산 전으로 이동
    if args.end is not None:
        files_to_calculate = files_to_calculate[args.start:args.end]
    else:
        files_to_calculate = files_to_calculate[args.start:]

    print(f"Need to calculate {len(files_to_calculate)} files")
    
    # PyRosetta 한 번만 초기화
    initialize_pyrosetta(silent=True)
    
    # 병렬 처리
    results = []
    errors = []
    
    if args.cpu > 1 and len(files_to_calculate) > 1:
        with mp.Pool(processes=args.cpu) as pool:
            for file_path in tqdm(files_to_calculate, desc="Calculating binding energy"):
                try:
                    result = process_file(file_path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    errors.append(f"{file_path}: Unexpected error: {str(e)}")
    else:
        # 단일 코어 처리
        for file_path in tqdm(files_to_calculate, desc="Calculating binding energy"):
            try:
                result = process_file(file_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                errors.append(f"{file_path}: Unexpected error: {str(e)}")
    
    # 결과 요약
    print(f"Successfully calculated binding energy for {len(results)} of {len(files_to_calculate)} files")
    if errors:
        print(f"Errors occurred for {len(errors)} files. First 5 errors:")
        for msg in errors[:5]:
            print(f"  {msg}")
    
    # 샘플 시간 측정
    sample_size = min(10, len(files_to_calculate))
    sample_files = files_to_calculate[:sample_size]
    start_time = time.time()
    # 샘플 파일 처리...
    sample_time = time.time() - start_time
    estimated_total_time = sample_time / sample_size * len(files_to_calculate)
    print(f"Estimated total time: {estimated_total_time/3600:.1f} hours")
    
    # 결과 저장
    checkpoint_interval = 100
    for i, (file_path, energy, error) in enumerate(results):
        # 결과 처리...
        
        if (i+1) % checkpoint_interval == 0:
            temp_output = f"{os.path.splitext(args.output)[0]}_temp.csv"
            df.to_csv(temp_output, index=False)
            print(f"Checkpoint saved: {i+1}/{len(files_to_calculate)} completed")
        
        # 주기적 메모리 정리
        if (i+1) % 500 == 0:
            gc.collect()
    
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
