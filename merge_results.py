#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLURM 배열 작업으로 생성된 결과 파일들을 병합하는 스크립트
"""

import os
import pandas as pd
import glob
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='SLURM 배열 작업 결과 파일 병합')
    parser.add_argument('pattern', help='결과 파일 패턴 (예: results/output_task*.csv)')
    parser.add_argument('output', help='병합된 결과를 저장할 파일')
    parser.add_argument('--error_pattern', help='오류 파일 패턴 (기본값: 결과 파일 패턴 기반)')
    parser.add_argument('--merge_errors', action='store_true', help='오류 결과도 병합합니다')
    args = parser.parse_args()

    # 결과 파일 검색
    result_files = sorted(glob.glob(args.pattern))
    
    if not result_files:
        print(f"오류: '{args.pattern}' 패턴과 일치하는 파일을 찾을 수 없습니다.")
        return
    
    print(f"병합할 결과 파일 {len(result_files)}개 발견:")
    for f in result_files:
        print(f"  - {f}")
    
    # 결과 파일 병합
    dfs = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"파일 '{file}' 로드: {len(df)} 항목")
        except Exception as e:
            print(f"파일 '{file}' 로드 실패: {e}")
    
    if not dfs:
        print("병합할 데이터가 없습니다.")
        return
    
    # 결과 병합 및 저장
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(args.output, index=False)
    print(f"병합된 결과 {len(merged_df)}개 항목을 '{args.output}'에 저장했습니다.")
    
    # 오류 파일 병합 (옵션)
    if args.merge_errors:
        if args.error_pattern:
            error_pattern = args.error_pattern
        else:
            # 결과 파일 패턴에서 오류 파일 패턴 유추
            base_pattern = Path(args.pattern).stem.replace('*', '')
            dir_pattern = Path(args.pattern).parent
            error_pattern = str(dir_pattern / f"{base_pattern}*_errors.csv")
        
        error_files = sorted(glob.glob(error_pattern))
        
        if error_files:
            print(f"\n병합할 오류 파일 {len(error_files)}개 발견:")
            for f in error_files:
                print(f"  - {f}")
            
            error_dfs = []
            for file in error_files:
                try:
                    df = pd.read_csv(file)
                    error_dfs.append(df)
                except Exception as e:
                    print(f"오류 파일 '{file}' 로드 실패: {e}")
            
            if error_dfs:
                merged_errors = pd.concat(error_dfs, ignore_index=True)
                error_output = Path(args.output).stem + "_errors.csv"
                merged_errors.to_csv(error_output, index=False)
                print(f"병합된 오류 {len(merged_errors)}개를 '{error_output}'에 저장했습니다.")

if __name__ == "__main__":
    main() 