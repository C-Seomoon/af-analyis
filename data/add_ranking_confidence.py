#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

# 1) 경로 설정
MAIN_CSV = '/home/cseomoon/appl/af_analysis-0.1.4/data/final_data_with_rosetta_scaledRMSD_20250423.csv'
RESULTS_DIR = '/home/cseomoon/project/ABAG/AbNb_benchmark/AF3_results'
OUT_CSV = '/home/cseomoon/appl/af_analysis-0.1.4/data/final_data_with_ranking_scores.csv'

# 2) 메인 데이터프레임 읽기
df_main = pd.read_csv(MAIN_CSV)

# 3) 쿼리별 ranking_scores.csv를 모아서 하나의 DataFrame으로 만들기
ranking_dfs = []
for query in df_main['query'].unique():
    # 폴더 이름 = query 값
    ranking_path = os.path.join(RESULTS_DIR, str(query), 'ranking_scores.csv')
    if os.path.isfile(ranking_path):
        df_r = pd.read_csv(ranking_path, usecols=['seed', 'sample', 'ranking_score'])
        df_r['query'] = query
        ranking_dfs.append(df_r)
    else:
        # 없는 쿼리에 대해 경고만 뜨우고 넘어갑니다.
        print(f'Warning: file not found for query "{query}": {ranking_path}')

if ranking_dfs:
    df_ranking_all = pd.concat(ranking_dfs, ignore_index=True)
else:
    # 전부 누락된 경우 빈 DF 생성
    df_ranking_all = pd.DataFrame(columns=['query', 'seed', 'sample', 'ranking_score'])

# 4) 메인 DF와 병합 (left join)
df_merged = df_main.merge(
    df_ranking_all,
    on=['query', 'seed', 'sample'],
    how='left'
)

# 5) 결과 저장
df_merged.to_csv(OUT_CSV, index=False)
print(f'▶ merged dataframe saved to: {OUT_CSV}')