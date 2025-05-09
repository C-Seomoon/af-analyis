#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DockQ 상관관계 분석 스크립트

이 스크립트는 AlphaFold3로 계산된 docking 메트릭(pdockq, pdockq2, mpdockq)과 
실험적으로 측정된 실제 DockQ 값 사이의 상관관계를 분석합니다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import argparse

def load_data(calculated_metrics_path, experimental_dockq_path=None):
    """
    계산된 메트릭과 실험적 DockQ 값을 로드합니다.
    
    Parameters
    ----------
    calculated_metrics_path : str
        계산된 메트릭이 저장된 TSV 파일 경로
    experimental_dockq_path : str, optional
        실험적 DockQ 값이 저장된 파일 경로
        
    Returns
    -------
    pandas.DataFrame
        병합된 데이터프레임
    """
    # 계산된 메트릭 로드
    calc_df = pd.read_csv(calculated_metrics_path, sep='\t')
    
    # 실험적 DockQ 데이터가 제공된 경우 로드 및 병합
    if experimental_dockq_path and os.path.exists(experimental_dockq_path):
        exp_df = pd.read_csv(experimental_dockq_path, sep='\t')
        
        # 공통 키로 병합(예: PDB ID 또는 복합체 식별자)
        # 여기서는 'name' 열을 공통 키로 가정합니다 - 실제 데이터에 맞게 수정하세요
        merged_df = pd.merge(calc_df, exp_df, on='name', how='inner')
        print(f"데이터 병합 완료: {len(merged_df)} 개의 공통 항목 찾음")
        
        return merged_df
    else:
        print("실험적 DockQ 데이터가 제공되지 않았거나 찾을 수 없습니다. 계산된 메트릭만 사용합니다.")
        return calc_df

def analyze_correlations(df, calculated_cols, experimental_col='experimental_dockq'):
    """
    계산된 메트릭과 실험적 DockQ 값 사이의 상관관계를 분석합니다.
    
    Parameters
    ----------
    df : pandas.DataFrame
        분석할 데이터프레임
    calculated_cols : list
        계산된 메트릭 열 이름 목록
    experimental_col : str
        실험적 DockQ 값 열 이름
        
    Returns
    -------
    pandas.DataFrame
        상관관계 분석 결과
    """
    if experimental_col not in df.columns:
        print(f"경고: '{experimental_col}' 열이 데이터프레임에 없습니다. 상관관계 분석을 건너뜁니다.")
        return None
    
    results = []
    
    for col in calculated_cols:
        if col not in df.columns:
            print(f"경고: '{col}' 열이 데이터프레임에 없습니다. 이 메트릭을 건너뜁니다.")
            continue
            
        # 결측값 제거
        valid_data = df[[col, experimental_col]].dropna()
        
        if len(valid_data) < 5:
            print(f"경고: '{col}'에 대한 유효한 데이터가 너무 적습니다.")
            continue
            
        # 피어슨 상관계수
        pearson_r, pearson_p = stats.pearsonr(valid_data[col], valid_data[experimental_col])
        
        # 스피어만 상관계수
        spearman_r, spearman_p = stats.spearmanr(valid_data[col], valid_data[experimental_col])
        
        # 켄달 타우
        kendall_tau, kendall_p = stats.kendalltau(valid_data[col], valid_data[experimental_col])
        
        # RMSE & R²
        rmse = np.sqrt(mean_squared_error(valid_data[experimental_col], valid_data[col]))
        r2 = r2_score(valid_data[experimental_col], valid_data[col])
        
        results.append({
            'Metric': col,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
            'Kendall_tau': kendall_tau,
            'Kendall_p': kendall_p,
            'RMSE': rmse,
            'R2': r2,
            'Sample_size': len(valid_data)
        })
    
    return pd.DataFrame(results)

def analyze_chain_pair_correlations(df, experimental_col='experimental_dockq'):
    """
    체인 쌍별 메트릭에 대한 상관관계를 분석합니다.
    
    Parameters
    ----------
    df : pandas.DataFrame
        분석할 데이터프레임
    experimental_col : str
        실험적 DockQ 값 열 이름
        
    Returns
    -------
    dict
        각 메트릭 유형별 상관관계 결과 딕셔너리
    """
    metric_prefixes = ['pdockq_', 'pdockq2_', 'mpdockq_']
    results = {}
    
    for prefix in metric_prefixes:
        # 해당 접두사로 시작하는 모든 열 찾기
        chain_pair_cols = [col for col in df.columns if col.startswith(prefix)]
        
        if chain_pair_cols:
            print(f"'{prefix}' 접두사로 시작하는 체인 쌍 열 {len(chain_pair_cols)}개 발견")
            results[prefix] = analyze_correlations(df, chain_pair_cols, experimental_col)
        else:
            print(f"'{prefix}' 접두사로 시작하는 체인 쌍 열이 없습니다.")
    
    return results

def create_correlation_plots(df, calculated_cols, experimental_col='experimental_dockq', output_dir='plots'):
    """
    계산된 메트릭과 실험적 DockQ 값 사이의 상관관계를 시각화합니다.
    
    Parameters
    ----------
    df : pandas.DataFrame
        데이터프레임
    calculated_cols : list
        계산된 메트릭 열 이름 목록
    experimental_col : str
        실험적 DockQ 값 열 이름
    output_dir : str
        그래프 저장 디렉토리
    """
    if experimental_col not in df.columns:
        print(f"경고: '{experimental_col}' 열이 데이터프레임에 없습니다. 시각화를 건너뜁니다.")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 페어플롯 생성
    valid_cols = [col for col in calculated_cols if col in df.columns]
    if len(valid_cols) > 0:
        valid_cols.append(experimental_col)
        pairplot_data = df[valid_cols].copy()
        
        plt.figure(figsize=(12, 10))
        pairplot = sns.pairplot(pairplot_data, corner=True)
        plt.tight_layout()
        pairplot.savefig(os.path.join(output_dir, 'metrics_pairplot.png'), dpi=300)
        plt.close()
    
    # 각 메트릭별 산점도
    for col in calculated_cols:
        if col not in df.columns:
            continue
            
        valid_data = df[[col, experimental_col]].dropna()
        
        if len(valid_data) < 5:
            continue
            
        # 피어슨 상관계수
        pearson_r, pearson_p = stats.pearsonr(valid_data[col], valid_data[experimental_col])
        
        plt.figure(figsize=(8, 6))
        sns.regplot(x=col, y=experimental_col, data=valid_data, scatter_kws={'alpha':0.5})
        
        plt.title(f'{col} vs {experimental_col} (r={pearson_r:.3f}, p={pearson_p:.3e})')
        plt.xlabel(col)
        plt.ylabel(experimental_col)
        plt.grid(alpha=0.3)
        
        # 추세선 신뢰구간 추가
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'correlation_{col}_vs_{experimental_col}.png'), dpi=300)
        plt.close()
    
    # 상관관계 히트맵
    valid_cols = [col for col in calculated_cols if col in df.columns]
    if len(valid_cols) > 0:
        valid_cols.append(experimental_col)
        corr_data = df[valid_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', mask=mask, vmin=-1, vmax=1)
        
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
        plt.close()

def analyze_binary_classification(df, calculated_cols, experimental_col='experimental_dockq', threshold=0.5):
    """
    이진 분류 관점에서 메트릭을 평가합니다 (좋은 docking vs 나쁜 docking)
    
    Parameters
    ----------
    df : pandas.DataFrame
        데이터프레임
    calculated_cols : list
        계산된 메트릭 열 이름 목록
    experimental_col : str
        실험적 DockQ 값 열 이름
    threshold : float
        좋은 docking으로 분류하는 임계값
        
    Returns
    -------
    pandas.DataFrame
        이진 분류 성능 결과
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    if experimental_col not in df.columns:
        print(f"경고: '{experimental_col}' 열이 데이터프레임에 없습니다. 이진 분류 분석을 건너뜁니다.")
        return None
    
    results = []
    
    # 실험적 참값 이진화
    df[f'{experimental_col}_binary'] = (df[experimental_col] >= threshold).astype(int)
    
    for col in calculated_cols:
        if col not in df.columns:
            continue
            
        valid_data = df[[col, f'{experimental_col}_binary']].dropna()
        
        if len(valid_data) < 5:
            continue
        
        # 예측값 이진화
        valid_data[f'{col}_binary'] = (valid_data[col] >= threshold).astype(int)
        
        # 성능 지표 계산
        accuracy = accuracy_score(valid_data[f'{experimental_col}_binary'], valid_data[f'{col}_binary'])
        precision = precision_score(valid_data[f'{experimental_col}_binary'], valid_data[f'{col}_binary'])
        recall = recall_score(valid_data[f'{experimental_col}_binary'], valid_data[f'{col}_binary'])
        f1 = f1_score(valid_data[f'{experimental_col}_binary'], valid_data[f'{col}_binary'])
        
        # AUC는 연속값 필요
        auc = roc_auc_score(valid_data[f'{experimental_col}_binary'], valid_data[col])
        
        results.append({
            'Metric': col,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            'Sample_size': len(valid_data)
        })
    
    return pd.DataFrame(results)

def analyze_subgroup_correlations(df, calculated_cols, groupby_col, experimental_col='experimental_dockq'):
    """
    하위 그룹별로 상관관계를 분석합니다 (예: 단백질 유형, 복합체 크기 등)
    
    Parameters
    ----------
    df : pandas.DataFrame
        데이터프레임
    calculated_cols : list
        계산된 메트릭 열 이름 목록
    groupby_col : str
        그룹화할 열 이름
    experimental_col : str
        실험적 DockQ 값 열 이름
        
    Returns
    -------
    dict
        그룹별 상관관계 결과 딕셔너리
    """
    if groupby_col not in df.columns:
        print(f"경고: '{groupby_col}' 열이 데이터프레임에 없습니다. 하위 그룹 분석을 건너뜁니다.")
        return None
    
    results = {}
    groups = df[groupby_col].unique()
    
    for group in groups:
        group_df = df[df[groupby_col] == group]
        
        print(f"그룹 '{group}' 분석 중 (n={len(group_df)})")
        results[group] = analyze_correlations(group_df, calculated_cols, experimental_col)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='계산된 docking 메트릭과 실험적 DockQ 값 사이의 상관관계 분석')
    parser.add_argument('--calculated', required=True, help='계산된 메트릭이 포함된 TSV 파일 경로')
    parser.add_argument('--experimental', help='실험적 DockQ 값이 포함된 파일 경로')
    parser.add_argument('--exp_col', default='experimental_dockq', help='실험적 DockQ 값이 포함된 열 이름')
    parser.add_argument('--output', default='correlation_results', help='결과 저장 디렉토리')
    parser.add_argument('--group_by', help='하위 그룹 분석을 위한 열 이름')
    parser.add_argument('--threshold', type=float, default=0.5, help='이진 분류를 위한 임계값')
    
    args = parser.parse_args()
    
    # 데이터 로드
    #df = load_data(args.calculated, args.experimental)
    df=pd.read_csv(args.calculated, sep="\t")
    # 결과 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 모든 메트릭 열 식별
    dockq_metrics = [col for col in df.columns if col in ['pdockq', 'pdockq2', 'mpdockq']]
    
    # 결과 목록
    results = []
    
    # 1. 전체 상관관계 분석
    print("전체 상관관계 분석 중...")
    correlation_result = analyze_correlations(df, dockq_metrics, args.exp_col)
    if correlation_result is not None:
        results.append(("Overall_Correlations", correlation_result))
        correlation_result.to_csv(os.path.join(args.output, 'overall_correlations.csv'), index=False)
    
    # 2. 체인 쌍별 상관관계 분석
    print("체인 쌍별 상관관계 분석 중...")
    chain_pair_results = analyze_chain_pair_correlations(df, args.exp_col)
    for prefix, result in chain_pair_results.items():
        if result is not None:
            metric_type = prefix.rstrip('_')
            results.append((f"{metric_type}_Chain_Pair_Correlations", result))
            result.to_csv(os.path.join(args.output, f'{metric_type}_chain_pair_correlations.csv'), index=False)
    
    # 3. 이진 분류 성능 분석
    print("이진 분류 성능 분석 중...")
    binary_result = analyze_binary_classification(df, dockq_metrics, args.exp_col, args.threshold)
    if binary_result is not None:
        results.append(("Binary_Classification", binary_result))
        binary_result.to_csv(os.path.join(args.output, 'binary_classification.csv'), index=False)
    
    # 4. 하위 그룹별 분석
    if args.group_by:
        print(f"'{args.group_by}' 열을 기준으로 하위 그룹 분석 중...")
        subgroup_results = analyze_subgroup_correlations(df, dockq_metrics, args.group_by, args.exp_col)
        
        if subgroup_results:
            for group, result in subgroup_results.items():
                if result is not None:
                    results.append((f"Subgroup_{group}", result))
                    result.to_csv(os.path.join(args.output, f'subgroup_{group}_correlations.csv'), index=False)
    
    # 5. 시각화
    print("상관관계 시각화 중...")
    create_correlation_plots(df, dockq_metrics, args.exp_col, os.path.join(args.output, 'plots'))
    
    # 6. 결과 요약
    print("\n=== 분석 결과 요약 ===")
    for name, result in results:
        if result is not None and not result.empty:
            print(f"\n{name}:")
            if 'Pearson_r' in result.columns:
                # 상관관계 결과
                sorted_result = result.sort_values('Pearson_r', ascending=False)
                print(sorted_result[['Metric', 'Pearson_r', 'Spearman_r', 'RMSE', 'Sample_size']].head(5))
            elif 'AUC' in result.columns:
                # 이진 분류 결과
                sorted_result = result.sort_values('AUC', ascending=False)
                print(sorted_result[['Metric', 'Accuracy', 'AUC', 'F1', 'Sample_size']].head(5))
    
    print(f"\n모든 결과가 '{args.output}' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 