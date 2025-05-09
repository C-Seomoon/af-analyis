#!/usr/bin/env python3
# compare_models.py - 다양한 모델 유형의 결과 시각화 및 비교

import os
import json
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def find_latest_results(base_dir='.'):
    """가장 최근 결과 디렉토리 찾기"""
    linear_dirs = sorted(glob.glob(os.path.join(base_dir, "linear_classification_results_*")))
    tree_dirs = sorted(glob.glob(os.path.join(base_dir, "classification_nested_cv_results_*")))
    
    result_dirs = []
    
    if linear_dirs:
        result_dirs.append(("linear", linear_dirs[-1]))
        
    if tree_dirs:
        result_dirs.append(("tree", tree_dirs[-1]))
    
    return result_dirs

def load_curve_data(result_dirs):
    """각 모델 결과 디렉토리에서 커브 데이터 로드"""
    all_curves = {
        'roc': [],
        'pr': []
    }
    
    for model_type, result_dir in result_dirs:
        curve_data_dir = os.path.join(result_dir, 'curve_data')
        
        if not os.path.exists(curve_data_dir):
            print(f"경고: {result_dir}에 curve_data 디렉토리가 없습니다.")
            continue
        
        for data_file in glob.glob(os.path.join(curve_data_dir, "*_curve_data.json")):
            with open(data_file, 'r') as f:
                data = json.load(f)
                
            model_name = data['model_name']
            
            # ROC 커브 데이터
            all_curves['roc'].append({
                'model_name': model_name,
                'model_type': model_type,
                'fpr_grid': data['fpr_grid'],
                'mean_tpr': data['mean_tpr'],
                'mean_auc': data['mean_auc'],
                'std_auc': data['std_auc']
            })
            
            # PR 커브 데이터
            all_curves['pr'].append({
                'model_name': model_name,
                'model_type': model_type,
                'recall_grid': data['recall_grid'],
                'mean_precision': data['mean_precision'],
                'mean_ap': data['mean_ap'],
                'std_ap': data['std_ap']
            })
    
    return all_curves

def load_metrics_data(result_dirs):
    """각 모델 결과 디렉토리에서 성능 지표 데이터 로드"""
    all_metrics = []
    
    for model_type, result_dir in result_dirs:
        # metrics_summary.csv 찾기
        summary_file = os.path.join(result_dir, 'metrics_summary.csv')
        if os.path.exists(summary_file):
            metrics_df = pd.read_csv(summary_file)
            metrics_df['model_type'] = model_type
            all_metrics.append(metrics_df)
            continue
        
        # 또는 폴드별 지표 찾기
        all_fold_metrics = []
        
        for fold_dir in glob.glob(os.path.join(result_dir, 'fold_*')):
            fold_idx = int(os.path.basename(fold_dir).split('_')[1])
            
            for model_dir in os.listdir(fold_dir):
                metrics_file = os.path.join(fold_dir, model_dir, 'metrics.csv')
                
                if os.path.exists(metrics_file):
                    fold_metrics = pd.read_csv(metrics_file)
                    fold_metrics['model'] = model_dir
                    fold_metrics['fold'] = fold_idx
                    all_fold_metrics.append(fold_metrics)
        
        if all_fold_metrics:
            fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
            
            # 평균 및 표준편차 계산
            metrics_summary = []
            
            for model in fold_metrics_df['model'].unique():
                model_data = fold_metrics_df[fold_metrics_df['model'] == model]
                
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc', 'pr_auc']:
                    if metric in model_data.columns:
                        metrics_summary.append({
                            'Model': model,
                            'Metric': metric,
                            'Mean': model_data[metric].mean(),
                            'Std': model_data[metric].std(),
                            'model_type': model_type
                        })
            
            metrics_df = pd.DataFrame(metrics_summary)
            all_metrics.append(metrics_df)
    
    if all_metrics:
        return pd.concat(all_metrics, ignore_index=True)
    else:
        return None

def create_roc_curves(curve_data, output_dir):
    """모든 모델의 ROC 커브 플롯 생성"""
    plt.figure(figsize=(12, 8))
    
    # 모델 타입별 색상 설정
    colors = {
        'tree': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # 파란색, 주황색, 녹색
        'linear': ['#d62728', '#9467bd', '#8c564b']  # 빨간색, 보라색, 갈색
    }
    
    for i, model_data in enumerate(curve_data['roc']):
        model_name = model_data['model_name']
        model_type = model_data['model_type']
        color_idx = i % len(colors[model_type])
        
        plt.plot(
            model_data['fpr_grid'],
            model_data['mean_tpr'],
            label=f"{model_name} ({model_type}, AUC={model_data['mean_auc']:.3f}±{model_data['std_auc']:.3f})",
            color=colors[model_type][color_idx],
            lw=2
        )
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for All Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_roc_curves.png'), dpi=300)
    plt.close()

def create_pr_curves(curve_data, output_dir):
    """모든 모델의 PR 커브 플롯 생성"""
    plt.figure(figsize=(12, 8))
    
    # 모델 타입별 색상 설정
    colors = {
        'tree': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # 파란색, 주황색, 녹색
        'linear': ['#d62728', '#9467bd', '#8c564b']  # 빨간색, 보라색, 갈색
    }
    
    for i, model_data in enumerate(curve_data['pr']):
        model_name = model_data['model_name']
        model_type = model_data['model_type']
        color_idx = i % len(colors[model_type])
        
        plt.plot(
            model_data['recall_grid'],
            model_data['mean_precision'],
            label=f"{model_name} ({model_type}, AP={model_data['mean_ap']:.3f}±{model_data['std_ap']:.3f})",
            color=colors[model_type][color_idx],
            lw=2
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves for All Models', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_pr_curves.png'), dpi=300)
    plt.close()

def create_metrics_comparison(metrics_data, output_dir):
    """모든 모델의 성능 지표 비교 그래프 생성"""
    if metrics_data is None:
        print("성능 지표 데이터를 로드할 수 없습니다.")
        return
    
    # 주요 지표만 선택
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    metrics_subset = metrics_data[metrics_data['Metric'].isin(key_metrics)]
    
    # 모델별 평균 성능 그래프
    for metric in key_metrics:
        metric_data = metrics_subset[metrics_subset['Metric'] == metric]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Model', 
            y='Mean', 
            hue='model_type',
            data=metric_data,
            palette=['#1f77b4', '#d62728'],  # 트리: 파란색, 선형: 빨간색
            errorbar=('ci', 95)
        )
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(f'{metric.upper()}', fontsize=12)
        plt.title(f'{metric.upper()} Comparison Across Models', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Model Type')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'combined_{metric}_comparison.png'), dpi=300)
        plt.close()
    
    # 모든 지표를 하나의 그래프로 통합 - 첫 번째 모델만
    for model_type in metrics_subset['model_type'].unique():
        type_data = metrics_subset[metrics_subset['model_type'] == model_type]
        models = type_data['Model'].unique()
        
        for model in models:
            model_data = type_data[type_data['Model'] == model]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='Metric',
                y='Mean',
                data=model_data,
                color='#1f77b4' if model_type == 'tree' else '#d62728',
                errorbar=('ci', 95)
            )
            
            plt.xlabel('Metric', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title(f'{model} ({model_type}) - Performance Metrics', fontsize=14)
            plt.ylim([0, 1])
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model}_{model_type}_metrics.png'), dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='모델 성능 비교 및 시각화')
    parser.add_argument('--result_dirs', nargs='+', help='결과 디렉토리 경로 (지정하지 않으면 자동 감지)')
    parser.add_argument('--output_dir', type=str, default=None, help='출력 디렉토리 (기본값: combined_results_YYYYMMDD_HHMMSS)')
    
    args = parser.parse_args()
    
    # 결과 디렉토리 설정
    if args.result_dirs:
        # 사용자 지정 디렉토리
        result_dirs = []
        for dir_path in args.result_dirs:
            if "linear_classification" in dir_path:
                result_dirs.append(("linear", dir_path))
            elif "classification_nested_cv" in dir_path:
                result_dirs.append(("tree", dir_path))
            else:
                print(f"경고: {dir_path}의 모델 유형을 결정할 수 없습니다.")
    else:
        # 자동 감지
        result_dirs = find_latest_results()
    
    if not result_dirs:
        print("결과 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"분석할 결과 디렉토리: {[d[1] for d in result_dirs]}")
    
    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'combined_model_comparison_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리: {output_dir}")
    
    # 커브 데이터 로드
    curve_data = load_curve_data(result_dirs)
    
    # 성능 지표 데이터 로드
    metrics_data = load_metrics_data(result_dirs)
    
    # 시각화 생성
    if curve_data['roc']:
        create_roc_curves(curve_data, output_dir)
        print("ROC 커브 플롯 생성 완료")
    
    if curve_data['pr']:
        create_pr_curves(curve_data, output_dir)
        print("PR 커브 플롯 생성 완료")
    
    if metrics_data is not None:
        create_metrics_comparison(metrics_data, output_dir)
        print("성능 지표 비교 플롯 생성 완료")
    
    print(f"모든 분석이 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
