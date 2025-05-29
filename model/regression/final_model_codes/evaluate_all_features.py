import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = "/home/cseomoon/appl/af_analysis-0.1.4"
sys.path.append(project_root)

def get_features_to_evaluate(df, target_col):
    """평가할 feature 컬럼 목록 반환"""
    # 제외할 컬럼 목록
    default_features_to_drop = [
        'pdb', 'seed', 'sample', 'data_file', 'chain_iptm', 'chain_pair_iptm',
        'chain_pair_pae_min', 'chain_ptm', 'format', 'model_path', 'native_path',
        'Fnat', 'Fnonnat', 'rRMS', 'iRMS', 'LRMS', 'LIS'
    ]
    
    # 모든 컬럼에서 제외할 컬럼과 타겟 컬럼을 제외
    features_to_evaluate = [col for col in df.columns 
                          if col not in default_features_to_drop 
                          and col != target_col]
    
    return features_to_evaluate

def evaluate_feature(df, y, feature_col, output_dir):
    """단일 feature 평가"""
    print(f"\n=== Evaluating {feature_col} ===")
    
    # feature 값 추출
    feature_values = df[feature_col].values
    
    # 성능 지표 계산
    metrics = {}
    
    # 기본 회귀 메트릭 계산
    metrics.update({
        "r2": r2_score(y, feature_values),
        "mse": mean_squared_error(y, feature_values),
        "rmse": np.sqrt(mean_squared_error(y, feature_values)),
        "mae": mean_absolute_error(y, feature_values)
    })
    
    # feature별 하위 디렉토리 생성
    feature_dir = os.path.join(output_dir, feature_col)
    os.makedirs(feature_dir, exist_ok=True)
    
    # 성능 지표 저장
    with open(os.path.join(feature_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics: {metrics}")
    
    # 예측 결과 저장
    predictions = pd.DataFrame({
        'true_value': y,
        'feature_value': feature_values
    })
    predictions.to_csv(os.path.join(feature_dir, "predictions.csv"), index=False)
    
    # 산점도 그리기
    plt.figure(figsize=(10, 8))
    plt.scatter(y, feature_values, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect prediction')
    plt.xlabel('True DockQ')
    plt.ylabel(f'{feature_col}')
    plt.title(f'True vs {feature_col}')
    plt.legend()
    plt.savefig(os.path.join(feature_dir, "scatter_plot.png"))
    plt.close()
    
    # 잔차 플롯
    residuals = y - feature_values
    plt.figure(figsize=(10, 8))
    plt.scatter(feature_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(f'{feature_col}')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {feature_col}')
    plt.savefig(os.path.join(feature_dir, "residual_plot.png"))
    plt.close()
    
    return metrics

def main():
    # 데이터 경로 설정
    data_path = "/home/cseomoon/appl/af_analysis-0.1.4/model/regression/data/final_data_with_rosetta_scaledRMSD_20250423.csv"
    target_col = "DockQ"
    
    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"/home/cseomoon/appl/af_analysis-0.1.4/model/regression/feature_evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 데이터 로드
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    y = df[target_col].values
    
    print(f"Data loaded: {df.shape[0]} samples")
    print(f"Target variable statistics:")
    print(f"Mean: {np.mean(y):.3f}")
    print(f"Std: {np.std(y):.3f}")
    print(f"Min: {np.min(y):.3f}")
    print(f"Max: {np.max(y):.3f}")
    
    # 평가할 feature 목록 가져오기
    features_to_evaluate = get_features_to_evaluate(df, target_col)
    print(f"\nNumber of features to evaluate: {len(features_to_evaluate)}")
    
    # 모든 feature에 대한 결과를 저장할 리스트
    all_metrics = []
    
    # 각 feature 평가
    for feature_col in features_to_evaluate:
        try:
            metrics = evaluate_feature(df, y, feature_col, output_dir)
            metrics['feature'] = feature_col
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error evaluating {feature_col}: {str(e)}")
    
    # 모든 feature의 성능 지표를 DataFrame으로 변환
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # R² 기준으로 정렬하여 상위 10개 추출
        top_10_features = metrics_df.sort_values('r2', ascending=False).head(10)
        
        # 전체 결과 저장
        metrics_df.to_csv(os.path.join(output_dir, "all_features_metrics.csv"), index=False)
        
        # 상위 10개 결과 저장
        top_10_features.to_csv(os.path.join(output_dir, "top_10_features_by_r2.csv"), index=False)
        
        print("\n=== Top 10 Features by R² ===")
        print(top_10_features[['feature', 'r2', 'rmse', 'mae']].to_string(index=False))
        
        print(f"\nAll metrics saved to {os.path.join(output_dir, 'all_features_metrics.csv')}")
        print(f"Top 10 features saved to {os.path.join(output_dir, 'top_10_features_by_r2.csv')}")
    
    print(f"\nEvaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
