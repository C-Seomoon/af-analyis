import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Evaluate single feature regression performance')
    
    # 데이터 경로
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data CSV file')
    
    # 타겟 변수 설정
    parser.add_argument('--target-col', type=str, default='DockQ',
                       help='Target column name (default: DockQ)')
    
    # 평가할 feature 설정
    parser.add_argument('--feature-cols', type=str, nargs='+', required=True,
                       help='Column names of the features to evaluate')
    
    # 출력 설정
    parser.add_argument('--output-dir', type=str,
                       help='Output directory path (default: auto-generated)')
    
    return parser.parse_args()

def load_data(data_path, target_col):
    """데이터 로드 및 전처리"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # 타겟 변수 추출
    y = df[target_col].values
    
    print(f"Data loaded: {df.shape[0]} samples")
    print(f"Target variable statistics:")
    print(f"Mean: {np.mean(y):.3f}")
    print(f"Std: {np.std(y):.3f}")
    print(f"Min: {np.min(y):.3f}")
    print(f"Max: {np.max(y):.3f}")
    
    return df, y

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
    # 명령행 인자 파싱
    args = parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"feature_evaluation_regression_{timestamp}"
        output_dir = os.path.abspath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 데이터 로드
    df, y = load_data(args.data_path, args.target_col)
    
    # 모든 feature에 대한 결과를 저장할 DataFrame
    all_metrics = []
    
    # 각 feature 평가
    for feature_col in args.feature_cols:
        try:
            metrics = evaluate_feature(df, y, feature_col, output_dir)
            metrics['feature'] = feature_col
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error evaluating {feature_col}: {str(e)}")
    
    # 모든 feature의 성능 지표를 하나의 CSV 파일로 저장
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(output_dir, "all_features_metrics.csv"), index=False)
        print(f"\nAll metrics saved to {os.path.join(output_dir, 'all_features_metrics.csv')}")
    
    print(f"\nEvaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
