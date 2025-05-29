import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    matthews_corrcoef, balanced_accuracy_score, average_precision_score
)
import argparse
from utils.data_loader import load_and_preprocess_data

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Evaluate single feature classification performance')
    
    # 데이터 경로
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data CSV file')
    
    # DockQ 설정
    parser.add_argument('--dockq-threshold', type=float, default=0.23,
                       help='DockQ threshold for binary classification')
    
    # 평가할 feature 설정
    parser.add_argument('--feature-cols', type=str, nargs='+', required=True,
                       help='Column names of the features to evaluate')
    
    # 출력 설정
    parser.add_argument('--output-dir', type=str,
                       help='Output directory path (default: auto-generated)')
    
    return parser.parse_args()

def load_data(data_path, dockq_threshold):
    """데이터 로드 및 전처리"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # DockQ 기준으로 이진 분류 레이블 생성
    y = (df['DockQ'] >= dockq_threshold).astype(int)
    
    print(f"Data loaded: {df.shape[0]} samples")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return df, y

def evaluate_feature(df, y, feature_col, output_dir):
    """단일 feature 평가"""
    print(f"\n=== Evaluating {feature_col} ===")
    
    # feature 값 추출
    feature_values = df[feature_col].values
    
    # 성능 지표 계산
    metrics = {}
    
    # ROC curve 데이터 계산
    fpr, tpr, thresholds = roc_curve(y, feature_values)
    roc_auc = auc(fpr, tpr)
    metrics["roc_auc"] = roc_auc
    
    # PR curve 데이터 계산
    precision, recall, pr_thresholds = precision_recall_curve(y, feature_values)
    pr_ap = average_precision_score(y, feature_values)
    metrics["pr_ap"] = pr_ap
    
    # 최적 임계값 찾기
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    
    metrics["optimal_threshold"] = optimal_threshold
    metrics["optimal_f1"] = optimal_f1
    
    # 최적 임계값으로 예측
    y_pred = (feature_values >= optimal_threshold).astype(int)
    
    # 성능 지표 계산
    metrics.update({
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "mcc": matthews_corrcoef(y, y_pred),
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
        'true_label': y,
        'feature_value': feature_values,
        'pred_label': y_pred
    })
    predictions.to_csv(os.path.join(feature_dir, "predictions.csv"), index=False)
    
    # ROC curve 데이터 저장
    roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })
    roc_data.to_csv(os.path.join(feature_dir, "roc_curve_data.csv"), index=False)
    
    # PR curve 데이터 저장
    pr_data = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'thresholds': np.append(pr_thresholds, 1.0)
    })
    pr_data.to_csv(os.path.join(feature_dir, "pr_curve_data.csv"), index=False)
    
    # ROC curve 그리기
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {feature_col}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(feature_dir, "roc_curve.png"))
    plt.close()
    
    # PR curve 그리기
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR curve (AP = {pr_ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {feature_col}')
    plt.axhline(y=sum(y)/len(y), linestyle='--', color='r', label=f'Baseline (No Skill)')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(feature_dir, "pr_curve.png"))
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
        output_dir = f"feature_evaluation_{timestamp}"
        output_dir = os.path.abspath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 데이터 로드
    df, y = load_data(args.data_path, args.dockq_threshold)
    
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
