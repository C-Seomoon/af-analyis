import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import json

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Visualize feature comparison results')
    
    parser.add_argument('--base-dir', type=str, 
                       default='/home/cseomoon/appl/af_analysis-0.1.4/model/classification/final_models',
                       help='Base directory containing all results')
    
    parser.add_argument('--comparison-dir', type=str,
                       help='Specific comparison directory (default: most recent)')
    
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for visualizations (default: base_dir/visualizations)')
    
    return parser.parse_args()

def get_latest_comparison_dir(base_dir):
    """가장 최근의 comparison 디렉토리 찾기"""
    comparison_dirs = [d for d in Path(base_dir).glob('comparison_*') if d.is_dir()]
    if not comparison_dirs:
        raise ValueError("No comparison directories found")
    return max(comparison_dirs, key=lambda x: x.stat().st_mtime)

def load_feature_data(base_dir, comparison_dir):
    """각 feature의 결과 데이터 로드"""
    base_dir = Path(base_dir)
    comparison_dir = Path(comparison_dir)
    
    # 모델 결과 로드
    model_data = {}
    for model in ['logistic', 'rf']:
        model_dir = comparison_dir / model
        if model_dir.exists():
            try:
                # ROC curve 데이터
                roc_data = pd.read_csv(model_dir / "roc_curve_data.csv")
                # PR curve 데이터
                pr_data = pd.read_csv(model_dir / "pr_curve_data.csv")
                # 메트릭스
                with open(model_dir / "metrics.json", 'r') as f:
                    metrics = json.load(f)
                
                # 클래스 비율 계산 (positive class의 비율)
                if 'predictions.csv' in os.listdir(model_dir):
                    preds = pd.read_csv(model_dir / "predictions.csv")
                    class_ratio = preds['true_label'].mean()
                    metrics['class_ratio'] = class_ratio
                
                model_data[model] = {
                    'roc': roc_data,
                    'pr': pr_data,
                    'metrics': metrics
                }
            except Exception as e:
                print(f"Error loading data for model {model}: {str(e)}")
    
    # 단일 feature 결과 로드
    feature_data = {}
    for feature_dir in base_dir.glob('*'):
        # comparison 디렉토리와 .ipynb_checkpoints 제외
        if (feature_dir.is_dir() and 
            not feature_dir.name.startswith('comparison_') and 
            not feature_dir.name.startswith('.') and
            feature_dir.name != 'visualizations'):
            
            feature = feature_dir.name
            try:
                # ROC curve 데이터
                roc_data = pd.read_csv(feature_dir / "roc_curve_data.csv")
                # PR curve 데이터
                pr_data = pd.read_csv(feature_dir / "pr_curve_data.csv")
                # 메트릭스
                with open(feature_dir / "metrics.json", 'r') as f:
                    metrics = json.load(f)
                
                # 클래스 비율 계산 (positive class의 비율)
                if 'predictions.csv' in os.listdir(feature_dir):
                    preds = pd.read_csv(feature_dir / "predictions.csv")
                    class_ratio = preds['true_label'].mean()
                    metrics['class_ratio'] = class_ratio
                
                feature_data[feature] = {
                    'roc': roc_data,
                    'pr': pr_data,
                    'metrics': metrics
                }
            except Exception as e:
                print(f"Error loading data for feature {feature}: {str(e)}")
    
    return model_data, feature_data

def plot_roc_curves(model_data, feature_data, output_dir):
    """모든 feature와 모델의 ROC curve를 하나의 그래프에 그리기"""
    plt.figure(figsize=(12, 8))
    
    # 색상 팔레트 설정
    n_colors = len(model_data) + len(feature_data)
    colors = sns.color_palette("husl", n_colors)
    
    # 모델 ROC curves
    for (model, data), color in zip(model_data.items(), colors):
        try:
            roc_data = data['roc']
            auc_score = data['metrics'].get('roc_auc', 0.0)
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'{model} (AUC = {auc_score:.3f})',
                    color=color, linestyle='--')
        except Exception as e:
            print(f"Error plotting ROC curve for model {model}: {str(e)}")
    
    # feature ROC curves
    for (feature, data), color in zip(feature_data.items(), colors[len(model_data):]):
        try:
            roc_data = data['roc']
            auc_score = data['metrics'].get('roc_auc', 0.0)
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'{feature} (AUC = {auc_score:.3f})',
                    color=color)
        except Exception as e:
            print(f"Error plotting ROC curve for feature {feature}: {str(e)}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'roc_curves_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_pr_curves(model_data, feature_data, output_dir):
    """모든 feature와 모델의 PR curve를 하나의 그래프에 그리기"""
    plt.figure(figsize=(12, 8))
    
    # 색상 팔레트 설정
    n_colors = len(model_data) + len(feature_data)
    colors = sns.color_palette("husl", n_colors)
    
    # 모델 PR curves
    for (model, data), color in zip(model_data.items(), colors):
        try:
            pr_data = data['pr']
            ap_score = data['metrics'].get('pr_ap', 0.0)
            plt.plot(pr_data['recall'], pr_data['precision'], 
                    label=f'{model} (AP = {ap_score:.3f})',
                    color=color, linestyle='--')
        except Exception as e:
            print(f"Error plotting PR curve for model {model}: {str(e)}")
    
    # feature PR curves
    for (feature, data), color in zip(feature_data.items(), colors[len(model_data):]):
        try:
            pr_data = data['pr']
            ap_score = data['metrics'].get('pr_ap', 0.0)
            plt.plot(pr_data['recall'], pr_data['precision'], 
                    label=f'{feature} (AP = {ap_score:.3f})',
                    color=color)
        except Exception as e:
            print(f"Error plotting PR curve for feature {feature}: {str(e)}")
    
    # Baseline 추가 (첫 번째 모델이나 feature의 데이터에서 클래스 비율 계산)
    try:
        first_data = next(iter(model_data.values())) if model_data else next(iter(feature_data.values()))
        if 'metrics' in first_data and 'class_ratio' in first_data['metrics']:
            class_ratio = first_data['metrics']['class_ratio']
            plt.axhline(y=class_ratio, linestyle='--', color='gray', 
                       label=f'Baseline (No Skill = {class_ratio:.3f})')
    except Exception as e:
        print(f"Error adding baseline: {str(e)}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'pr_curves_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_metrics_summary(model_data, feature_data, output_dir):
    """성능 지표 요약 테이블 생성"""
    # 모든 메트릭스 수집
    all_metrics = []
    
    # 모델 메트릭스
    for model, data in model_data.items():
        try:
            metrics = data['metrics'].copy()
            metrics['name'] = model
            metrics['type'] = 'model'
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing metrics for model {model}: {str(e)}")
    
    # feature 메트릭스
    for feature, data in feature_data.items():
        try:
            metrics = data['metrics'].copy()
            metrics['name'] = feature
            metrics['type'] = 'feature'
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing metrics for feature {feature}: {str(e)}")
    
    if not all_metrics:
        print("No metrics data available")
        return None
    
    # DataFrame 생성
    metrics_df = pd.DataFrame(all_metrics)
    
    # 중요 지표만 선택 (존재하는 컬럼만)
    important_metrics = ['name', 'type', 'roc_auc', 'pr_ap', 'accuracy', 
                        'precision', 'recall', 'f1', 'balanced_accuracy', 'mcc']
    available_metrics = [m for m in important_metrics if m in metrics_df.columns]
    
    summary_df = metrics_df[available_metrics].copy()
    
    # 소수점 3자리까지 반올림
    summary_df = summary_df.round(3)
    
    # CSV로 저장
    summary_df.to_csv(output_dir / 'metrics_summary.csv', index=False)
    
    # LaTeX 테이블로 저장
    latex_table = summary_df.to_latex(index=False, float_format=lambda x: f'{x:.3f}')
    with open(output_dir / 'metrics_summary.tex', 'w') as f:
        f.write(latex_table)
    
    return summary_df

def plot_metrics_heatmap(metrics_df, output_dir):
    """성능 지표 히트맵 생성"""
    # 중요 지표만 선택
    important_metrics = ['roc_auc', 'pr_ap', 'accuracy', 
                        'precision', 'recall', 'f1', 'balanced_accuracy', 'mcc']
    
    # 히트맵 데이터 준비
    heatmap_data = metrics_df.set_index('name')[important_metrics]
    
    # 히트맵 그리기
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Performance Metrics Heatmap')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'metrics_heatmap.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # 명령행 인자 파싱
    args = parse_args()
    
    # 출력 디렉토리 설정
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.base_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # comparison 디렉토리 설정
    comparison_dir = Path(args.comparison_dir) if args.comparison_dir else get_latest_comparison_dir(args.base_dir)
    
    print(f"Loading results from {args.base_dir}")
    print(f"Using comparison directory: {comparison_dir}")
    print(f"Saving visualizations to {output_dir}")
    
    # 데이터 로드
    model_data, feature_data = load_feature_data(args.base_dir, comparison_dir)
    
    # ROC curves 비교 그래프 생성
    print("Generating ROC curves comparison...")
    plot_roc_curves(model_data, feature_data, output_dir)
    
    # PR curves 비교 그래프 생성
    print("Generating PR curves comparison...")
    plot_pr_curves(model_data, feature_data, output_dir)
    
    # 성능 지표 요약 생성
    print("Creating metrics summary...")
    summary_df = create_metrics_summary(model_data, feature_data, output_dir)
    
    # 성능 지표 히트맵 생성
    print("Generating metrics heatmap...")
    plot_metrics_heatmap(summary_df, output_dir)
    
    print(f"\nVisualization completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
