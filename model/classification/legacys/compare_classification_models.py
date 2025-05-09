import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# 현재 시간을 파일명에 사용
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 결과 디렉토리 생성
output_dir = f'classification_model_comparison_results_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"결과 디렉토리 생성: {output_dir}")

# 모델 디렉토리 경로 설정 - 실제 경로로 수정 필요
model_dirs = {
    'RF': 'RF_classifier_output_20250422_170902',
    'LightGBM': 'lightgbm_binary_output_20250422_174715',
    'XGBoost': 'xgboost_binary_native_api_20250422_175057'
}

# 각 모델의 결과 로드 함수
def load_model_results(model_name, model_dir):
    """특정 모델의 결과 데이터를 로드"""
    results = {}
    
    # 비교 데이터 디렉토리 경로
    comparison_dir = os.path.join(model_dir, 'comparison_data')
    
    try:
        # 1. 예측 결과 로드
        pred_file = os.path.join(comparison_dir, f'{model_name}_predictions.csv')
        if os.path.exists(pred_file):
            results['predictions'] = pd.read_csv(pred_file)
            print(f"{model_name} 예측 결과 로드 완료: {pred_file}")
        else:
            print(f"경고: {pred_file} 파일이 존재하지 않습니다.")
        
        # 2. 성능 지표 로드
        metrics_file = os.path.join(comparison_dir, f'{model_name}_metrics.csv')
        if os.path.exists(metrics_file):
            results['metrics'] = pd.read_csv(metrics_file)
            print(f"{model_name} 성능 지표 로드 완료: {metrics_file}")
        else:
            print(f"경고: {metrics_file} 파일이 존재하지 않습니다.")
        
        # 3. 특성 중요도 로드
        imp_file = os.path.join(comparison_dir, f'{model_name}_feature_importance.csv')
        if os.path.exists(imp_file):
            results['feature_importance'] = pd.read_csv(imp_file)
            print(f"{model_name} 특성 중요도 로드 완료: {imp_file}")
        else:
            print(f"경고: {imp_file} 파일이 존재하지 않습니다.")
        
        # 4. 혼동 행렬 로드
        cm_file = os.path.join(comparison_dir, f'{model_name}_confusion_matrix.csv')
        if os.path.exists(cm_file):
            results['confusion_matrix'] = pd.read_csv(cm_file)
            print(f"{model_name} 혼동 행렬 로드 완료: {cm_file}")
        else:
            print(f"경고: {cm_file} 파일이 존재하지 않습니다.")
        
        # 5. 교차 검증 점수 로드
        cv_file = os.path.join(comparison_dir, f'{model_name}_cv_scores.csv')
        if os.path.exists(cv_file):
            results['cv_scores'] = pd.read_csv(cv_file)
            print(f"{model_name} 교차 검증 점수 로드 완료: {cv_file}")
        else:
            print(f"경고: {cv_file} 파일이 존재하지 않습니다.")
        
        # 6. 하이퍼파라미터 로드
        hp_file = os.path.join(comparison_dir, f'{model_name}_hyperparams.json')
        if os.path.exists(hp_file):
            with open(hp_file, 'r') as f:
                results['hyperparams'] = json.load(f)
            print(f"{model_name} 하이퍼파라미터 로드 완료: {hp_file}")
        else:
            print(f"경고: {hp_file} 파일이 존재하지 않습니다.")
            
        return results
    
    except Exception as e:
        print(f"{model_name} 결과 로드 중 오류 발생: {str(e)}")
        return results

# 모델 결과 로드
model_results = {}
for model_name, model_dir in model_dirs.items():
    print(f"\n{model_name} 모델 결과 로드 중...")
    model_results[model_name] = load_model_results(model_name, model_dir)

# 성능 지표 비교 표 생성
def create_metrics_comparison(model_results, output_dir, timestamp):
    """각 모델의 성능 지표를 비교하는 표 생성"""
    metrics_rows = []
    
    # 추출할 지표 정의
    metrics_of_interest = [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'roc_auc', 'pr_auc', 'mcc', 'log_loss',
        'training_time'
    ]
    
    # 각 모델의 지표 추출
    for model_name, results in model_results.items():
        if 'metrics' in results:
            metrics_df = results['metrics']
            
            # 모델 이름과 지표를 포함하는 행 생성
            metrics_row = {'model': model_name}
            
            # 관심 있는 지표만 추출
            for metric in metrics_of_interest:
                if metric in metrics_df.columns:
                    metrics_row[metric] = metrics_df[metric].values[0]
                else:
                    metrics_row[metric] = np.nan
                    
            metrics_rows.append(metrics_row)
    
    # 성능 지표 데이터프레임 생성
    if metrics_rows:
        metrics_comparison = pd.DataFrame(metrics_rows)
        
        # 훈련 시간을 시/분/초 형식으로 변환
        if 'training_time' in metrics_comparison.columns:
            metrics_comparison['training_time'] = metrics_comparison['training_time'].apply(
                lambda x: f"{int(x // 3600)}h {int((x % 3600) // 60)}m {int(x % 60)}s" if not pd.isna(x) else np.nan
            )
        
        # 결과 출력 및 저장
        print("\n모델 성능 지표 비교:")
        print(metrics_comparison)
        
        # CSV 파일로 저장
        output_file = os.path.join(output_dir, f'metrics_comparison_{timestamp}.csv')
        metrics_comparison.to_csv(output_file, index=False)
        print(f"성능 지표 비교 표 저장: {output_file}")
        
        # 읽기 쉬운 HTML 테이블로도 저장
        html_file = os.path.join(output_dir, f'metrics_comparison_{timestamp}.html')
        metrics_comparison.to_html(html_file, index=False)
        print(f"성능 지표 비교 HTML 저장: {html_file}")
        
        return metrics_comparison
    else:
        print("성능 지표를 찾을 수 없습니다.")
        return None

# 성능 지표 통합 바 차트 생성 (변경됨 - 하나의 그래프에 모든 지표 표시)
def plot_metrics_comparison(metrics_comparison, output_dir, timestamp):
    """성능 지표 비교 통합 바 차트 생성"""
    if metrics_comparison is None or len(metrics_comparison) == 0:
        print("성능 지표 데이터가 없어 바 차트를 생성할 수 없습니다.")
        return
    
    # 시각화할 지표 선택 (training_time 제외)
    plot_metrics = [col for col in metrics_comparison.columns 
                    if col not in ['model', 'training_time']]
    
    # 모델 이름 목록
    model_names = metrics_comparison['model'].tolist()
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 바 너비와 위치 계산
    n_metrics = len(plot_metrics)
    width = 0.8 / n_metrics
    indices = np.arange(len(model_names))
    
    # 각 지표별로 바 그리기
    for i, metric in enumerate(plot_metrics):
        # 누락된 값 처리
        values = []
        for model in model_names:
            val = metrics_comparison.loc[metrics_comparison['model'] == model, metric].values[0]
            values.append(0 if pd.isna(val) else val)
        
        # 바 위치 계산
        pos = indices + (i - n_metrics/2 + 0.5) * width
        
        # 바 그리기
        bars = ax.bar(pos, values, width, label=metric.upper())
        
        # 바 위에 값 표시
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', rotation=90, fontsize=8)
    
    # 그래프 설정
    ax.set_xticks(indices)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.2)  # 값이 1을 넘지 않도록 설정
    ax.set_title('All Metrics Comparison Across Models', fontsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'combined_metrics_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"통합 성능 지표 비교 차트 저장: {output_file}")
    
    # 추가: 지표별 모델 비교 (열 방향 비교)
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(plot_metrics[:8]):  # 최대 8개 지표
        if i < len(axes):
            # 지표별 모델 비교 바 차트
            values = []
            for model in model_names:
                val = metrics_comparison.loc[metrics_comparison['model'] == model, metric].values[0]
                values.append(0 if pd.isna(val) else val)
            
            bars = axes[i].bar(model_names, values, color=plt.cm.tab10(i))
            
            # 바 위에 값 표시
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            
            axes[i].set_title(metric.upper())
            axes[i].set_ylim(0, min(1.0, max(values) * 1.15) if values else 1.0)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'metrics_by_category_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"지표별 모델 비교 차트 저장: {output_file}")

# ROC 곡선 비교 (기존 함수 유지)
def plot_roc_comparison(model_results, output_dir, timestamp):
    """모델들의 ROC 곡선 비교"""
    plt.figure(figsize=(10, 8))
    
    for model_name, results in model_results.items():
        if 'predictions' in results:
            preds_df = results['predictions']
            
            if 'actual' in preds_df.columns and 'probability' in preds_df.columns:
                # ROC 곡선 계산
                fpr, tpr, _ = roc_curve(preds_df['actual'], preds_df['probability'])
                roc_auc = auc(fpr, tpr)
                
                # ROC 곡선 그리기
                plt.plot(fpr, tpr, lw=2, 
                         label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # 기준선 추가
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'roc_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"ROC 곡선 비교 저장: {output_file}")

# PR 곡선 비교 (기존 함수 유지)
def plot_pr_comparison(model_results, output_dir, timestamp):
    """모델들의 PR 곡선 비교"""
    plt.figure(figsize=(10, 8))
    
    baseline = None
    
    for model_name, results in model_results.items():
        if 'predictions' in results:
            preds_df = results['predictions']
            
            if 'actual' in preds_df.columns and 'probability' in preds_df.columns:
                # PR 곡선 계산
                precision, recall, _ = precision_recall_curve(preds_df['actual'], preds_df['probability'])
                pr_auc = average_precision_score(preds_df['actual'], preds_df['probability'])
                
                # PR 곡선 그리기
                plt.plot(recall, precision, lw=2, 
                         label=f'{model_name} (AP = {pr_auc:.4f})')
                
                # 첫 번째 모델의 기준선을 사용
                if baseline is None:
                    baseline = preds_df['actual'].mean()
    
    # 기준선 추가 (클래스 1의 비율)
    if baseline is not None:
        plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', 
                   label=f'Baseline (Class 1 ratio = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve Comparison', fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'pr_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"PR 곡선 비교 저장: {output_file}")

# 새로운 함수: Calibration 곡선 비교 (추가됨)
def plot_calibration_comparison(model_results, output_dir, timestamp):
    """모델들의 Calibration 곡선 비교"""
    plt.figure(figsize=(10, 8))
    
    for model_name, results in model_results.items():
        if 'predictions' in results:
            preds_df = results['predictions']
            
            if 'actual' in preds_df.columns and 'probability' in preds_df.columns:
                # Calibration 곡선 계산
                prob_true, prob_pred = calibration_curve(
                    preds_df['actual'], preds_df['probability'], n_bins=10)
                
                # Calibration 곡선 그리기
                plt.plot(prob_pred, prob_true, marker='s', markersize=5, 
                        label=f'{model_name}')
    
    # 이상적인 보정선 추가
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve Comparison', fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'calibration_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Calibration 곡선 비교 저장: {output_file}")

# 예측 확률 분포 비교
def plot_probability_distribution_comparison(model_results, output_dir, timestamp):
    """각 모델의 예측 확률 분포 비교"""
    plt.figure(figsize=(12, 8))
    
    for model_name, results in model_results.items():
        if 'predictions' in results:
            preds_df = results['predictions']
            
            if 'probability' in preds_df.columns:
                # 예측 확률 분포 플롯
                sns.kdeplot(preds_df['probability'], 
                           label=f'{model_name}', 
                           fill=True, alpha=0.3)
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Predicted Probability of Class 1', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Prediction Probability Distribution Comparison', fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'probability_distribution_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"예측 확률 분포 비교 저장: {output_file}")
    
    # 클래스별 확률 분포 비교 (클래스 0과 1 구분)
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, results) in enumerate(model_results.items()):
        if 'predictions' in results:
            preds_df = results['predictions']
            
            if 'actual' in preds_df.columns and 'probability' in preds_df.columns:
                # 서브플롯 생성
                plt.subplot(len(model_results), 1, i+1)
                
                # 클래스 0의 확률 분포
                class0_probs = preds_df.loc[preds_df['actual'] == 0, 'probability']
                if len(class0_probs) > 0:
                    sns.histplot(class0_probs, bins=30, kde=True, 
                                color='blue', alpha=0.6, label='Class 0')
                
                # 클래스 1의 확률 분포
                class1_probs = preds_df.loc[preds_df['actual'] == 1, 'probability']
                if len(class1_probs) > 0:
                    sns.histplot(class1_probs, bins=30, kde=True, 
                                color='red', alpha=0.6, label='Class 1')
                
                plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
                plt.title(f'{model_name} - Prediction Probability by Actual Class')
                plt.xlabel('Predicted Probability of Class 1')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'probability_by_class_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"클래스별 예측 확률 분포 비교 저장: {output_file}")

# 특성 중요도 비교
def plot_feature_importance_comparison(model_results, output_dir, timestamp, top_n=20):
    """각 모델의 상위 특성 중요도 비교"""
    # 각 모델에서 상위 N개 특성 추출
    all_top_features = set()
    model_top_features = {}
    
    for model_name, results in model_results.items():
        if 'feature_importance' in results:
            imp_df = results['feature_importance']
            if 'feature' in imp_df.columns and 'importance' in imp_df.columns:
                # 내림차순 정렬 후 상위 N개 특성 추출
                top_features = imp_df.sort_values('importance', ascending=False).head(top_n)
                model_top_features[model_name] = dict(zip(top_features['feature'], top_features['importance']))
                all_top_features.update(top_features['feature'])
    
    if not model_top_features:
        print("특성 중요도 데이터가 없어 비교 차트를 생성할 수 없습니다.")
        return
    
    # 모든 모델에 대한 특성 중요도 통합
    all_top_features = list(all_top_features)
    combined_data = []
    
    for feature in all_top_features:
        feature_data = {'feature': feature}
        for model_name, top_features in model_top_features.items():
            feature_data[model_name] = top_features.get(feature, 0)
        combined_data.append(feature_data)
    
    # 데이터프레임 생성
    combined_df = pd.DataFrame(combined_data)
    
    # 모든 모델의 중요도 총합으로 정렬
    model_cols = list(model_top_features.keys())
    combined_df['total_importance'] = combined_df[model_cols].sum(axis=1)
    combined_df = combined_df.sort_values('total_importance', ascending=False).head(top_n)
    combined_df = combined_df.drop(columns=['total_importance'])
    
    # 그래프 준비
    fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
    
    # 각 모델별 특성 중요도 플롯
    ind = np.arange(len(combined_df))
    width = 0.8 / len(model_cols)
    
    for i, model_name in enumerate(model_cols):
        offset = (i - len(model_cols) / 2 + 0.5) * width
        ax.barh(ind + offset, combined_df[model_name], width, label=model_name)
    
    ax.set_yticks(ind)
    ax.set_yticklabels(combined_df['feature'])
    ax.invert_yaxis()  # 위에서부터 표시
    
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importance Comparison')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'feature_importance_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"특성 중요도 비교 저장: {output_file}")
    
    # 각 모델별 상위 특성 중요도 그래프 (서브플롯)
    fig, axes = plt.subplots(len(model_cols), 1, figsize=(12, 5 * len(model_cols)))
    
    # 단일 모델인 경우 axes를 리스트로 변환
    if len(model_cols) == 1:
        axes = [axes]
    
    for i, model_name in enumerate(model_cols):
        # 해당 모델의 특성 중요도만 추출
        model_data = combined_df[['feature', model_name]].sort_values(model_name, ascending=False).head(10)
        
        # 바 차트 생성
        axes[i].barh(range(len(model_data)), model_data[model_name], color='skyblue')
        axes[i].set_yticks(range(len(model_data)))
        axes[i].set_yticklabels(model_data['feature'])
        axes[i].invert_yaxis()  # 위에서부터 표시
        axes[i].set_title(f'{model_name} - Top 10 Features')
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f'feature_importance_by_model_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"모델별 특성 중요도 저장: {output_file}")

# CV 결과 비교 함수 추가
def plot_cv_results_comparison(model_results, output_dir, timestamp):
    """교차 검증 결과를 비교하는 boxplot과 bar+errorbar 그래프 생성"""
    # CV 데이터 수집
    cv_data = []
    
    for model_name, results in model_results.items():
        if 'cv_scores' in results and not results['cv_scores'].empty:
            # CV 점수를 확인하고 데이터프레임에 추가
            for _, row in results['cv_scores'].iterrows():
                cv_score = row['cv_score'] if 'cv_score' in row else 0
                cv_data.append({
                    'Model': model_name,
                    'CV Score': cv_score
                })
    
    if not cv_data:
        print("CV 결과 데이터가 없어 CV 비교 그래프를 생성할 수 없습니다.")
        return
    
    # 데이터프레임 생성
    cv_df = pd.DataFrame(cv_data)
    
    # 1. Boxplot 생성
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Model', y='CV Score', data=cv_df)
    
    # 개별 점 추가
    sns.stripplot(x='Model', y='CV Score', data=cv_df, 
                 size=4, color='black', alpha=0.5)
    
    plt.title('Cross-Validation Score Distribution', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('CV Score (ROC AUC or F1)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 그래프 저장
    boxplot_file = os.path.join(output_dir, f'cv_boxplot_comparison_{timestamp}.png')
    plt.savefig(boxplot_file, dpi=300)
    plt.close()
    print(f"CV Boxplot 비교 저장: {boxplot_file}")
    
    # 2. Bar + Errorbar 그래프 생성
    plt.figure(figsize=(10, 6))
    
    # 모델별 평균 및 표준편차 계산
    cv_stats = cv_df.groupby('Model')['CV Score'].agg(['mean', 'std']).reset_index()
    
    # 바 차트 생성
    x = np.arange(len(cv_stats))
    width = 0.6
    bars = plt.bar(x, cv_stats['mean'], width, 
                  yerr=cv_stats['std'], capsize=5, 
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    # 그래프 설정
    plt.xticks(x, cv_stats['Model'])
    plt.title('Cross-Validation Performance (Mean ± Std)', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Mean CV Score (ROC AUC or F1)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 각 바 위에 평균값 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cv_stats["mean"].iloc[i]:.4f} ± {cv_stats["std"].iloc[i]:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 그래프 저장
    barplot_file = os.path.join(output_dir, f'cv_barplot_comparison_{timestamp}.png')
    plt.savefig(barplot_file, dpi=300)
    plt.close()
    print(f"CV Bar+Errorbar 비교 저장: {barplot_file}")
    
    return cv_stats

# 메인 실행 코드
if __name__ == "__main__":
    # 성능 지표 비교 표 생성
    metrics_comparison = create_metrics_comparison(model_results, output_dir, timestamp)
    
    # CV 결과 비교 (추가됨)
    cv_stats = plot_cv_results_comparison(model_results, output_dir, timestamp)
    
    # 성능 지표 통합 바 차트 생성
    if metrics_comparison is not None:
        plot_metrics_comparison(metrics_comparison, output_dir, timestamp)
    
    # ROC 곡선 비교
    plot_roc_comparison(model_results, output_dir, timestamp)
    
    # PR 곡선 비교
    plot_pr_comparison(model_results, output_dir, timestamp)
    
    # Calibration 곡선 비교
    plot_calibration_comparison(model_results, output_dir, timestamp)
    
    # 예측 확률 분포 비교
    plot_probability_distribution_comparison(model_results, output_dir, timestamp)
    
    # 특성 중요도 비교
    plot_feature_importance_comparison(model_results, output_dir, timestamp, top_n=20)
    
    print(f"\n모델 비교 분석 완료. 결과는 {output_dir} 디렉토리에 저장되었습니다.")
