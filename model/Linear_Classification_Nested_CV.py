#!/usr/bin/env python3
# linear_classification_nested_cv.py

import pandas as pd
import numpy as np
import os
import joblib
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap
import argparse
from pathlib import Path

# 모델링 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                          matthews_corrcoef, confusion_matrix, roc_curve, 
                          precision_recall_curve, auc, average_precision_score,
                          roc_auc_score)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='선형 모델 기반 Nested CV 분류')
parser.add_argument('--n_jobs', type=int, default=0, help='사용할 CPU 코어 수')
parser.add_argument('--input_file', type=str, required=True, help='입력 데이터 파일 경로')
parser.add_argument('--outer_folds', type=int, default=5, help='외부 CV 폴드 수')
parser.add_argument('--inner_folds', type=int, default=3, help='내부 CV 폴드 수')
parser.add_argument('--random_iter', type=int, default=50, help='RandomizedSearchCV 반복 횟수')
parser.add_argument('--models', type=str, default='logistic', help='모델 선택')
parser.add_argument('--threshold', type=float, default=0.23, help='클래스 구분 임계값')

def get_cpu_count(requested_jobs=0):
    total_cpus = os.cpu_count() or 4
    if requested_jobs == 0: return max(1, int(total_cpus * 0.75))
    elif requested_jobs == -1: return total_cpus
    else: return min(requested_jobs, total_cpus)

def load_and_preprocess_data(input_file, threshold=0.23):
    """데이터 로드 및 전처리"""
    df = pd.read_csv(input_file)
    print(f"원본 데이터 크기: {df.shape}")

    # 불필요한 컬럼 제거
    cols_to_drop = ['pdb', 'seed', 'sample', 'data_file', 'chain_iptm', 'chain_pair_iptm', 
                    'chain_pair_pae_min', 'chain_ptm', 'format', 'model_path', 'native_path',
                    'Fnat', 'Fnonnat', 'rRMS', 'iRMS', 'LRMS']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df = df.dropna()

    # DockQ 값 기반으로 이진 클래스 생성 (0.23 이상: 1, 미만: 0)
    df['target'] = (df['DockQ'] >= threshold).astype(int)
    print(f"클래스 분포: 0 (incorrect) = {(df['target']==0).sum()}, 1 (acceptable+) = {(df['target']==1).sum()}")

    # 특성과 타겟 분리
    X = df.drop(columns=['DockQ', 'target'])
    y = df['target']
    query_ids = df['query'].copy()
    
    return X, y, query_ids

def get_hyperparameter_grid():
    """모델별 하이퍼파라미터 탐색 범위 정의"""
    return {
        'logistic': [
            # L2 규제 (Ridge)
            {
                'model__C': np.logspace(-4, 4, 10),
                'model__penalty': ['l2'],
                'model__solver': ['lbfgs', 'newton-cg', 'saga'],
                'model__class_weight': ['balanced', None],
                'model__multi_class': ['ovr']
            },
            # L1 규제 (Lasso)
            {
                'model__C': np.logspace(-4, 4, 10),
                'model__penalty': ['l1'],
                'model__solver': ['saga'],  # liblinear도 가능하지만 멀티클래스 지원 위해 saga만 사용
                'model__class_weight': ['balanced', None],
                'model__multi_class': ['ovr']
            },
            # ElasticNet 규제
            {
                'model__C': np.logspace(-4, 4, 10),
                'model__penalty': ['elasticnet'],
                'model__solver': ['saga'],  # ElasticNet은 saga만 지원
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'model__class_weight': ['balanced', None],
                'model__multi_class': ['ovr']
            }
        ]
    }

def create_model(model_name, params={}, n_jobs=1):
    """파이프라인 모델 인스턴스 생성"""
    if model_name == 'logistic':
        # 파이프라인으로 스케일러와 모델 결합
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=42, 
                max_iter=1000,
                n_jobs=n_jobs,
                **params))
        ])

def tune_hyperparameters(X_train, y_train, groups_train, model_name, inner_folds, n_iter, n_jobs):
    """내부 CV로 하이퍼파라미터 최적화"""
    param_grid = get_hyperparameter_grid()[model_name]
    base_model = create_model(model_name, n_jobs=1)
    inner_cv = GroupKFold(n_splits=inner_folds)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=inner_cv.split(X_train, y_train, groups=groups_train),
        scoring='average_precision',
        n_jobs=max(1, n_jobs // 2),
        verbose=0,
        random_state=42
    )
    
    random_search.fit(X_train, y_train, groups=groups_train)
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def calculate_metrics(y_true, y_pred, y_prob):
    """평가 지표 계산"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob)
    }

def calculate_and_save_shap(pipeline, X_test, model_name, fold_idx, output_dir):
    """선형 모델의 SHAP 값 계산 및 저장"""
    try:
        # 샘플 크기 제한
        shap_sample_size = min(1000, len(X_test))
        X_sample = X_test.iloc[:shap_sample_size].copy()
        
        # 스케일러 추출 및 데이터 변환
        scaler = pipeline.named_steps['scaler']
        model = pipeline.named_steps['model']
        X_sample_scaled = scaler.transform(X_sample)
        
        # SHAP 계산 - LinearExplainer 사용
        background = shap.utils.sample(X_sample_scaled, 100)
        explainer = shap.LinearExplainer(
            model, 
            background,
            feature_perturbation="correlation_dependent"
        )
        shap_values = explainer.shap_values(X_sample_scaled)
        
        # 이진 분류의 경우 클래스 1(양성)에 대한 SHAP만 사용
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # 저장 디렉토리 설정
        shap_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
        os.makedirs(shap_dir, exist_ok=True)
        
        # SHAP 값과 데이터 저장
        np.save(os.path.join(shap_dir, 'shap_values.npy'), shap_values)
        X_sample.to_csv(os.path.join(shap_dir, 'shap_samples.csv'), index=False)
        
        # 특성 이름 목록
        feature_names = X_sample.columns.tolist()
        
        # SHAP 벌떼 플롯 저장
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample_scaled, feature_names=feature_names, show=False)
        plt.title(f'{model_name} - SHAP Values (Fold {fold_idx})')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, 'shap_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP 막대 플롯 저장
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample_scaled, feature_names=feature_names, plot_type='bar', show=False)
        plt.title(f'{model_name} - Feature Importance (Fold {fold_idx})')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 계수 시각화
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
            coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
            
            plt.figure(figsize=(12, 8))
            plt.barh(y=coef_df['Feature'][:20], width=coef_df['Coefficient'][:20])
            plt.xlabel('Coefficient Value')
            plt.title(f'{model_name} - Top Coefficients (Fold {fold_idx})')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, 'coefficients.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 계수 저장
            coef_df.to_csv(os.path.join(shap_dir, 'coefficients.csv'), index=False)
        
        return True
    except Exception as e:
        print(f"SHAP 계산 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def save_fold_results(pipeline, X_test, y_test, y_pred, y_prob, metrics, best_params, 
                     model_name, fold_idx, output_dir):
    """각 폴드의 결과 저장"""
    # 결과 디렉토리 생성
    fold_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
    os.makedirs(fold_dir, exist_ok=True)
    
    # 모델 저장
    joblib.dump(pipeline, os.path.join(fold_dir, 'best_model.pkl'))
    
    # 예측값 및 확률 저장
    pred_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_prob
    })
    pred_df.to_csv(os.path.join(fold_dir, 'predictions.csv'), index=False)
    
    # 평가 지표 저장
    pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)
    
    # 하이퍼파라미터 저장
    with open(os.path.join(fold_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # ROC 곡선
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'roc_curve.png'), dpi=300)
    plt.close()

def analyze_coefficient_stability(models, feature_names, model_name, output_dir):
    """모든 폴드의 계수 안정성 분석"""
    # 계수 저장 배열
    all_coefs = []
    
    for i, pipeline in enumerate(models):
        model = pipeline.named_steps['model']
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            all_coefs.append(coef)
    
    if not all_coefs:
        return
    
    # 계수를 데이터프레임으로 변환
    coef_df = pd.DataFrame(all_coefs, columns=feature_names)
    
    # 계수 통계 계산
    coef_stats = pd.DataFrame({
        'feature': feature_names,
        'mean_coef': coef_df.mean(),
        'std_coef': coef_df.std(),
        'cv_percent': coef_df.std() / coef_df.mean().abs() * 100
    })
    
    # 중요도 순으로 정렬
    coef_stats = coef_stats.sort_values('mean_coef', key=abs, ascending=False)
    
    # 결과 저장
    output_dir = os.path.join(output_dir, 'coefficient_analysis')
    os.makedirs(output_dir, exist_ok=True)
    coef_stats.to_csv(os.path.join(output_dir, f'{model_name}_coefficient_stats.csv'), index=False)
    
    # 상위 20개 계수 시각화
    top_coefs = coef_stats.head(20)
    
    plt.figure(figsize=(12, 10))
    plt.barh(y=top_coefs['feature'], width=top_coefs['mean_coef'], xerr=top_coefs['std_coef'], capsize=5)
    plt.xlabel('Coefficient Value')
    plt.title(f'{model_name} - Top 20 Coefficients with Std Dev')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_coefficient_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return coef_stats

def analyze_global_shap_linear(models, X, feature_names, model_name, output_dir, max_samples=5000):
    """전체 데이터셋에 대한 SHAP 분석 및 시각화 - 선형 모델용"""
    # 결과 디렉토리
    global_dir = os.path.join(output_dir, 'global_shap', model_name)
    os.makedirs(global_dir, exist_ok=True)
    
    try:
        print(f"\n{model_name} 모델의 전체 SHAP 분석 시작...")
        
        # 데이터가 너무 크면 샘플링
        if len(X) > max_samples:
            print(f"데이터 크기 제한: {len(X)} → {max_samples} 샘플")
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_indices].copy()
        else:
            X_sample = X.copy()
        
        # 모델 중 첫 번째 유효한 모델 선택
        valid_model = None
        for model in models:
            if model is not None:
                valid_model = model
                break
        
        if valid_model is None:
            print("유효한 모델이 없습니다.")
            return
        
        # 스케일러와 모델 추출
        scaler = valid_model.named_steps['scaler']
        model = valid_model.named_steps['model']
        
        # 데이터 스케일링
        X_sample_scaled = scaler.transform(X_sample)
        
        # SHAP 계산
        background = shap.utils.sample(X_sample_scaled, 100)
        explainer = shap.LinearExplainer(
            model, 
            background,
            feature_perturbation="correlation_dependent"
        )
        shap_values = explainer.shap_values(X_sample_scaled)
        
        # 이진 분류의 경우 클래스 1(양성)에 대한 SHAP 값만 사용
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # SHAP 값 저장 - 트리 기반 모델과 동일한 형식
        np.save(os.path.join(global_dir, 'global_shap_values.npy'), shap_values)
        X_sample.to_csv(os.path.join(global_dir, 'global_shap_samples.csv'), index=True)
        
        # 1. 전체 데이터 벌떼 플롯
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_sample_scaled, feature_names=feature_names, show=False)
        plt.title(f'{model_name} - Global SHAP Values')
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, 'global_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 특성 중요도 막대 플롯
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample_scaled, feature_names=feature_names, 
                         plot_type='bar', show=False)
        plt.title(f'{model_name} - Global Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, 'global_shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 평균 SHAP 값 및 표준편차 계산
        mean_shap = np.abs(shap_values).mean(axis=0)
        std_shap = np.abs(shap_values).std(axis=0)
        
        # 특성 중요도 DataFrame 생성 - 트리 기반 모델과 동일한 형식
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_importance': mean_shap,
            'std_importance': std_shap
        })
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        importance_df.to_csv(os.path.join(global_dir, 'feature_importance.csv'), index=False)
        
        # 4. 상위 20개 특성 중요도 시각화 (평균 +/- 표준편차)
        top_n = min(20, len(feature_names))
        top_features = importance_df['feature'][:top_n].values
        top_importance = importance_df['mean_importance'][:top_n].values
        top_std = importance_df['std_importance'][:top_n].values
        
        plt.figure(figsize=(12, 10))
        plt.barh(y=np.arange(len(top_features)), width=top_importance, xerr=top_std, 
                capsize=5, color='skyblue')
        plt.yticks(np.arange(len(top_features)), top_features)
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'{model_name} - Top {top_n} Feature Importance with Std Dev')
        plt.gca().invert_yaxis()  # 중요도 높은 특성을 위에 표시
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, 'top_features_with_std.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{model_name} 모델의 전체 SHAP 분석 완료")
        
    except Exception as e:
        print(f"전체 SHAP 분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

def save_curve_data(model_names, n_folds, output_dir):
    """각 모델별 ROC 및 PR 커브 데이터 저장"""
    curves_dir = os.path.join(output_dir, 'curve_data')
    os.makedirs(curves_dir, exist_ok=True)
    
    for model_name in model_names:
        # ROC 커브 데이터
        tprs = []
        aucs = []
        fpr_grid = np.linspace(0, 1, 100)
        
        # PR 커브 데이터
        precisions = []
        recalls = []
        ap_scores = []
        recall_grid = np.linspace(0, 1, 100)
        
        for fold_idx in range(n_folds):
            fold_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
            pred_file = os.path.join(fold_dir, 'predictions.csv')
            
            if os.path.exists(pred_file):
                preds = pd.read_csv(pred_file)
                
                # ROC 데이터 계산
                fpr, tpr, _ = roc_curve(preds['actual'], preds['probability'])
                roc_auc = auc(fpr, tpr)
                tprs.append(np.interp(fpr_grid, fpr, tpr))
                aucs.append(roc_auc)
                
                # PR 데이터 계산
                precision, recall, _ = precision_recall_curve(preds['actual'], preds['probability'])
                ap = average_precision_score(preds['actual'], preds['probability'])
                precisions.append(np.interp(recall_grid, recall[::-1], precision[::-1]))
                ap_scores.append(ap)
        
        if not tprs:
            continue
            
        # 평균 ROC 데이터
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        # 평균 PR 데이터
        mean_precision = np.mean(precisions, axis=0)
        mean_ap = np.mean(ap_scores)
        std_ap = np.std(ap_scores)
        
        # 데이터 저장
        curve_data = {
            'model_name': model_name,
            'fpr_grid': fpr_grid.tolist(),
            'mean_tpr': mean_tpr.tolist(),
            'mean_auc': float(mean_auc),
            'std_auc': float(std_auc),
            'recall_grid': recall_grid.tolist(),
            'mean_precision': mean_precision.tolist(),
            'mean_ap': float(mean_ap),
            'std_ap': float(std_ap),
        }
        
        # JSON 파일로 저장
        with open(os.path.join(curves_dir, f'{model_name}_curve_data.json'), 'w') as f:
            json.dump(curve_data, f, indent=2)
    
    print(f"모든 모델의 ROC/PR 커브 데이터 저장 완료")

def summarize_model_performance(model_results, model_names, output_dir, outer_folds):
    """Nested CV 최종 성능 요약 및 시각화"""
    summary_dir = os.path.join(output_dir, 'performance_summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # 1. 모든 지표에 대한 통계 테이블 생성
    all_metrics = []
    
    for model_name in model_names:
        model_metrics = model_results[model_name]['metrics']
        if not model_metrics:
            continue
            
        # 모든 폴드의 지표 통합
        for fold_idx, fold_metrics in enumerate(model_metrics):
            fold_data = {
                'model': model_name,
                'fold': fold_idx,
                **fold_metrics
            }
            all_metrics.append(fold_data)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(summary_dir, 'all_fold_metrics.csv'), index=False)
    
    # 2. 모델별 평균 성능 및 표준편차 계산
    summary_stats = metrics_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'mcc': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'pr_auc': ['mean', 'std']
    })
    
    # 보기 좋게 포맷팅
    formatted_summary = pd.DataFrame()
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc', 'pr_auc']:
        for model in summary_stats.index:
            mean_val = summary_stats.loc[model, (metric, 'mean')]
            std_val = summary_stats.loc[model, (metric, 'std')]
            formatted_summary.loc[model, metric] = f"{mean_val:.4f} ± {std_val:.4f}"
    
    formatted_summary.to_csv(os.path.join(summary_dir, 'performance_summary.csv'))
    
    # 3. 성능 지표 박스플롯 (모델 간 비교)
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='model', y=metric, data=metrics_df)
        plt.title(f'{metric.upper()} by Model')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, f'{metric}_boxplot.png'), dpi=300)
        plt.close()
        
        # 바플롯 (평균 ± 표준편차)
        plt.figure(figsize=(10, 6))
        summary_data = summary_stats.reset_index()
        
        for i, model in enumerate(summary_data['model']):
            mean_val = summary_stats.loc[model, (metric, 'mean')]
            std_val = summary_stats.loc[model, (metric, 'std')]
            plt.bar(i, mean_val, yerr=std_val, capsize=10, label=model)
            
        plt.xticks(range(len(summary_data['model'])), summary_data['model'])
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} Performance by Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, f'{metric}_barplot.png'), dpi=300)
        plt.close()
    
    # 4. 모든 모델의 예측 대 실제값 통합 시각화
    plt.figure(figsize=(12, 8))
    
    for model_name in model_names:
        all_actual = []
        all_predicted_prob = []
        
        for fold_idx in range(outer_folds):
            pred_file = os.path.join(output_dir, f'fold_{fold_idx}', model_name, 'predictions.csv')
            if os.path.exists(pred_file):
                preds = pd.read_csv(pred_file)
                all_actual.extend(preds['actual'])
                all_predicted_prob.extend(preds['probability'])
        
        if all_actual:
            # 예측 확률 구간에 따른 실제 양성 비율 계산
            bins = np.linspace(0, 1, 11)  # 10개 구간으로 나눔
            bin_indices = np.digitize(all_predicted_prob, bins) - 1
            bin_counts = np.bincount(bin_indices, minlength=len(bins)-1)
            bin_positive = np.zeros(len(bins)-1)
            
            for i, (prob, actual) in enumerate(zip(all_predicted_prob, all_actual)):
                bin_idx = min(int(prob * 10), 9)  # 0-9 인덱스
                bin_positive[bin_idx] += actual
            
            # 0으로 나누기 방지
            bin_positive_rate = np.zeros(len(bins)-1)
            for i in range(len(bins)-1):
                if bin_counts[i] > 0:
                    bin_positive_rate[i] = bin_positive[i] / bin_counts[i]
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.plot(bin_centers, bin_positive_rate, 'o-', label=model_name)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Positive Rate')
    plt.title('Calibration Curves (Reliability Diagram)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'calibration_curves.png'), dpi=300)
    plt.close()
    
    # 5. 최적의 하이퍼파라미터 요약
    for model_name in model_names:
        best_params_summary = {}
        
        for fold_idx in range(outer_folds):
            params_file = os.path.join(output_dir, f'fold_{fold_idx}', model_name, 'best_params.json')
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    fold_params = json.load(f)
                    
                # 각 파라미터에 대한 폴드별 값 저장
                for param, value in fold_params.items():
                    if param not in best_params_summary:
                        best_params_summary[param] = []
                    best_params_summary[param].append(value)
        
        # 각 파라미터별 가장 많이 선택된 값 (모드) 계산
        most_common_params = {}
        for param, values in best_params_summary.items():
            # 이산값인 경우 최빈값, 연속값인 경우 평균 사용
            if all(isinstance(v, (int, float)) for v in values):
                most_common_params[param] = f"평균: {np.mean(values):.6f}, 표준편차: {np.std(values):.6f}"
            else:
                from collections import Counter
                counter = Counter(values)
                most_common_value, count = counter.most_common(1)[0]
                most_common_params[param] = f"{most_common_value} (빈도: {count}/{len(values)})"
        
        # 결과 저장
        with open(os.path.join(summary_dir, f'{model_name}_best_params_summary.json'), 'w') as f:
            json.dump(most_common_params, f, indent=4, ensure_ascii=False)
        
    print(f"모델 성능 요약이 {summary_dir} 디렉토리에 저장되었습니다.")

def run_nested_cv(X, y, query_ids, model_names, output_dir, outer_folds=5, inner_folds=3, 
                 n_iter=100, n_jobs=4):
    """Nested CV 실행"""
    # 외부 CV 폴드 (쿼리 기반)
    outer_cv = GroupKFold(n_splits=outer_folds)
    
    # 모델별 결과 저장
    model_results = {model: {
        'metrics': [], 
        'models': [],
        'feature_names': []
    } for model in model_names}
    
    # query 열이 있는지 확인하고 처리
    has_query_column = 'query' in X.columns
    
    # 데이터 준비
    if has_query_column:
        X_no_query = X.drop(columns=['query'])
        feature_names = X_no_query.columns.tolist()
    else:
        X_no_query = X
        feature_names = X.columns.tolist()
    
    for model_name in model_names:
        model_results[model_name]['feature_names'] = feature_names
    
    # 외부 CV 루프
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_no_query, y, groups=query_ids)):
        print(f"\n=== 외부 폴드 {fold_idx+1}/{outer_folds} ===")
        
        # 훈련 데이터와 테스트 데이터 분할
        X_train, X_test = X_no_query.iloc[train_idx], X_no_query.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        query_train = query_ids.iloc[train_idx] if has_query_column else None
        
        # 각 모델에 대해 내부 CV 및 테스트
        for model_name in model_names:
            print(f"\n{model_name.upper()} 모델 훈련 중 (폴드 {fold_idx+1})")
            start_time = time.time()
            
            try:
                # 내부 CV로 하이퍼파라미터 최적화
                best_model, best_params, best_score = tune_hyperparameters(
                    X_train, y_train, query_train, model_name, 
                    inner_folds, n_iter, n_jobs
                )
                
                # 테스트 세트 예측
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]
                
                # 평가 지표 계산
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                
                # 결과 저장
                model_results[model_name]['metrics'].append(metrics)
                model_results[model_name]['models'].append(best_model)
                
                # 폴드별 결과 저장
                save_fold_results(
                    best_model, X_test, y_test, y_pred, y_prob,
                    metrics, best_params, model_name, fold_idx, output_dir
                )
                
                # SHAP 값 계산 및 저장
                calculate_and_save_shap(
                    best_model, X_test, model_name, fold_idx, output_dir
                )
                
                elapsed_time = time.time() - start_time
                print(f"{model_name.upper()} 폴드 {fold_idx+1} 완료: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"{model_name.upper()} 모델 처리 중 오류: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # 각 모델의 계수 안정성 분석
    for model_name in model_names:
        analyze_coefficient_stability(
            model_results[model_name]['models'],
            model_results[model_name]['feature_names'],
            model_name,
            output_dir
        )
    
    # 전체 데이터셋에 대한 SHAP 분석
    for model_name in model_names:
        if model_results[model_name]['models']:  # 모델이 존재하는 경우에만 실행
            analyze_global_shap_linear(
                model_results[model_name]['models'],
                X_no_query,
                feature_names,
                model_name,
                output_dir
            )
    
    # 각 모델별 ROC/PR 커브 데이터 저장
    save_curve_data(model_names, outer_folds, output_dir)
    
    # 모델 성능 요약
    summarize_model_performance(model_results, model_names, output_dir, outer_folds)
    
    return model_results

def main():
    args = parser.parse_args()
    
    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'linear_classification_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성: {output_dir}")
    
    # CPU 코어 수 설정
    n_jobs = get_cpu_count(args.n_jobs)
    
    # 학습할 모델 목록
    model_names = [model.strip() for model in args.models.split(',')]
    print(f"학습할 모델: {', '.join(model_names)}")
    
    # 데이터 로드 및 전처리
    X, y, query_ids = load_and_preprocess_data(args.input_file, args.threshold)
    
    # Nested CV 실행
    run_nested_cv(
        X, y, query_ids, model_names, output_dir,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        n_iter=args.random_iter,
        n_jobs=n_jobs
    )
    
    print(f"\n분석 완료. 결과는 {output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
