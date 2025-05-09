#!/usr/bin/env python3
# classification_nested_cv.py

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            matthews_corrcoef, confusion_matrix, roc_curve, 
                            precision_recall_curve, auc, average_precision_score,
                            roc_auc_score)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='Nested CV를 이용한 분류 모델 비교')
parser.add_argument('--n_jobs', type=int, default=0, help='사용할 CPU 코어 수')
parser.add_argument('--input_file', type=str, required=True, help='입력 데이터 파일 경로')
parser.add_argument('--outer_folds', type=int, default=5, help='외부 CV 폴드 수')
parser.add_argument('--inner_folds', type=int, default=3, help='내부 CV 폴드 수')
parser.add_argument('--random_iter', type=int, default=100, help='RandomizedSearchCV 반복 횟수')
parser.add_argument('--models', type=str, default='rf,lgb,xgb', help='모델 선택')
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
    
    # 데이터 스케일링
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    return X_scaled, y, query_ids, scaler

def get_hyperparameter_grid():
    """모델별 하이퍼파라미터 탐색 범위 정의"""
    return {
        'rf': {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 1/3, 0.5, 0.7],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'lgb': {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'num_leaves': [20, 31, 50, 100],
            'max_depth': [5, 10, 15, 20],
            'min_child_samples': [20, 30, 50],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0.01, 0.1, 0.5, 1],
            'reg_lambda': [0.01, 0.1, 0.5, 1],
            'is_unbalance': [True, False]
        },
        'xgb': {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0.01, 0.1, 0.3, 0.5],
            'reg_alpha': [0.01, 0.1, 0.5, 1],
            'reg_lambda': [0.01, 0.1, 0.5, 1],
            'scale_pos_weight': [1, 3, 5]
        }
    }

def create_model(model_name, params, n_jobs):
    """모델 인스턴스 생성"""
    if model_name == 'rf':
        return RandomForestClassifier(random_state=42, n_jobs=n_jobs, **params)
    elif model_name == 'lgb':
        default_params = {
            'min_child_samples': 20, 'min_child_weight': 1e-3, 
            'min_split_gain': 0.001, 'verbose': -1
        }
        default_params.update(params)
        return lgb.LGBMClassifier(random_state=42, n_jobs=n_jobs, **default_params)
    elif model_name == 'xgb':
        return xgb.XGBClassifier(random_state=42, n_jobs=n_jobs, 
                               eval_metric='logloss', **params)

def tune_hyperparameters(X_train, y_train, groups_train, model_name, inner_folds, n_iter, n_jobs):
    """내부 CV로 하이퍼파라미터 최적화"""
    param_grid = get_hyperparameter_grid()[model_name]
    base_model = create_model(model_name, {}, n_jobs=1)
    inner_cv = GroupKFold(n_splits=inner_folds)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=inner_cv.split(X_train, y_train, groups=groups_train),
        scoring='average_precision',  # PR-AUC 사용
        n_jobs=max(1, n_jobs // 2),
        verbose=1,
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

def calculate_and_save_shap(model, X_test, model_name, fold_idx, output_dir):
    """SHAP 값 계산 및 저장 - 분류 모델에 최적화"""
    try:
        # 샘플 크기 제한
        shap_sample_size = min(1000, len(X_test))
        X_sample = X_test.iloc[:shap_sample_size].copy()
        print(f"SHAP 계산 중: {model_name}, 폴드 {fold_idx} (샘플 수: {shap_sample_size})")
        
        # SHAP 계산
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        
        # 저장 디렉토리 설정
        shap_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
        os.makedirs(shap_dir, exist_ok=True)
        
        # 클래스 1(양성)에 대한 SHAP 값 추출
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
            print(f"다차원 SHAP 값 감지: {shap_values.values.shape}")
            # 클래스 1(양성)의 SHAP 값만 사용
            class_idx = 1
            # 새 설명 객체 생성
            values_for_class1 = shap_values.values[:, :, class_idx]
            base_values = shap_values.base_values
            if isinstance(base_values, np.ndarray) and len(base_values.shape) > 1:
                base_values = base_values[:, class_idx]
            
            # 새 설명 객체 생성
            class1_exp = shap.Explanation(
                values=values_for_class1,
                base_values=base_values,
                data=shap_values.data,
                feature_names=shap_values.feature_names
            )
            shap_values = class1_exp
            print(f"클래스 1에 대한 SHAP 값으로 변환 완료: {class1_exp.values.shape}")
        
        # SHAP 값과 데이터 저장
        joblib.dump(shap_values, os.path.join(shap_dir, 'shap_values.pkl'))
        X_sample.to_csv(os.path.join(shap_dir, 'shap_samples.csv'), index=False)
        
        # SHAP 벌떼 플롯 저장
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, 'shap_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP 막대 플롯 저장
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"SHAP 계산 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def save_fold_results(model, X_test, y_test, y_pred, y_prob, metrics, best_params, 
                     model_name, fold_idx, output_dir):
    """각 폴드의 결과 저장"""
    # 결과 디렉토리 생성
    fold_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
    os.makedirs(fold_dir, exist_ok=True)
    
    # 모델 저장
    joblib.dump(model, os.path.join(fold_dir, 'best_model.pkl'))
    
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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # PR 곡선
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'pr_curve.png'), dpi=300)
    plt.close()
    
    # 캘리브레이션 곡선
    try:
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title(f'{model_name} - Calibration Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, 'calibration_curve.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"캘리브레이션 곡선 생성 중 오류: {str(e)}")

def combine_shap_values(model_names, n_folds, output_dir):
    """모든 폴드의 SHAP 값 통합 및 시각화 - 분류 모델에 최적화"""
    for model_name in model_names:
        print(f"{model_name} 모델의 SHAP 값 통합 중...")
        combined_dir = os.path.join(output_dir, 'combined_shap', model_name)
        os.makedirs(combined_dir, exist_ok=True)
        
        # 폴드별 SHAP 중요도 저장
        all_importance_dfs = []
        
        # 첫 번째 유효한 폴드 찾기 (전체 시각화용)
        first_valid_fold = None
        first_valid_shap = None
        first_valid_sample = None
        
        for fold_idx in range(n_folds):
            shap_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
            shap_file = os.path.join(shap_dir, 'shap_values.pkl')
            sample_file = os.path.join(shap_dir, 'shap_samples.csv')
            
            if not (os.path.exists(shap_file) and os.path.exists(sample_file)):
                print(f"폴드 {fold_idx}의 SHAP 파일이 없습니다. 건너뜁니다.")
                continue
                
            try:
                # 데이터 로드
                samples = pd.read_csv(sample_file)
                shap_values = joblib.load(shap_file)
                
                print(f"폴드 {fold_idx} SHAP 값 로드 완료:")
                print(f"  - 샘플 형태: {samples.shape}")
                if hasattr(shap_values, 'values'):
                    print(f"  - SHAP 값 형태: {shap_values.values.shape}")
                else:
                    print(f"  - SHAP 값 형태: {shap_values.shape if hasattr(shap_values, 'shape') else '알 수 없음'}")
                
                # 첫 번째 유효한 폴드 저장
                if first_valid_fold is None:
                    first_valid_fold = fold_idx
                    first_valid_shap = shap_values
                    first_valid_sample = samples
                
                # 특성 중요도 계산
                if hasattr(shap_values, 'values'):
                    importance = np.abs(shap_values.values).mean(0)
                    feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else samples.columns.tolist()
                else:
                    importance = np.abs(shap_values).mean(0)
                    feature_names = samples.columns.tolist()
                
                # 중요도 데이터프레임 생성
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance,
                    'fold': fold_idx
                })
                all_importance_dfs.append(importance_df)
                print(f"폴드 {fold_idx}의 특성 중요도 계산 완료")
                
            except Exception as e:
                print(f"폴드 {fold_idx}의 SHAP 값 처리 중 오류: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 모든 폴드의 특성 중요도 통합
        if all_importance_dfs:
            try:
                # 특성 중요도 통합 및 평균 계산
                all_importance = pd.concat(all_importance_dfs)
                mean_importance = all_importance.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
                mean_importance = mean_importance.sort_values('mean', ascending=False)
                mean_importance.to_csv(os.path.join(combined_dir, 'mean_feature_importance.csv'), index=False)
                
                # 상위 20개 특성에 대한 막대 플롯
                top_features = mean_importance['feature'][:20].tolist()
                plt.figure(figsize=(12, 8))
                plt.barh(
                    y=top_features[::-1],  # 역순으로 표시 (상위 특성이 위에 오도록)
                    width=mean_importance.loc[mean_importance['feature'].isin(top_features), 'mean'][:20],
                    xerr=mean_importance.loc[mean_importance['feature'].isin(top_features), 'std'][:20],
                    capsize=5,
                    color='skyblue'
                )
                plt.xlabel('Mean |SHAP value|')
                plt.title(f'{model_name} - Mean Feature Importance (All Folds)')
                plt.tight_layout()
                plt.savefig(os.path.join(combined_dir, 'combined_shap_bar.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 첫 번째 유효한 폴드로 벌떼 플롯 생성 (전체 데이터로)
                if first_valid_shap is not None:
                    try:
                        plt.figure(figsize=(14, 10))
                        # 원본 SHAP 값과 데이터 사용 (필터링하지 않음)
                        shap.summary_plot(first_valid_shap, first_valid_sample, show=False, max_display=20)
                        plt.title(f'{model_name} - SHAP Values (Fold {first_valid_fold})')
                        plt.tight_layout()
                        plt.savefig(os.path.join(combined_dir, 'combined_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"벌떼 플롯 저장 완료: combined_shap_beeswarm.png")
                    except Exception as e:
                        print(f"벌떼 플롯 생성 중 오류: {str(e)}")
                        traceback.print_exc()
                        
                        # 대안적 방법 시도
                        try:
                            print("대안적 방법으로 벌떼 플롯 생성 시도...")
                            plt.figure(figsize=(14, 10))
                            
                            # SHAP 객체 유형에 따라 다르게 처리
                            if hasattr(first_valid_shap, 'values'):
                                # Explanation 객체
                                shap_values_np = first_valid_shap.values
                                feature_names = first_valid_shap.feature_names
                                
                                # 바로 기본 plot 함수 사용
                                shap.summary_plot(shap_values_np, first_valid_sample, 
                                                 feature_names=feature_names, show=False)
                            else:
                                # 단순 numpy 배열
                                shap.summary_plot(first_valid_shap, first_valid_sample, show=False)
                                
                            plt.title(f'{model_name} - SHAP Values (Fold {first_valid_fold})')
                            plt.tight_layout()
                            plt.savefig(os.path.join(combined_dir, 'combined_shap_beeswarm_alt.png'), dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"대안적 벌떼 플롯 저장 완료: combined_shap_beeswarm_alt.png")
                        except Exception as e2:
                            print(f"대안적 벌떼 플롯 생성도 실패: {str(e2)}")
                            
                print(f"{model_name} 모델의 SHAP 값 통합 완료")
                
            except Exception as e:
                print(f"SHAP 값 통합 중 오류: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"{model_name} 모델의 유효한 SHAP 데이터가 없습니다.")

def overlay_curves(model_names, n_folds, output_dir):
    """모든 모델의 ROC 및 PR 곡선 비교"""
    # ROC 곡선 오버레이
    plt.figure(figsize=(10, 8))
    
    for model_name in model_names:
        tprs = []
        aucs = []
        fpr_grid = np.linspace(0, 1, 100)
        
        for fold_idx in range(n_folds):
            fold_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
            pred_file = os.path.join(fold_dir, 'predictions.csv')
            
            if os.path.exists(pred_file):
                preds = pd.read_csv(pred_file)
                fpr, tpr, _ = roc_curve(preds['actual'], preds['probability'])
                roc_auc = auc(fpr, tpr)
                tprs.append(np.interp(fpr_grid, fpr, tpr))
                aucs.append(roc_auc)
        
        if not tprs:
            continue
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        plt.plot(fpr_grid, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})', 
                 lw=2, alpha=0.8)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_models_roc.png'), dpi=300)
    plt.close()
    
    # PR 곡선 오버레이
    plt.figure(figsize=(10, 8))
    
    for model_name in model_names:
        precisions = []
        recalls = []
        ap_scores = []
        recall_grid = np.linspace(0, 1, 100)
        
        for fold_idx in range(n_folds):
            fold_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
            pred_file = os.path.join(fold_dir, 'predictions.csv')
            
            if os.path.exists(pred_file):
                preds = pd.read_csv(pred_file)
                precision, recall, _ = precision_recall_curve(preds['actual'], preds['probability'])
                ap = average_precision_score(preds['actual'], preds['probability'])
                precisions.append(np.interp(recall_grid, recall[::-1], precision[::-1]))
                ap_scores.append(ap)
        
        if not precisions:
            continue
            
        mean_precision = np.mean(precisions, axis=0)
        mean_ap = np.mean(ap_scores)
        std_ap = np.std(ap_scores)
        plt.plot(recall_grid, mean_precision, label=f'{model_name} (AP = {mean_ap:.3f} ± {std_ap:.3f})', 
                 lw=2, alpha=0.8)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Models')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_models_pr.png'), dpi=300)
    plt.close()

def compare_models(model_results, output_dir):
    """모델 성능 비교 분석"""
    # 모델별 평균 성능 지표 계산
    metrics_summary = {}
    
    for model_name, fold_results in model_results.items():
        metrics_summary[model_name] = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'mcc': [], 'roc_auc': [], 'pr_auc': []
        }
        
        for fold_metrics in fold_results['metrics']:
            for metric, value in fold_metrics.items():
                metrics_summary[model_name][metric].append(value)
    
    # 평균 및 표준편차 계산
    comparison_data = []
    
    for model_name, metrics in metrics_summary.items():
        for metric, values in metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Metric': metric,
                'Mean': np.mean(values),
                'Std': np.std(values)
            })
    
    # 성능 지표 테이블 생성
    if comparison_data:
        metrics_comparison = pd.DataFrame(comparison_data)
        metrics_comparison.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
        
        # 모델별 성능 시각화
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['accuracy', 'f1', 'roc_auc', 'pr_auc']
        colors = ['blue', 'green', 'red']
        
        for i, metric in enumerate(metrics_to_plot):
            metric_data = metrics_comparison[metrics_comparison['Metric'] == metric]
            axes[i].bar(
                metric_data['Model'],
                metric_data['Mean'],
                yerr=metric_data['Std'],
                capsize=5,
                color=colors[:len(metric_data)]
            )
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()
    
    return comparison_data

def run_nested_cv(X, y, query_ids, model_names, output_dir, outer_folds=5, inner_folds=3, 
                 n_iter=100, n_jobs=4):
    """Nested CV 실행 및 SHAP 값 누적"""
    # 외부 CV 폴드 (쿼리 기반)
    outer_cv = GroupKFold(n_splits=outer_folds)
    
    # 모델별 결과 저장
    model_results = {model: {'metrics': [], 'predictions': []} for model in model_names}
    
    # query 열이 있는지 확인하고 처리
    has_query_column = 'query' in X.columns
    
    # 특성 이름 저장 (쿼리 제외)
    if has_query_column:
        X_no_query = X.drop(columns=['query'])
        feature_names = X_no_query.columns.tolist()
    else:
        X_no_query = X
        feature_names = X.columns.tolist()
    
    # 전체 데이터의 SHAP 값을 저장할 딕셔너리 초기화 - 쿼리 제외한 특성 수 사용
    n_samples, n_features = len(X), len(feature_names)
    shap_results = {
        model: {
            'values': np.zeros((n_samples, n_features)),
            'is_calculated': np.zeros(n_samples, dtype=bool)  # 계산 여부 추적
        } for model in model_names
    }
    
    # 외부 CV 루프
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=query_ids)):
        print(f"\n=== 외부 폴드 {fold_idx+1}/{outer_folds} ===")
        
        # 훈련 데이터와 테스트 데이터 분할
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        query_train = query_ids.iloc[train_idx]
        
        # 데이터 전처리 (쿼리 제거)
        if has_query_column:
            X_train_no_query = X_train.drop(columns=['query'])
            X_test_no_query = X_test.drop(columns=['query'])
        else:
            X_train_no_query = X_train
            X_test_no_query = X_test
        
        # 각 모델에 대해 내부 CV 및 테스트
        for model_name in model_names:
            print(f"\n{model_name.upper()} 모델 훈련 중 (폴드 {fold_idx+1})")
            start_time = time.time()
            
            try:
                # 내부 CV로 하이퍼파라미터 최적화
                best_model, best_params, best_score = tune_hyperparameters(
                    X_train_no_query, y_train, query_train, model_name, 
                    inner_folds, n_iter, n_jobs
                )
                
                # 테스트 세트 예측
                y_pred = best_model.predict(X_test_no_query)
                y_prob = best_model.predict_proba(X_test_no_query)[:, 1]
                
                # 평가 지표 계산
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                
                # 결과 저장
                model_results[model_name]['metrics'].append(metrics)
                model_results[model_name]['predictions'].append((y_test, y_pred, y_prob))
                
                # 폴드별 결과 저장
                save_fold_results(
                    best_model, X_test_no_query, y_test, y_pred, y_prob,
                    metrics, best_params, model_name, fold_idx, output_dir
                )
                
                # 외부 테스트 폴드의 SHAP 값 계산 및 누적
                calculate_and_accumulate_shap(
                    best_model, X_test_no_query, test_idx, model_name, 
                    fold_idx, shap_results, output_dir
                )
                
                elapsed_time = time.time() - start_time
                print(f"{model_name.upper()} 폴드 {fold_idx+1} 완료: "
                      f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
                      f"처리 시간: {elapsed_time:.2f}초")
                
            except Exception as e:
                print(f"{model_name.upper()} 모델 처리 중 오류: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # 모델 비교 분석
    compare_models(model_results, output_dir)
    
    # 누적된 SHAP 값으로 전체 데이터 분석
    for model_name in model_names:
        save_and_visualize_global_shap(
            model_name, 
            shap_results[model_name]['values'], 
            shap_results[model_name]['is_calculated'],
            X_no_query,  # 쿼리 제거된 데이터 사용
            feature_names,
            output_dir
        )
    
    # ROC 및 PR 곡선 오버레이
    overlay_curves(model_names, outer_folds, output_dir)
    
    return model_results, shap_results

def calculate_and_accumulate_shap(model, X_test, test_idx, model_name, fold_idx, shap_results, output_dir):
    """외부 테스트 폴드의 SHAP 값 계산 및 누적"""
    try:
        print(f"SHAP 계산 중: {model_name}, 폴드 {fold_idx} (샘플 수: {len(X_test)})")
        
        # SHAP 계산
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        
        # 저장 디렉토리 설정
        shap_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
        os.makedirs(shap_dir, exist_ok=True)
        
        # 클래스 1(양성)에 대한 SHAP 값 추출
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
            print(f"다차원 SHAP 값 감지: {shap_values.values.shape}")
            # 클래스 1(양성)의 SHAP 값만 사용
            class_idx = 1
            shap_values_class1 = shap_values.values[:, :, class_idx]
        elif hasattr(shap_values, 'values'):
            shap_values_class1 = shap_values.values
        else:
            # 이전 버전 SHAP 호환성
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # [클래스 0 SHAP, 클래스 1 SHAP] 형태
                shap_values_class1 = shap_values[1]
            else:
                shap_values_class1 = shap_values
        
        # 디버깅 정보 출력
        print(f"SHAP 값 형태: {shap_values_class1.shape}, 결과 배열 형태: {shap_results[model_name]['values'].shape[1]}")
        
        # SHAP 값 차원과 결과 배열의 차원 확인
        if shap_values_class1.shape[1] != shap_results[model_name]['values'].shape[1]:
            print(f"경고: SHAP 값 차원({shap_values_class1.shape[1]})과 결과 배열 차원({shap_results[model_name]['values'].shape[1]})이 다릅니다.")
            return False
        
        # test_idx는 이미 원본 X 데이터프레임에서의 인덱스를 직접 가리킴
        # X_test의 위치와 test_idx를 일대일 대응시킴
        for i, idx in enumerate(range(len(X_test))):
            # 원본 배열에서의 인덱스 위치
            orig_idx_pos = test_idx[idx]
            shap_results[model_name]['values'][orig_idx_pos, :] = shap_values_class1[i]
            shap_results[model_name]['is_calculated'][orig_idx_pos] = True
        
        # 폴드별 SHAP 값과 샘플 저장 (시각화 위해)
        fold_df = X_test.copy()
        fold_df.to_csv(os.path.join(shap_dir, 'shap_samples.csv'), index=True)
        
        # SHAP 값 폴드별 저장
        np.save(os.path.join(shap_dir, 'shap_values.npy'), shap_values_class1)
        
        # 폴드별 SHAP 시각화 (선택 사항)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_class1, X_test, show=False, plot_type='bar')
        plt.title(f'{model_name} - Fold {fold_idx} SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"폴드 {fold_idx}의 SHAP 값 누적 완료")
        return True
    except Exception as e:
        print(f"SHAP 계산 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def save_and_visualize_global_shap(model_name, shap_values, is_calculated, X, feature_names, output_dir):
    """누적된 SHAP 값으로 전체 데이터 분석 및 시각화"""
    # 결과 디렉토리
    global_dir = os.path.join(output_dir, 'global_shap', model_name)
    os.makedirs(global_dir, exist_ok=True)
    
    try:
        print(f"\n{model_name} 모델의 전체 SHAP 분석 시작...")
        
        # 계산된 샘플만 사용
        valid_indices = np.where(is_calculated)[0]
        if len(valid_indices) == 0:
            print(f"경고: {model_name}에 대한 유효한 SHAP 값이 없습니다.")
            return
            
        valid_shap = shap_values[valid_indices]
        valid_X = X.iloc[valid_indices]
        
        print(f"유효한 SHAP 데이터: {len(valid_indices)}/{len(X)} 샘플")
        
        # SHAP 값 저장
        np.save(os.path.join(global_dir, 'global_shap_values.npy'), valid_shap)
        valid_X.to_csv(os.path.join(global_dir, 'global_shap_samples.csv'), index=True)
        
        # 1. 전체 데이터 벌떼 플롯
        plt.figure(figsize=(14, 10))
        shap.summary_plot(valid_shap, valid_X, show=False)
        plt.title(f'{model_name} - Global SHAP Values')
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, 'global_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 특성 중요도 막대 플롯
        plt.figure(figsize=(12, 8))
        shap.summary_plot(valid_shap, valid_X, plot_type='bar', show=False)
        plt.title(f'{model_name} - Global Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, 'global_shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 평균 SHAP 값 및 표준편차 계산
        mean_shap = np.abs(valid_shap).mean(axis=0)
        std_shap = np.abs(valid_shap).std(axis=0)
        
        # 특성 중요도 DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_importance': mean_shap,
            'std_importance': std_shap
        })
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        importance_df.to_csv(os.path.join(global_dir, 'feature_importance.csv'), index=False)
        
        # 4. 상위 20개 특성 중요도 시각화 (평균 +/- 표준편차)
        top_features = importance_df['feature'][:20].values
        top_importance = importance_df['mean_importance'][:20].values
        top_std = importance_df['std_importance'][:20].values
        
        plt.figure(figsize=(12, 10))
        plt.barh(y=np.arange(len(top_features)), width=top_importance, xerr=top_std, 
                capsize=5, color='skyblue')
        plt.yticks(np.arange(len(top_features)), top_features)
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'{model_name} - Top 20 Feature Importance with Std Dev')
        plt.gca().invert_yaxis()  # 중요도 높은 특성을 위에 표시
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, 'top_features_with_std.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 상위 5개 특성에 대한 의존성 플롯
        top5_features = top_features[:5]
        for feature in top5_features:
            try:
                plt.figure(figsize=(10, 7))
                feature_idx = feature_names.index(feature)
                
                # 데이터 준비
                feature_shap = valid_shap[:, feature_idx]
                feature_value = valid_X[feature]
                
                # 산점도
                plt.scatter(feature_value, feature_shap, alpha=0.5)
                
                # 추세선
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(feature_value, feature_shap)
                x_line = np.array([min(feature_value), max(feature_value)])
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, color='red', linestyle='--')
                
                plt.xlabel(feature)
                plt.ylabel(f'SHAP value for {feature}')
                plt.title(f'{model_name} - {feature} Dependence Plot (r={r_value:.3f})')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(global_dir, f'dependence_{feature.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"{feature} 의존성 플롯 생성 중 오류: {str(e)}")
        
        print(f"{model_name} 모델의 전체 SHAP 분석 완료")
        
    except Exception as e:
        print(f"전체 SHAP 분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    args = parser.parse_args()
    
    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'classification_nested_cv_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성: {output_dir}")
    
    # CPU 코어 수 설정
    n_jobs = get_cpu_count(args.n_jobs)
    
    # 학습할 모델 목록
    model_names = [model.strip() for model in args.models.split(',')]
    print(f"학습할 모델: {', '.join(model_names)}")
    
    # 데이터 로드 및 전처리
    X, y, query_ids, scaler = load_and_preprocess_data(args.input_file, args.threshold)
    
    # 스케일러 저장
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
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
