#!/usr/bin/env python3
# nested_cv_model_comparison.py

import pandas as pd
import numpy as np
import os
import joblib
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import shap

# 모델링 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

# 평가 지표
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, average_precision_score, matthews_corrcoef,
                            log_loss, confusion_matrix, classification_report,
                            roc_curve, precision_recall_curve, auc)

# 명령줄 인자 처리
import argparse

# 설정
CLASS_NAMES = ['Incorrect', 'Acceptable+']
CLASS_THRESHOLD = 0.23

def parse_arguments():
    parser = argparse.ArgumentParser(description='Nested CV를 이용한 모델 비교')
    parser.add_argument('--input_file', type=str, required=True, help='입력 데이터 파일 경로')
    parser.add_argument('--n_jobs', type=int, default=0, help='사용할 CPU 코어 수 (0=자동감지, -1=모든 코어)')
    parser.add_argument('--outer_folds', type=int, default=5, help='외부 CV 폴드 수')
    parser.add_argument('--inner_folds', type=int, default=3, help='내부 CV 폴드 수')
    parser.add_argument('--models', type=str, default='rf,lgb,xgb', help='학습할 모델들 (쉼표로 구분)')
    return parser.parse_args()

def get_cpu_count(requested_jobs=0):
    total_cpus = os.cpu_count() or 4
    
    if requested_jobs == 0:  # 자동 감지 (75% 사용)
        cpu_count = max(1, int(total_cpus * 0.75))
        print(f"자동 감지된 CPU 수: {total_cpus}, 사용할 코어 수: {cpu_count} (75%)")
    elif requested_jobs == -1:  # 모든 코어
        cpu_count = total_cpus
        print(f"모든 CPU 코어 사용: {cpu_count}")
    else:  # 사용자 지정
        cpu_count = min(requested_jobs, total_cpus)
        print(f"사용자 지정 CPU 코어 수: {cpu_count}/{total_cpus}")
    
    return cpu_count

def load_and_preprocess_data(file_path):
    """데이터 로드 및 전처리"""
    print(f"데이터 파일 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    print(f"원본 데이터 크기: {df.shape}")
    
    # DockQ 값을 기준으로 이진 클래스 레이블 생성
    df['DockQ_orig'] = df['DockQ'].copy()
    df['DockQ'] = df['DockQ_orig'].apply(lambda x: 1 if x >= CLASS_THRESHOLD else 0)
    
    # 클래스 분포 확인
    class_counts = df['DockQ'].value_counts().sort_index()
    print("클래스 분포:")
    for i, class_name in enumerate(CLASS_NAMES):
        threshold_str = f"< {CLASS_THRESHOLD}" if i == 0 else f">= {CLASS_THRESHOLD}"
        count = class_counts.get(i, 0)
        percent = 100 * count / len(df) if len(df) > 0 else 0
        print(f"  Class {i} ({class_name}, DockQ {threshold_str}): {count} 샘플 ({percent:.2f}%)")
    
    # 불필요한 컬럼 제거
    cols_to_drop = ['pdb', 'seed', 'sample', 'data_file', 
                   'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm',
                   'format', 'model_path', 'native_path',
                   'Fnat', 'Fnonnat', 'rRMS', 'iRMS', 'LRMS']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 결측치 처리
    df = df.dropna()
    
    # 특성과 레이블 분리
    X = df.drop(columns=['DockQ', 'DockQ_orig'])
    y = df['DockQ']
    
    # 쿼리 정보 추출
    query_ids = X['query'].copy()
    
    return X, y, query_ids

def get_model_configs(n_jobs):
    """모델별 설정 정의"""
    # RandomForest 설정
    rf_params = {
        'param_grid': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
    }
    
    # LightGBM 설정
    lgb_params = {
        'param_grid': {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63],
            'max_depth': [5, -1],
            'min_child_samples': [20, 50],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'class_weight': [None, 'balanced'],
            # 경고 메시지 방지를 위한 추가 파라미터
            'min_gain_to_split': [0.0001],
            'min_data_in_leaf': [10],
            'verbose': [-1]  # 경고 메시지 숨김
        },
        'fit_params': {
            'verbose': False
        }
    }
    
    # XGBoost 설정
    xgb_params = {
        'param_grid': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1],
            'scale_pos_weight': [1, 3]
        },
        'fit_params': {
            'num_boost_round': 500,
            'early_stopping_rounds': 50,
            'verbose_eval': False
        }
    }
    
    return {
        'rf': rf_params,
        'lgb': lgb_params,
        'xgb': xgb_params
    }

def create_model(model_name, params, n_jobs):
    """모델 인스턴스 생성"""
    if model_name == 'rf':
        return RandomForestClassifier(random_state=42, n_jobs=n_jobs, **params)
    
    elif model_name == 'lgb':
        # verbose 파라미터가 명시적으로 설정되지 않은 경우 기본값으로 추가
        if 'verbose' not in params:
            params['verbose'] = -1  # 경고 메시지 숨김
        return lgb.LGBMClassifier(random_state=42, n_jobs=n_jobs, **params)
    
    elif model_name == 'xgb':
        # 스케일링된 파라미터로 XGBoost 모델 생성
        return xgb.XGBClassifier(
            random_state=42,
            n_jobs=n_jobs,
            use_label_encoder=False,
            eval_metric='auc',
            **params
        )

def tune_hyperparams(model_name, X_train, y_train, query_ids_train, inner_folds, model_config, n_jobs):
    """내부 CV에서 하이퍼파라미터 튜닝"""
    print(f"{model_name.upper()} 모델의 하이퍼파라미터 튜닝 중...")
    
    # 내부 CV 폴드 생성 (쿼리 기반)
    cv = GroupKFold(n_splits=inner_folds)
    param_grid = model_config['param_grid']
    
    if model_name == 'rf' or model_name == 'lgb':
        # sklearn 호환 API 사용
        from sklearn.model_selection import GridSearchCV
        
        base_model = create_model(model_name, {}, n_jobs=1)
        
        # GridSearchCV 설정
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=list(cv.split(X_train, y_train, groups=query_ids_train)),
            scoring='f1',
            n_jobs=max(1, n_jobs // 2),
            verbose=1
        )
        
        # 최적 모델 찾기
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"최적 {model_name.upper()} 파라미터: {best_params}")
        print(f"내부 CV 최고 F1 점수: {best_score:.4f}")
        
        # 최적 모델 생성
        best_model = create_model(model_name, best_params, n_jobs)
        
        # LightGBM인 경우 추가 파라미터 설정
        if model_name == 'lgb' and 'fit_params' in model_config:
            # 최적 파라미터로 모델 학습
            X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # 새로운 fit_params 딕셔너리 생성 (기존 것을 사용하지 않음)
            fit_params = {
                'eval_set': [(X_v, y_v)]
                # verbose 제거 - 모델 생성 시에만 사용해야 함
            }
            
            # early_stopping 콜백 추가
            from lightgbm.callback import early_stopping
            fit_params['callbacks'] = [early_stopping(50, verbose=False)]
            
            # 모델 학습
            best_model.fit(X_t, y_t, **fit_params)
        else:
            # RandomForest 등 일반 모델
            best_model.fit(X_train, y_train)
            
        return best_model, best_params, best_score
        
    elif model_name == 'xgb':
        # XGBoost 네이티브 API 사용
        from itertools import product
        
        # 파라미터 조합 생성
        param_keys = param_grid.keys()
        param_values = param_grid.values()
        param_combinations = [dict(zip(param_keys, combo)) for combo in product(*param_values)]
        
        # 기본 파라미터 설정
        dtrain = xgb.DMatrix(X_train, label=y_train)
        best_score = -float('inf')
        best_params = None
        best_booster = None
        
        # 각 파라미터 조합 테스트
        for params in param_combinations:
            # 모든 파라미터 설정
            full_params = params.copy()
            full_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'nthread': max(1, n_jobs // 2)
            })
            
            # 교차 검증 수행
            cv_results = xgb.cv(
                params=full_params,
                dtrain=dtrain,
                num_boost_round=100,
                nfold=inner_folds,
                stratified=True,
                metrics='auc',
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # 최적 AUC 점수와 반복 횟수 추출
            best_round = cv_results.shape[0]
            mean_auc = cv_results.iloc[best_round-1]['test-auc-mean']
            
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = full_params
                best_iteration = best_round
        
        print(f"최적 XGB 파라미터: {best_params}")
        print(f"내부 CV 최고 AUC 점수: {best_score:.4f}")
        print(f"최적 반복 횟수: {best_iteration}")
        
        # 최종 모델 학습
        best_booster = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=best_iteration
        )
        
        return best_booster, best_params, best_score

def evaluate_model(model_name, model, X_test, y_test):
    """모델 성능 평가"""
    # 예측 생성
    if model_name == 'xgb':
        # XGBoost 네이티브 API
        dtest = xgb.DMatrix(X_test)
        y_proba = model.predict(dtest)
        y_pred = (y_proba > 0.5).astype(int)
    else:
        # sklearn 호환 API
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # 성능 지표 계산
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'log_loss': log_loss(y_test, y_proba)
    }
    
    # 특성 중요도 추출
    if model_name == 'rf':
        importances = {
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }
    elif model_name == 'lgb':
        importances = model.booster_.feature_importance(importance_type='gain')
        importances = {
            'feature': X_test.columns,
            'importance': importances
        }
    elif model_name == 'xgb':
        importances = model.get_score(importance_type='gain')
        # XGBoost는 중요도가 딕셔너리 형태로 반환됨
        importances = {
            'feature': list(importances.keys()),
            'importance': list(importances.values())
        }
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': metrics,
        'importances': importances,
        'confusion_matrix': cm
    }

def analyze_shap(model_name, model, X_test_sample):
    """SHAP 분석 수행"""
    try:
        print(f"{model_name.upper()} 모델의 SHAP 분석 중...")
        
        if model_name == 'xgb':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
        
        # SHAP 값 계산
        shap_values = explainer(X_test_sample)
        
        # 다차원 SHAP 값 처리
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
            # 클래스 1의 SHAP 값만 추출
            class_idx = 1
            values = shap_values.values[:, :, class_idx]
            base_values = shap_values.base_values
            if isinstance(base_values, np.ndarray) and len(base_values.shape) > 1:
                base_values = base_values[:, class_idx]
            
            # 새 SHAP 객체 생성
            shap_values = shap.Explanation(
                values=values,
                base_values=base_values,
                data=shap_values.data,
                feature_names=shap_values.feature_names
            )
        
        return shap_values
    
    except Exception as e:
        print(f"SHAP 분석 중 오류 발생: {str(e)}")
        return None

def save_fold_results(fold_idx, model_name, results, output_dir):
    """각 폴드별 결과 저장"""
    fold_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
    os.makedirs(fold_dir, exist_ok=True)
    
    # 예측값 저장
    pd.DataFrame({
        'actual': results['y_test'],
        'predicted': results['y_pred'],
        'probability': results['y_proba']
    }).to_csv(os.path.join(fold_dir, 'predictions.csv'), index=False)
    
    # 성능 지표 저장
    pd.DataFrame([results['metrics']]).to_csv(
        os.path.join(fold_dir, 'metrics.csv'), index=False)
    
    # 특성 중요도 저장
    pd.DataFrame(results['importances']).to_csv(
        os.path.join(fold_dir, 'feature_importance.csv'), index=False)
    
    # 혼동 행렬 저장
    pd.DataFrame(results['confusion_matrix']).to_csv(
        os.path.join(fold_dir, 'confusion_matrix.csv'), index=False)
    
    # 하이퍼파라미터 저장
    with open(os.path.join(fold_dir, 'best_params.json'), 'w') as f:
        json.dump(results['best_params'], f, indent=2)
        
    # SHAP 플롯 저장 (추가)
    if results['shap_values'] is not None:
        try:
            # SHAP summary 플롯
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                results['shap_values'], 
                results['shap_sample'],
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(fold_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # SHAP bar 플롯
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                results['shap_values'], 
                results['shap_sample'],
                plot_type='bar',
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(fold_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP 플롯 저장 완료: {fold_dir}")
        except Exception as e:
            print(f"SHAP 플롯 저장 중 오류 발생: {str(e)}")

def create_summary_plots(all_results, model_names, output_dir):
    """모델별 요약 시각화"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = os.path.join(output_dir, 'summary_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 성능 지표 비교 박스플롯
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'mcc']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        metric_data = []
        
        for model in model_names:
            model_metric = [results['metrics'][metric] for results in all_results[model]]
            metric_data.append(model_metric)
        
        plt.boxplot(metric_data, labels=model_names, 
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    medianprops=dict(color='red'))
        
        # 개별 데이터 포인트 표시
        for i, data in enumerate(metric_data, 1):
            plt.scatter([i] * len(data), data, color='black', alpha=0.5)
        
        plt.title(f'{metric.upper()} Comparison Across Models')
        plt.ylabel(metric.upper())
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, f'{metric}_comparison_{timestamp}.png'), dpi=300)
        plt.close()
    
    # 평균 성능 지표 테이블 생성
    summary_data = {model: {} for model in model_names}
    
    for model in model_names:
        for metric in metrics_to_plot:
            values = [results['metrics'][metric] for results in all_results[model]]
            summary_data[model][f'{metric}_mean'] = np.mean(values)
            summary_data[model][f'{metric}_std'] = np.std(values)
    
    # 요약 테이블 저장
    summary_df = pd.DataFrame(summary_data).T
    summary_df.to_csv(os.path.join(output_dir, f'model_comparison_summary_{timestamp}.csv'))
    
    # SHAP 중요도 종합 비교 (추가)
    try:
        # SHAP 중요도 데이터 수집
        shap_importance_data = {}
        for model_name in model_names:
            # 각 모델에 대한 폴드별 SHAP 중요도 값 
            fold_shap_values = []
            fold_feature_names = []
            
            for fold_results in all_results[model_name]:
                if fold_results['shap_values'] is not None:
                    # SHAP 값이 있는 경우만 처리
                    if hasattr(fold_results['shap_values'], 'values'):
                        # shap 값과 특성 이름 추출
                        values = np.abs(fold_results['shap_values'].values).mean(0)
                        feature_names = fold_results['shap_values'].feature_names
                        
                        if values is not None and feature_names is not None:
                            fold_shap_values.append(values)
                            fold_feature_names.append(feature_names)
            
            # 모든 폴드에서 SHAP 값이 있는 경우 평균 계산
            if fold_shap_values and len(set(tuple(names) for names in fold_feature_names)) == 1:
                feature_names = fold_feature_names[0]
                mean_importance = np.mean(fold_shap_values, axis=0)
                
                # 모델별 SHAP 중요도 저장
                shap_importance_data[model_name] = {
                    'feature': feature_names,
                    'importance': mean_importance
                }
        
        # 모든 모델에 대해 SHAP 중요도가 있는 경우 비교 플롯 생성
        if len(shap_importance_data) == len(model_names):
            # 상위 20개 특성 추출
            all_features = set()
            for model_name, data in shap_importance_data.items():
                # 중요도 기준 상위 20개 특성
                top_idx = np.argsort(data['importance'])[-20:]
                all_features.update([data['feature'][i] for i in top_idx])
            
            # 모든 모델의 상위 특성에 대한 중요도 비교
            all_features = list(all_features)
            comparison_data = []
            
            for feature in all_features:
                feature_data = {'feature': feature}
                for model_name, data in shap_importance_data.items():
                    if feature in data['feature']:
                        idx = list(data['feature']).index(feature)
                        feature_data[model_name] = data['importance'][idx]
                    else:
                        feature_data[model_name] = 0
                comparison_data.append(feature_data)
            
            # 데이터프레임으로 변환
            comparison_df = pd.DataFrame(comparison_data)
            
            # 합계로 정렬하고 상위 20개 추출
            comparison_df['total'] = comparison_df.drop('feature', axis=1).sum(axis=1)
            comparison_df = comparison_df.sort_values('total', ascending=False).head(20)
            comparison_df = comparison_df.drop('total', axis=1)
            
            # SHAP 중요도 비교 바 차트
            plt.figure(figsize=(12, 10))
            comparison_df.set_index('feature').plot(kind='barh', figsize=(12, 10))
            plt.title('SHAP Feature Importance Comparison Across Models')
            plt.xlabel('Mean |SHAP value|')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'shap_importance_comparison_{timestamp}.png'), dpi=300)
            plt.close()
            
            print(f"모델 간 SHAP 중요도 비교 저장: {os.path.join(plots_dir, f'shap_importance_comparison_{timestamp}.png')}")
        else:
            print("일부 모델의 SHAP 값이 계산되지 않아 SHAP 중요도 비교를 생성할 수 없습니다.")
    except Exception as e:
        print(f"SHAP 중요도 비교 생성 중 오류 발생: {str(e)}")
    
    print(f"모델 성능 요약:\n{summary_df}")
    return summary_df

def run_nested_cv(X, y, query_ids, model_names, output_dir, outer_folds=5, inner_folds=3, n_jobs=4):
    """Nested CV 실행"""
    print(f"Nested CV 시작 (외부 폴드: {outer_folds}, 내부 폴드: {inner_folds})")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 전체 결과 저장 변수
    all_results = {model: [] for model in model_names}
    
    # 모델 설정 불러오기
    model_configs = get_model_configs(n_jobs)
    
    # 외부 CV 폴드 (쿼리 기반)
    outer_cv = GroupKFold(n_splits=outer_folds)
    
    # 데이터 스케일링 준비
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    
    # 외부 CV 루프
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=query_ids)):
        print(f"\n외부 폴드 {fold_idx+1}/{outer_folds} 시작")
        
        # 폴드별 데이터 분할
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        query_ids_train = query_ids.iloc[train_idx]
        
        # 데이터 스케일링
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
        
        # 모델에서 쿼리 열 제외
        if 'query' in X_train_scaled.columns:
            X_train_no_query = X_train_scaled.drop(columns=['query']) 
            X_test_no_query = X_test_scaled.drop(columns=['query'])
        else:
            X_train_no_query = X_train_scaled
            X_test_no_query = X_test_scaled
        
        # 각 모델 학습 및 평가
        for model_name in model_names:
            print(f"\n{model_name.upper()} 모델 처리 중 (폴드 {fold_idx+1})")
            fold_start_time = time.time()
            
            # 내부 CV로 하이퍼파라미터 최적화
            best_model, best_params, best_score = tune_hyperparams(
                model_name, X_train_no_query, y_train, query_ids_train, 
                inner_folds, model_configs[model_name], n_jobs
            )
            
            # 최적 모델로 테스트 세트 평가
            evaluation = evaluate_model(model_name, best_model, X_test_no_query, y_test)
            
            # SHAP 분석 (샘플)
            shap_sample_size = min(100, len(X_test_no_query))
            X_test_sample = X_test_no_query.iloc[:shap_sample_size]
            shap_values = analyze_shap(model_name, best_model, X_test_sample)
            
            # 처리 시간 계산
            fold_time = time.time() - fold_start_time
            
            # 폴드 결과 저장
            fold_results = {
                'model': best_model,
                'best_params': best_params,
                'best_score': best_score,
                'y_test': y_test,
                'y_pred': evaluation['y_pred'],
                'y_proba': evaluation['y_proba'],
                'metrics': evaluation['metrics'],
                'importances': evaluation['importances'],
                'confusion_matrix': evaluation['confusion_matrix'],
                'shap_values': shap_values,
                'shap_sample': X_test_sample,
                'processing_time': fold_time
            }
            
            # 전체 결과 업데이트
            all_results[model_name].append(fold_results)
            
            # 폴드별 결과 저장
            save_fold_results(fold_idx, model_name, fold_results, output_dir)
            
            print(f"{model_name.upper()} 폴드 {fold_idx+1} 완료: "
                  f"F1={fold_results['metrics']['f1']:.4f}, "
                  f"ROC-AUC={fold_results['metrics']['roc_auc']:.4f}, "
                  f"PR-AUC={fold_results['metrics']['pr_auc']:.4f}, "
                  f"시간={fold_time:.2f}초")
    
    # 종합 결과 분석 및 시각화
    summary_df = create_summary_plots(all_results, model_names, output_dir)
    
    # 최고 성능 모델 선택 및 전체 데이터로 최종 학습
    best_model_overall = select_best_model(summary_df, model_names)
    print(f"\n전체 데이터에서 최고 성능 모델: {best_model_overall}")
    
    return all_results, summary_df, best_model_overall

def select_best_model(summary_df, model_names):
    """가장 좋은 평균 성능을 보인 모델 선택"""
    # 기본적으로 F1 점수 기준으로 최고 모델 선택
    best_model = summary_df['f1_mean'].idxmax()
    print(f"F1 점수 기준 최고 모델: {best_model} (F1={summary_df.loc[best_model, 'f1_mean']:.4f})")
    
    # 추가로 다른 지표도 출력
    print("모델별 주요 지표 (평균):")
    for model in model_names:
        print(f"  {model}: "
              f"F1={summary_df.loc[model, 'f1_mean']:.4f}, "
              f"ROC-AUC={summary_df.loc[model, 'roc_auc_mean']:.4f}, "
              f"PR-AUC={summary_df.loc[model, 'pr_auc_mean']:.4f}")
    
    return best_model

def main():
    args = parse_arguments()
    
    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'nested_cv_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성: {output_dir}")
    
    # CPU 코어 수 설정
    n_jobs = get_cpu_count(args.n_jobs)
    
    # 학습할 모델
    model_names = [model.strip().lower() for model in args.models.split(',')]
    print(f"학습할 모델: {', '.join(model_names)}")
    
    # 데이터 로드 및 전처리
    X, y, query_ids = load_and_preprocess_data(args.input_file)
    
    # Nested CV 수행
    results, summary, best_model = run_nested_cv(
        X, y, query_ids, model_names, output_dir, 
        outer_folds=args.outer_folds, 
        inner_folds=args.inner_folds,
        n_jobs=n_jobs
    )
    
    print(f"\n분석 완료. 결과는 {output_dir} 디렉토리에 저장되었습니다.")
    print(f"최고 성능 모델: {best_model}")

if __name__ == "__main__":
    main()
