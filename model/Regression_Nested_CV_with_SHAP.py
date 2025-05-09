#!/usr/bin/env python3
# regression_nested_cv.py

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
from collections import defaultdict
from pathlib import Path

# 모델링 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='Nested CV를 이용한 회귀 모델 비교')
parser.add_argument('--n_jobs', type=int, default=0, 
                    help='사용할 CPU 코어 수 (0=자동감지, -1=모든 코어 사용)')
parser.add_argument('--input_file', type=str, required=True,
                    help='입력 데이터 파일 경로')
parser.add_argument('--outer_folds', type=int, default=5,
                    help='외부 CV 폴드 수')
parser.add_argument('--inner_folds', type=int, default=3,
                    help='내부 CV 폴드 수')
parser.add_argument('--random_iter', type=int, default=100,
                    help='RandomizedSearchCV 반복 횟수')
parser.add_argument('--models', type=str, default='rf,lgb,xgb',
                    help='학습 및 비교할 모델 (쉼표로 구분, rf=Random Forest, lgb=LightGBM, xgb=XGBoost)')

# CPU 사용 개수 결정 함수
def get_cpu_count(requested_jobs=0):
    total_cpus = os.cpu_count() or 4
    
    if requested_jobs == 0:
        cpu_count = max(1, int(total_cpus * 0.75))
        print(f"자동 감지된 CPU 수: {total_cpus}, 사용할 코어 수: {cpu_count} (75%)")
    elif requested_jobs == -1:
        cpu_count = total_cpus
        print(f"모든 CPU 코어 사용: {cpu_count}")
    else:
        if requested_jobs > total_cpus:
            print(f"경고: 요청한 코어 수({requested_jobs})가 가용 코어 수({total_cpus})보다 많습니다.")
            cpu_count = total_cpus
        else:
            cpu_count = requested_jobs
        print(f"사용자 지정 CPU 코어 수: {cpu_count}/{total_cpus}")
    
    return cpu_count

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(input_file):
    """데이터 로드 및 전처리 (전체 스케일링 방식)"""
    print(f"데이터 파일 로드 중: {input_file}")
    df = pd.read_csv(input_file)
    print(f"원본 데이터 크기: {df.shape}")

    # 불필요한 컬럼 제거
    cols_to_drop = ['pdb', 'seed', 'sample', 'data_file', 
                   'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm',
                   'format', 'model_path', 'native_path',
                   'Fnat', 'Fnonnat', 'rRMS', 'iRMS', 'LRMS']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 결측치 처리
    initial_rows = len(df)
    df = df.dropna()
    print(f"결측치 제거 후 데이터 크기: {df.shape} ({initial_rows - len(df)}개 행 제거됨)")

    # 특성과 타겟 분리
    X = df.drop(columns=['DockQ'])
    y = df['DockQ']
    
    # 쿼리 정보 추출 (GroupKFold용)
    query_ids = df['query'].copy()
    
    # 데이터 스케일링 (전체 데이터 한 번)
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # 상관관계 분석 및 시각화 (선택 사항)
    correlation = X.drop(columns=['query'], errors='ignore').corr()
    
    return X_scaled, y, query_ids, scaler, correlation

def get_hyperparameter_grid():
    """모델별 하이퍼파라미터 탐색 범위 정의"""
    param_grids = {
        'rf': {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 1/3, 0.5, 0.7]
        },
        'lgb': {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'num_leaves': [20, 31, 50, 100],
            'max_depth': [-1, 5, 10, 15, 20],
            'min_child_samples': [10, 20, 30, 50],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1]
        },
        'xgb': {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.3, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1]
        }
    }
    return param_grids

def create_model(model_name, params, n_jobs):
    """모델 인스턴스 생성"""
    if model_name == 'rf':
        return RandomForestRegressor(random_state=42, n_jobs=n_jobs, **params)
    
    elif model_name == 'lgb':
        # LightGBM 기본 파라미터 설정
        default_params = {
            'min_child_samples': 20,  # 리프 노드의 최소 데이터 수
            'min_child_weight': 1e-3,  # 리프 노드의 최소 가중치 합
            'min_split_gain': 0,      # 분할을 위한 최소 이득
            'verbose': -1             # 경고 메시지 숨김
        }
        
        # 사용자 지정 파라미터가 기본값을 덮어쓰도록 함
        for key, value in params.items():
            default_params[key] = value
            
        return lgb.LGBMRegressor(random_state=42, n_jobs=n_jobs, **default_params)
    
    elif model_name == 'xgb':
        return xgb.XGBRegressor(
            random_state=42,
            n_jobs=n_jobs,
            **params
        )

def tune_hyperparameters(X_train, y_train, groups_train, model_name, 
                         inner_folds, n_iter, n_jobs, random_state=42):
    """내부 CV에서 하이퍼파라미터 튜닝"""
    param_grids = get_hyperparameter_grid()
    param_grid = param_grids[model_name]
    
    # 기본 모델 생성
    base_model = create_model(model_name, {}, n_jobs=1)
    
    # 내부 CV 생성 (query 기반)
    inner_cv = GroupKFold(n_splits=inner_folds)
    
    # RandomizedSearchCV 설정
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=inner_cv.split(X_train, y_train, groups=groups_train),
        scoring='neg_mean_squared_error',
        n_jobs=max(1, n_jobs // 2),
        verbose=1,
        random_state=random_state
    )
    
    # 하이퍼파라미터 탐색
    random_search.fit(X_train, y_train, groups=groups_train)
    
    # 최적 모델 및 파라미터 반환
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def calculate_metrics(y_true, y_pred):
    """평가 지표 계산"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def calculate_and_save_shap(model, X_test, model_name, fold_idx, output_dir):
    """SHAP 값 계산 및 저장 - 모든 데이터 사용"""
    try:
        if len(X_test) > 2000:  # 1000은 적절한 임계값으로 조정 가능
            X_sample = X_test.sample(2000, random_state=42).copy()
        else:
            X_sample = X_test.copy()
        
        # SHAP 계산
        print(f"SHAP 계산 중: {model_name}, fold {fold_idx} (샘플 수: {len(X_sample)})")
        if model_name == 'rf':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)
            print(f"RF SHAP 계산 완료: {type(shap_values)}")
        elif model_name == 'lgb':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)
            print(f"LightGBM SHAP 계산 완료: {type(shap_values)}")
        elif model_name == 'xgb':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)
            print(f"XGBoost SHAP 계산 완료: {type(shap_values)}")
        
        # SHAP 값 저장
        shap_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
        os.makedirs(shap_dir, exist_ok=True)
        
        # SHAP 값과 데이터 저장
        print(f"SHAP 값 저장 중: {os.path.join(shap_dir, 'shap_values.pkl')}")
        joblib.dump(shap_values, os.path.join(shap_dir, 'shap_values.pkl'))
        X_sample.to_csv(os.path.join(shap_dir, 'shap_samples.csv'), index=False)
        
        # SHAP 요약 플롯 저장 (기본 beeswarm plot)
        print("SHAP beeswarm 플롯 생성 중...")
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, 'shap_beeswarm.png'), dpi=300, bbox_inches='tight')
            print(f"Beeswarm 플롯 저장 완료: {os.path.join(shap_dir, 'shap_beeswarm.png')}")
        except Exception as plot_err:
            print(f"Beeswarm 플롯 생성 중 오류 발생: {str(plot_err)}")
        plt.close()
        
        # SHAP 막대 플롯 저장
        print("SHAP bar 플롯 생성 중...")
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
            print(f"Bar 플롯 저장 완료: {os.path.join(shap_dir, 'shap_bar.png')}")
        except Exception as plot_err:
            print(f"Bar 플롯 생성 중 오류 발생: {str(plot_err)}")
        plt.close()
        
        return True
    except Exception as e:
        print(f"SHAP 계산 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()  # 상세 오류 정보 출력
        return False

def save_fold_results(model, X_test, y_test, y_pred, metrics, best_params, 
                      model_name, fold_idx, output_dir):
    """각 폴드의 결과 저장"""
    # 결과 디렉토리 생성
    fold_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
    os.makedirs(fold_dir, exist_ok=True)
    
    # 모델 저장
    model_path = os.path.join(fold_dir, 'best_model.pkl')
    joblib.dump(model, model_path)
    
    # 예측값 저장
    pred_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    pred_df.to_csv(os.path.join(fold_dir, 'predictions.csv'), index=False)
    
    # 평가 지표 저장
    pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)
    
    # 하이퍼파라미터 저장
    with open(os.path.join(fold_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 특성 중요도 계산 및 저장 (가능한 경우)
    try:
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif model_name == 'lgb' and hasattr(model, 'booster_'):
            importances = model.booster_.feature_importance(importance_type='gain')
        elif model_name == 'xgb' and hasattr(model, 'get_booster'):
            importances = model.get_booster().get_score(importance_type='gain')
            importances = [importances.get(f, 0) for f in X_test.columns]
        
        if importances is not None:
            importance_df = pd.DataFrame({
                'feature': X_test.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            importance_df.to_csv(os.path.join(fold_dir, 'feature_importance.csv'), index=False)
    except Exception as e:
        print(f"특성 중요도 계산 중 오류 발생: {str(e)}")

def combine_shap_values(model_names, n_folds, output_dir):
    """모든 폴드의 SHAP 값 통합 및 시각화"""
    for model_name in model_names:
        print(f"{model_name} 모델의 SHAP 값 통합 중...")
        
        # 모든 폴드의 SHAP 값 특성 중요도만 먼저 추출
        abs_vals = []
        feature_names = None
        
        # 통합 SHAP 계산을 위한 데이터
        all_shap_raw_values = []  # 원시 SHAP 값만 저장
        all_samples = []
        
        for fold_idx in range(n_folds):
            shap_dir = os.path.join(output_dir, f'fold_{fold_idx}', model_name)
            shap_file = os.path.join(shap_dir, 'shap_values.pkl')
            sample_file = os.path.join(shap_dir, 'shap_samples.csv')
            
            if os.path.exists(shap_file) and os.path.exists(sample_file):
                try:
                    # 샘플 데이터 로드
                    samples = pd.read_csv(sample_file)
                    if feature_names is None:
                        feature_names = samples.columns.tolist()
                    
                    # SHAP 값 로드
                    print(f"폴드 {fold_idx}의 SHAP 값 로드 중: {shap_file}")
                    shap_values = joblib.load(shap_file)
                    
                    # 특성 중요도 계산과 원시 SHAP 값 추출
                    if hasattr(shap_values, 'values'):
                        abs_val = np.abs(shap_values.values).mean(0)
                        all_shap_raw_values.append(shap_values.values)  # 원시 값만 저장
                    else:
                        abs_val = np.abs(shap_values).mean(0)
                        all_shap_raw_values.append(shap_values)  # 이미 원시 값
                    
                    abs_vals.append(abs_val)
                    all_samples.append(samples)
                    print(f"폴드 {fold_idx}의 SHAP 값 로드 완료")
                except Exception as e:
                    print(f"폴드 {fold_idx}의 SHAP 값 로드 중 오류: {str(e)}")
        
        if not abs_vals:
            print(f"SHAP 값을 찾을 수 없음: {model_name}")
            continue
        
        # 통합 SHAP 분석 디렉토리
        combined_dir = os.path.join(output_dir, 'combined_shap', model_name)
        os.makedirs(combined_dir, exist_ok=True)
        
        try:
            # 평균 절대 SHAP 값 계산
            mean_abs_vals = np.mean(abs_vals, axis=0)
            
            # 특성 중요도 DataFrame 생성 및 저장
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'mean_importance': mean_abs_vals
            }).sort_values('mean_importance', ascending=False)
            
            importance_df.to_csv(os.path.join(combined_dir, 'shap_importance.csv'), index=False)
            
            # 1. 통합 SHAP 막대 플롯 (Bar Plot)
            print(f"통합 SHAP bar 플롯 생성 중: {model_name}")
            plt.figure(figsize=(12, 8))
            plt.barh(
                y=importance_df['feature'][:20],  # 상위 20개 특성
                width=importance_df['mean_importance'][:20],
                color='skyblue'
            )
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'{model_name} - Mean Feature Importance (Across All Folds)')
            plt.gca().invert_yaxis()  # 중요도 높은 순으로 정렬
            plt.tight_layout()
            plt.savefig(os.path.join(combined_dir, 'combined_shap_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 모든 폴드를 통합한 SHAP 벌떼 플롯 생성
            print(f"모든 폴드의 SHAP 값을 통합한 벌떼 플롯 생성 중: {model_name}")
            
            # 모든 샘플 데이터 통합
            combined_samples = pd.concat(all_samples, ignore_index=True)
            
            # 모든 원시 SHAP 값 통합 (이제 vstack이 잘 작동해야 함)
            try:
                combined_shap_values = np.vstack(all_shap_raw_values)
                print(f"통합된 SHAP 값 형태: {combined_shap_values.shape}, 샘플 형태: {combined_samples.shape}")
                
                # 상위 20개 특성만 선택
                top_features = importance_df['feature'][:20].tolist()
                top_indices = [feature_names.index(f) for f in top_features]
                
                # 벌떼 플롯 생성
                plt.figure(figsize=(14, 10))
                
                # shap.summary_plot은 특성 이름이 있어야 작동함
                shap.summary_plot(
                    combined_shap_values[:, top_indices],  # 상위 특성만 선택
                    combined_samples[top_features],        # 상위 특성 데이터만 선택
                    feature_names=top_features,
                    max_display=20,
                    show=False
                )
                plt.title(f'{model_name} - Combined SHAP Beeswarm Plot (All Folds)')
                plt.tight_layout()
                combined_beeswarm_path = os.path.join(combined_dir, 'combined_shap_beeswarm.png')
                plt.savefig(combined_beeswarm_path, dpi=300, bbox_inches='tight')
                print(f"통합 SHAP beeswarm 플롯 저장 완료: {combined_beeswarm_path}")
                plt.close()
                
            except Exception as e:
                print(f"SHAP 값 통합 중 오류 발생: {str(e)}. 첫 번째 폴드만 사용합니다.")
                import traceback
                traceback.print_exc()
                
                # 오류 발생 시 첫 번째 폴드만 사용하는 대체 코드
                plt.figure(figsize=(14, 10))
                if hasattr(joblib.load(os.path.join(output_dir, f'fold_0', model_name, 'shap_values.pkl')), 'values'):
                    shap_values = joblib.load(os.path.join(output_dir, f'fold_0', model_name, 'shap_values.pkl'))
                    shap.summary_plot(
                        shap_values,
                        max_display=20,
                        show=False
                    )
                else:
                    shap_values = joblib.load(os.path.join(output_dir, f'fold_0', model_name, 'shap_values.pkl'))
                    samples = pd.read_csv(os.path.join(output_dir, f'fold_0', model_name, 'shap_samples.csv'))
                    top_features = importance_df['feature'][:20].tolist()
                    shap.summary_plot(
                        shap_values,
                        samples[top_features],
                        feature_names=top_features,
                        max_display=20,
                        show=False
                    )
                plt.title(f'{model_name} - SHAP Beeswarm Plot (First Fold Only)')
                plt.tight_layout()
                plt.savefig(os.path.join(combined_dir, 'combined_shap_beeswarm_fallback.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 메모리 정리
            del all_shap_raw_values
            del all_samples
            
        except Exception as e:
            print(f"SHAP 값 통합 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()

def compare_models(model_results, output_dir):
    """모델 성능 비교 분석"""
    print("\n=== 모델 성능 비교 분석 ===")
    
    # 모델별 평균 성능 지표 계산
    metrics_summary = {}
    
    for model_name, fold_results in model_results.items():
        metrics_summary[model_name] = {
            'MSE': [],
            'RMSE': [],
            'MAE': [],
            'R2': []
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
    
    # 빈 DataFrame 경고 해결
    if comparison_data:
        metrics_comparison = pd.DataFrame(comparison_data)
    else:
        metrics_comparison = pd.DataFrame(columns=['Model', 'Metric', 'Mean', 'Std'])
    
    # 평가 지표 테이블 저장
    metrics_table = os.path.join(output_dir, 'metrics_summary.csv')
    metrics_comparison.to_csv(metrics_table, index=False)
    print(f"평가 지표 요약 테이블 저장: {metrics_table}")
    
    # 평가 지표 시각화
    plt.figure(figsize=(14, 10))
    
    # 지표별 모델 비교
    for i, metric in enumerate(['MSE', 'RMSE', 'MAE', 'R2']):
        plt.subplot(2, 2, i+1)
        
        metric_data = metrics_comparison[metrics_comparison['Metric'] == metric]
        
        # 막대그래프와 오차 막대
        plt.bar(
            metric_data['Model'],
            metric_data['Mean'],
            yerr=metric_data['Std'],
            capsize=5,
            color=['blue', 'green', 'red'][:len(metric_data)]
        )
        
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()
    
    # 모델별 R² 박스플롯
    plt.figure(figsize=(10, 6))
    
    boxplot_data = []
    for model_name, metrics in metrics_summary.items():
        for r2 in metrics['R2']:
            boxplot_data.append({
                'Model': model_name,
                'R2': r2
            })
    
    boxplot_df = pd.DataFrame(boxplot_data)
    sns.boxplot(x='Model', y='R2', data=boxplot_df)
    plt.title('R² Scores Across Folds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_boxplot.png'), dpi=300)
    plt.close()
    
    return metrics_comparison

def run_nested_cv(X, y, query_ids, model_names, output_dir, outer_folds=5, inner_folds=3, 
                 n_iter=100, n_jobs=4):
    """Nested CV 실행"""
    print(f"Nested CV 시작 (외부 폴드: {outer_folds}, 내부 폴드: {inner_folds}, 반복 횟수: {n_iter})")
    
    # 외부 CV 폴드 (쿼리 기반)
    outer_cv = GroupKFold(n_splits=outer_folds)
    
    # 모델별 결과 저장
    model_results = {model: {'metrics': [], 'predictions': []} for model in model_names}
    
    # 외부 CV 루프
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=query_ids)):
        print(f"\n=== 외부 폴드 {fold_idx+1}/{outer_folds} ===")
        
        # 훈련 데이터와 테스트 데이터 분할
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        query_train = query_ids.iloc[train_idx]
        
        # 데이터 전처리 (이미 스케일링됨)
        # query 열 제거 (모델에 사용하지 않음)
        if 'query' in X_train.columns:
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
                
                # 평가 지표 계산
                metrics = calculate_metrics(y_test, y_pred)
                
                # 결과 저장
                model_results[model_name]['metrics'].append(metrics)
                model_results[model_name]['predictions'].append((y_test, y_pred))
                
                # 폴드별 결과 저장
                save_fold_results(
                    best_model, X_test_no_query, y_test, y_pred, 
                    metrics, best_params, model_name, fold_idx, output_dir
                )
                
                # SHAP 값 계산 및 저장 (명시적으로 여기서 호출)
                print(f"SHAP 값 계산 및 저장 중: {model_name}, 폴드 {fold_idx+1}")
                shap_success = calculate_and_save_shap(
                    best_model, X_test_no_query, model_name, fold_idx, output_dir
                )
                if shap_success:
                    print(f"SHAP 값 저장 완료: {model_name}, 폴드 {fold_idx+1}")
                else:
                    print(f"SHAP 값 저장 실패: {model_name}, 폴드 {fold_idx+1}")
                
                # 처리 시간 출력
                elapsed_time = time.time() - start_time
                print(f"{model_name.upper()} 폴드 {fold_idx+1} 완료: "
                      f"R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                      f"처리 시간: {elapsed_time:.2f}초")
                
            except Exception as e:
                print(f"{model_name.upper()} 모델 처리 중 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()  # 상세 오류 정보 출력
    
    # 모델 비교 분석
    metrics_comparison = compare_models(model_results, output_dir)
    
    # SHAP 값 통합 및 시각화
    combine_shap_values(model_names, outer_folds, output_dir)
    
    return model_results, metrics_comparison

def main():
    args = parser.parse_args()
    
    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'regression_nested_cv_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성: {output_dir}")
    
    # CPU 코어 수 설정
    n_jobs = get_cpu_count(args.n_jobs)
    
    # 학습할 모델 목록
    model_names = [model.strip() for model in args.models.split(',')]
    print(f"학습할 모델: {', '.join(model_names)}")
    
    # 데이터 로드 및 전처리
    X, y, query_ids, scaler, correlation = load_and_preprocess_data(args.input_file)
    
    # 상관관계 행렬 저장
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 스케일러 저장
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Nested CV 실행
    model_results, metrics_comparison = run_nested_cv(
        X, y, query_ids, model_names, output_dir,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        n_iter=args.random_iter,
        n_jobs=n_jobs
    )
    
    print(f"\n분석 완료. 결과는 {output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
