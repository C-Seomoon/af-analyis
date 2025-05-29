import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from datetime import datetime
import shap
import time

# lightgbm과 xgboost 추가
import lightgbm as lgb
import xgboost as xgb

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='회귀 모델 비교 학습')
parser.add_argument('--n_jobs', type=int, default=0, 
                    help='사용할 CPU 코어 수 (0=자동감지, -1=모든 코어 사용)')
parser.add_argument('--input_file', type=str, default='/home/cseomoon/appl/af_analysis-0.1.4/data/final_data_with_rosetta_20250418.csv',
                    help='입력 데이터 파일 경로')
parser.add_argument('--point_size', type=float, default=20.0,
                   help='예측-실제값 플롯의 데이터 포인트 크기 (기본값: 20.0)')
parser.add_argument('--point_alpha', type=float, default=0.5,
                   help='예측-실제값 플롯의 데이터 포인트 투명도 (기본값: 0.5)')
parser.add_argument('--models', type=str, default='rf,lgb,xgb',
                   help='학습 및 비교할 모델 (쉼표로 구분, rf=Random Forest, lgb=LightGBM, xgb=XGBoost)')

args = parser.parse_args()

# 현재 날짜와 시각을 파일명에 사용할 형식으로 가져오기
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 타임스탬프가 포함된 출력 디렉토리 생성
output_dir = f'model_comparison_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"출력 디렉토리 생성: {output_dir}")

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

# 모델들의 결과를 저장할 딕셔너리
model_results = {}

# 데이터 전처리 및 분할 함수
def preprocess_data(input_file):
    print(f"데이터 파일 로드 중: {input_file}")
    df = pd.read_csv(input_file)
    print(f"원본 데이터 크기: {df.shape}")

    # 불필요한 컬럼 제거 (query는 유지)
    cols_to_drop = ['pdb', 'seed', 'sample', 'data_file', 
                   'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm',
                   'format', 'model_path', 'native_path',
                   'Fnat', 'Fnonnat', 'rRMS', 'iRMS', 'LRMS']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 결측치 확인 및 처리
    print("결측치 개수:\n", df.isnull().sum())
    initial_rows = len(df)
    df = df.dropna()
    print(f"결측치 제거 후 데이터 크기: {df.shape} ({initial_rows - len(df)}개 행 제거됨)")

    # 학습 데이터(X)와 레이블(y) 분리
    X = df.drop(columns=['DockQ'])
    y = df['DockQ']

    # 데이터 스케일링
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

    # 쿼리 기반 데이터 분할 (Data Leakage 방지)
    unique_queries = df['query'].unique()
    print(f"고유 query 수: {len(unique_queries)}")

    query_train, query_temp = train_test_split(unique_queries, test_size=0.3, random_state=42)
    query_val, query_test = train_test_split(query_temp, test_size=0.5, random_state=42)

    train_mask = df['query'].isin(query_train)
    val_mask = df['query'].isin(query_val)
    test_mask = df['query'].isin(query_test)

    X_train = X_scaled[train_mask]
    y_train = y[train_mask]
    X_val = X_scaled[val_mask] 
    y_val = y[val_mask]
    X_test = X_scaled[test_mask]
    y_test = y[test_mask]

    print(f"훈련 세트: {X_train.shape}, 검증 세트: {X_val.shape}, 테스트 세트: {X_test.shape}")
    print(f"훈련 쿼리 수: {len(query_train)}, 검증 쿼리 수: {len(query_val)}, 테스트 쿼리 수: {len(query_test)}")

    # 모델링에는 쿼리 열 제외
    if 'query' in X_train.columns:
        X_train = X_train.drop(columns=['query']) 
        X_val = X_val.drop(columns=['query'])
        X_test = X_test.drop(columns=['query'])
    
    # 상관관계 분석 및 시각화
    plt.figure(figsize=(12, 10))
    correlation = X.drop(columns=['query'], errors='ignore').corr()
    sns.heatmap(correlation, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    corr_file = os.path.join(output_dir, f'correlation_matrix_{timestamp}.png')
    plt.savefig(corr_file)
    plt.close()
    print(f"상관관계 행렬 저장: {corr_file}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, numeric_features, scaler

# Random Forest 모델 학습 및 평가 함수
def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, n_jobs, output_dir, timestamp):
    print("\n=== Random Forest 모델 학습 및 평가 ===")
    start_time = time.time()
    
    # 5-fold 교차 검증
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    cv_scores = cross_val_score(base_rf, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=n_jobs)
    print(f"교차 검증 MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 하이퍼파라미터 최적화
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=n_jobs),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
    
    # 최적 모델 평가
    best_rf = grid_search.best_estimator_
    y_val_pred = best_rf.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print(f"검증 세트 MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
    
    # 테스트 세트 평가
    y_test_pred = best_rf.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"테스트 세트 MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    print(f"학습 시간: {training_time:.2f}초")
    
    # 특성 중요도
    feature_importances = best_rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # 모델 저장
    model_path = os.path.join(output_dir, f'best_rf_model_{timestamp}.pkl')
    joblib.dump(best_rf, model_path)
    
    # 결과 저장
    results = {
        'model': best_rf,
        'y_pred': y_test_pred,
        'metrics': {
            'MSE': test_mse,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'R2': test_r2
        },
        'feature_importance': importance_df,
        'cv_scores': -cv_scores,
        'training_time': training_time,
        'hyperparams': grid_search.best_params_,
        'color': 'blue'
    }
    
    return results

# LightGBM 모델 학습 및 평가 함수
def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, n_jobs, output_dir, timestamp):
    print("\n=== LightGBM 모델 학습 및 평가 ===")
    start_time = time.time()
    
    # 5-fold 교차 검증
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    cv_scores = cross_val_score(base_lgb, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=n_jobs)
    print(f"교차 검증 MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 하이퍼파라미터 최적화
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [20, 30, 50]
    }
    
    grid_search = GridSearchCV(
        lgb.LGBMRegressor(random_state=42, n_jobs=n_jobs),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
    
    # 최적 모델 평가
    best_lgb = grid_search.best_estimator_
    y_val_pred = best_lgb.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print(f"검증 세트 MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
    
    # 테스트 세트 평가
    y_test_pred = best_lgb.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"테스트 세트 MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    print(f"학습 시간: {training_time:.2f}초")
    
    # 특성 중요도
    feature_importances = best_lgb.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # 모델 저장
    model_path = os.path.join(output_dir, f'best_lgb_model_{timestamp}.pkl')
    joblib.dump(best_lgb, model_path)
    
    # 결과 저장
    results = {
        'model': best_lgb,
        'y_pred': y_test_pred,
        'metrics': {
            'MSE': test_mse,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'R2': test_r2
        },
        'feature_importance': importance_df,
        'cv_scores': -cv_scores,
        'training_time': training_time,
        'hyperparams': grid_search.best_params_,
        'color': 'green'
    }
    
    return results

# XGBoost 모델 학습 및 평가 함수
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, n_jobs, output_dir, timestamp):
    print("\n=== XGBoost 모델 학습 및 평가 ===")
    start_time = time.time()
    
    # 5-fold 교차 검증
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    cv_scores = cross_val_score(base_xgb, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=n_jobs)
    print(f"교차 검증 MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 하이퍼파라미터 최적화
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=n_jobs),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
    
    # 최적 모델 평가
    best_xgb = grid_search.best_estimator_
    y_val_pred = best_xgb.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print(f"검증 세트 MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
    
    # 테스트 세트 평가
    y_test_pred = best_xgb.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"테스트 세트 MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    print(f"학습 시간: {training_time:.2f}초")
    
    # 특성 중요도
    feature_importances = best_xgb.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # 모델 저장
    model_path = os.path.join(output_dir, f'best_xgb_model_{timestamp}.pkl')
    joblib.dump(best_xgb, model_path)
    
    # 결과 저장
    results = {
        'model': best_xgb,
        'y_pred': y_test_pred,
        'metrics': {
            'MSE': test_mse,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'R2': test_r2
        },
        'feature_importance': importance_df,
        'cv_scores': -cv_scores,
        'training_time': training_time,
        'hyperparams': grid_search.best_params_,
        'color': 'red'
    }
    
    return results

# 비교 분석 함수
def compare_models(model_results, y_test, output_dir, timestamp, point_size=20, point_alpha=0.5):
    print("\n=== 모델 비교 분석 ===")
    
    # 1. 평가 지표 비교 표
    metrics_comparison = pd.DataFrame({
        'Model': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R²': [],
        'Training Time (s)': []
    })
    
    for model_name, results in model_results.items():
        metrics_comparison = metrics_comparison.append({
            'Model': model_name,
            'MSE': results['metrics']['MSE'],
            'RMSE': results['metrics']['RMSE'],
            'MAE': results['metrics']['MAE'],
            'R²': results['metrics']['R2'],
            'Training Time (s)': results['training_time']
        }, ignore_index=True)
    
    # 평가 지표 테이블 저장
    metrics_table = os.path.join(output_dir, f'metrics_comparison_{timestamp}.csv')
    metrics_comparison.to_csv(metrics_table, index=False)
    print(f"평가 지표 비교 테이블 저장: {metrics_table}")
    print(metrics_comparison)
    
    # 2. 모델 예측 비교 그래프
    plt.figure(figsize=(12, 10))
    
    for model_name, results in model_results.items():
        plt.scatter(y_test, results['y_pred'], alpha=point_alpha, s=point_size, 
                   label=f"{model_name} (R²={results['metrics']['R2']:.4f})", 
                   color=results['color'])
    
    # 이상적인 예측선 추가
    min_val = min(y_test.min(), min([results['y_pred'].min() for results in model_results.values()]))
    max_val = max(y_test.max(), max([results['y_pred'].max() for results in model_results.values()]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Model Comparison: Predictions vs Actual Values', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    pred_comp_file = os.path.join(output_dir, f'prediction_comparison_{timestamp}.png')
    plt.savefig(pred_comp_file, dpi=300)
    plt.close()
    print(f"예측 비교 그래프 저장: {pred_comp_file}")
    
    # 3. 잔차 비교 그래프
    plt.figure(figsize=(12, 10))
    
    for model_name, results in model_results.items():
        residuals = y_test - results['y_pred']
        plt.scatter(results['y_pred'], residuals, alpha=point_alpha, s=point_size, 
                   label=model_name, color=results['color'])
    
    plt.axhline(y=0, color='k', linestyle='-')
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Model Comparison: Residual Plots', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    resid_comp_file = os.path.join(output_dir, f'residual_comparison_{timestamp}.png')
    plt.savefig(resid_comp_file, dpi=300)
    plt.close()
    print(f"잔차 비교 그래프 저장: {resid_comp_file}")
    
    # 4. 교차 검증 점수 비교
    plt.figure(figsize=(10, 6))
    cv_data = []
    
    for model_name, results in model_results.items():
        for score in results['cv_scores']:
            cv_data.append({
                'Model': model_name,
                'MSE': score
            })
    
    cv_df = pd.DataFrame(cv_data)
    sns.boxplot(x='Model', y='MSE', data=cv_df)
    plt.title('Cross-Validation MSE Comparison')
    plt.tight_layout()
    
    cv_comp_file = os.path.join(output_dir, f'cv_comparison_{timestamp}.png')
    plt.savefig(cv_comp_file)
    plt.close()
    print(f"교차 검증 비교 그래프 저장: {cv_comp_file}")
    
    # 5. 특성 중요도 비교
    plt.figure(figsize=(12, 8))
    
    # 모든 모델의 상위 10개 특성 추출
    top_features = set()
    for model_name, results in model_results.items():
        top_10 = results['feature_importance'].head(10)['feature'].tolist()
        top_features.update(top_10)
    
    top_features = list(top_features)
    feature_importance_comparison = pd.DataFrame(index=top_features)
    
    for model_name, results in model_results.items():
        # 해당 모델의 특성 중요도 추출
        importance_dict = results['feature_importance'].set_index('feature')['importance'].to_dict()
        # 상위 특성에 대한 중요도 값 추가 (없는 경우 0)
        model_importance = [importance_dict.get(feature, 0) for feature in top_features]
        feature_importance_comparison[model_name] = model_importance
    
    # 특성 중요도 정규화 (최대값을 1로)
    for model in feature_importance_comparison.columns:
        max_val = feature_importance_comparison[model].max()
        if max_val > 0:
            feature_importance_comparison[model] = feature_importance_comparison[model] / max_val
    
    # 특성 중요도 그래프 그리기
    feature_importance_comparison.plot(kind='barh', figsize=(12, 8))
    plt.title('Normalized Feature Importance Comparison (Top Features)')
    plt.xlabel('Normalized Importance')
    plt.tight_layout()
    
    importance_comp_file = os.path.join(output_dir, f'feature_importance_comparison_{timestamp}.png')
    plt.savefig(importance_comp_file)
    plt.close()
    print(f"특성 중요도 비교 그래프 저장: {importance_comp_file}")
    
    # 6. 결과 요약 텍스트 파일 생성
    results_file = os.path.join(output_dir, f'model_comparison_results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write(f"모델 비교 분석 결과 - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 평가 지표 비교\n")
        f.write("-" * 30 + "\n")
        f.write(metrics_comparison.to_string(index=False) + "\n\n")
        
        f.write("2. 최적 모델 하이퍼파라미터\n")
        f.write("-" * 30 + "\n")
        for model_name, results in model_results.items():
            f.write(f"{model_name}:\n")
            for param, value in results['hyperparams'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
        
        f.write("3. 모델 분석 요약\n")
        f.write("-" * 30 + "\n")
        # 가장 성능이 좋은 모델 찾기
        best_model = metrics_comparison.loc[metrics_comparison['R²'].idxmax()]['Model']
        worst_model = metrics_comparison.loc[metrics_comparison['R²'].idxmin()]['Model']
        fastest_model = metrics_comparison.loc[metrics_comparison['Training Time (s)'].idxmin()]['Model']
        
        f.write(f"가장 성능이 좋은 모델: {best_model} (R² = {metrics_comparison.loc[metrics_comparison['Model'] == best_model, 'R²'].values[0]:.4f})\n")
        f.write(f"가장 성능이 낮은 모델: {worst_model} (R² = {metrics_comparison.loc[metrics_comparison['Model'] == worst_model, 'R²'].values[0]:.4f})\n")
        f.write(f"가장 빠른 학습 모델: {fastest_model} ({metrics_comparison.loc[metrics_comparison['Model'] == fastest_model, 'Training Time (s)'].values[0]:.2f}초)\n\n")
        
        f.write("4. 상위 공통 중요 특성\n")
        f.write("-" * 30 + "\n")
        for feature in feature_importance_comparison.index[:5]:  # 상위 5개 특성 출력
            f.write(f"{feature}\n")
    
    print(f"모델 비교 결과 요약 저장: {results_file}")
    
    return metrics_comparison

# 메인 실행 함수
def main():
    # 사용할 CPU 코어 수 설정
    n_jobs = get_cpu_count(args.n_jobs)
    
    # 데이터 전처리 및 분할
    X_train, y_train, X_val, y_val, X_test, y_test, numeric_features, scaler = preprocess_data(args.input_file)
    
    # 스케일러 저장
    scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.pkl')
    joblib.dump(scaler, scaler_path)
    
    # 학습할 모델 목록
    models_to_train = args.models.split(',')
    
    # 모델 학습 및 결과 저장
    if 'rf' in models_to_train:
        try:
            from sklearn.model_selection import GridSearchCV
            model_results['Random Forest'] = train_random_forest(
                X_train, y_train, X_val, y_val, X_test, y_test, n_jobs, output_dir, timestamp
            )
        except Exception as e:
            print(f"Random Forest 모델 학습 중 오류 발생: {str(e)}")
    
    if 'lgb' in models_to_train:
        try:
            model_results['LightGBM'] = train_lightgbm(
                X_train, y_train, X_val, y_val, X_test, y_test, n_jobs, output_dir, timestamp
            )
        except Exception as e:
            print(f"LightGBM 모델 학습 중 오류 발생: {str(e)}")
    
    if 'xgb' in models_to_train:
        try:
            model_results['XGBoost'] = train_xgboost(
                X_train, y_train, X_val, y_val, X_test, y_test, n_jobs, output_dir, timestamp
            )
        except Exception as e:
            print(f"XGBoost 모델 학습 중 오류 발생: {str(e)}")
    
    # 모델 비교 분석 수행
    if len(model_results) > 1:
        metrics_comparison = compare_models(
            model_results, y_test, output_dir, timestamp, 
            point_size=args.point_size, point_alpha=args.point_alpha
        )
    else:
        print("비교할 모델이 충분하지 않습니다. 최소 2개 이상의 모델이 필요합니다.")

# 프로그램 시작
if __name__ == "__main__":
    main()

def run_nested_cv_with_repeated_shap(X, y, query_ids, model_names, output_dir, 
                                     outer_folds=5, inner_folds=3, cv_repeats=10, n_jobs=4):
    # 기존 외부 CV 코드에 반복 루프 추가
    all_cv_results = {model: [] for model in model_names}
    
    for repeat in range(cv_repeats):
        print(f"\nCV 반복 {repeat+1}/{cv_repeats} 시작")
        # 다른 랜덤 시드로 CV 객체 생성
        outer_cv = GroupKFold(n_splits=outer_folds)
        # ... 기존 nested CV 코드 ...
        
        # 반복별 결과 저장
        all_cv_results[model_name].append(fold_results)

    # 모든 샘플의 SHAP 값을 추적하는 딕셔너리
    sample_shap_values = {}
    
    # 각 CV 반복마다 테스트 샘플의 SHAP 값 저장
    for sample_idx in test_idx:
        if sample_idx not in sample_shap_values:
            sample_shap_values[sample_idx] = {model_name: []}
        elif model_name not in sample_shap_values[sample_idx]:
            sample_shap_values[sample_idx][model_name] = []
        
        # 해당 샘플의 SHAP 값 추가
        sample_shap_values[sample_idx][model_name].append(shap_value_for_sample)

def plot_shap_variation(sample_shap_values, feature_names, output_dir):
    # 각 특성별 SHAP 값의 범위 계산
    feature_ranges = []
    for feature_idx, feature_name in enumerate(feature_names):
        ranges = []
        for sample_idx, model_values in sample_shap_values.items():
            for model_name, shap_values_list in model_values.items():
                # 이 샘플의 이 특성에 대한 모든 SHAP 값
                feature_shap_values = [sv[feature_idx] for sv in shap_values_list]
                ranges.append({
                    'sample': sample_idx,
                    'feature': feature_name,
                    'model': model_name,
                    'range': max(feature_shap_values) - min(feature_shap_values),
                    'std': np.std(feature_shap_values),
                    'mean': np.mean(feature_shap_values)
                })
        feature_ranges.extend(ranges)
    
    # 데이터프레임으로 변환하여 시각화
    range_df = pd.DataFrame(feature_ranges)
    
    # 1. 범위 시각화
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='feature', y='range', data=range_df)
    plt.xticks(rotation=45)
    plt.title('SHAP Value Range per Feature across CV Repeats')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_value_range.png'), dpi=300)
    plt.close()
    
    # 2. 스케일링된 범위 시각화 (평균으로 나눔)
    range_df['scaled_range'] = range_df['range'] / range_df['mean'].abs()
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='feature', y='scaled_range', data=range_df)
    plt.xticks(rotation=45)
    plt.title('Scaled SHAP Value Range per Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaled_shap_value_range.png'), dpi=300)
    plt.close()

def assess_feature_importance_stability(all_cv_results, model_names, output_dir):
    # 각 CV 반복별 특성 중요도 순위 추적
    feature_ranks = {}
    
    for model_name in model_names:
        feature_ranks[model_name] = []
        
        for repeat_results in all_cv_results[model_name]:
            # 특성 중요도 계산 (평균 절대 SHAP 값)
            importances = repeat_results['importances']
            # 순위 매기기
            ranks = pd.Series(importances['importance']).rank(ascending=False)
            feature_ranks[model_name].append(
                pd.DataFrame({
                    'feature': importances['feature'],
                    'rank': ranks
                })
            )
        
        # 특성별 순위 통계 계산
        rank_stats = pd.concat(feature_ranks[model_name])
        rank_stats = rank_stats.groupby('feature').agg({
            'rank': ['mean', 'std', 'min', 'max']
        })
        
        # 평균 순위로 정렬
        rank_stats = rank_stats.sort_values(('rank', 'mean'))
        
        # 순위 안정성 시각화
        plt.figure(figsize=(12, 10))
        plt.errorbar(
            x=rank_stats.index,
            y=rank_stats[('rank', 'mean')],
            yerr=rank_stats[('rank', 'std')],
            fmt='o',
            capsize=5
        )
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{model_name} - Feature Importance Rank Stability')
        plt.ylabel('Average Rank (lower is more important)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_rank_stability.png'), dpi=300)
        plt.close()
