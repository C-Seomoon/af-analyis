import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from datetime import datetime
import psutil
import time
import threading
import shap
import json

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='XGBoost Regressor 모델 학습')
parser.add_argument('--n_jobs', type=int, default=0, 
                    help='사용할 CPU 코어 수 (0=자동감지, -1=모든 코어 사용)')
parser.add_argument('--input_file', type=str, default='/home/cseomoon/appl/af_analysis-0.1.4/data/final_data_with_rosetta_20250418.csv',
                    help='입력 데이터 파일 경로')
parser.add_argument('--monitor_interval', type=float, default=5.0,
                    help='CPU 모니터링 간격(초)')
parser.add_argument('--point_size', type=float, default=20.0,
                   help='예측-실제값 플롯의 데이터 포인트 크기 (기본값: 20.0)')
parser.add_argument('--point_alpha', type=float, default=0.5,
                   help='예측-실제값 플롯의 데이터 포인트 투명도 (기본값: 0.5)')

args = parser.parse_args()

# 현재 날짜와 시각을 파일명에 사용할 형식으로 가져오기
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 타임스탬프가 포함된 출력 디렉토리 생성
output_dir = f'xgboost_regressor_native_api_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"출력 디렉토리 생성: {output_dir}")

# 타임스탬프 직후 중요 경로 정의
model_path = os.path.join(output_dir, f'best_xgb_model_{timestamp}.model')
scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.pkl')
results_file = os.path.join(output_dir, f'model_results_{timestamp}.txt')

def save_model_results(model, y_test, y_pred, metrics, feature_importance_df, cv_scores, training_time, hyperparams, output_dir, model_name):
    """
    모델 결과를 표준 형식으로 저장
    """
    # 결과 디렉토리 생성
    results_dir = os.path.join(output_dir, 'comparison_data')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 예측값 저장
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    predictions_df.to_csv(os.path.join(results_dir, f'{model_name}_predictions.csv'), index=False)
    
    # 2. 성능 지표 저장
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = model_name
    metrics_df['training_time'] = training_time
    metrics_df.to_csv(os.path.join(results_dir, f'{model_name}_metrics.csv'), index=False)
    
    # 3. 특성 중요도 저장
    feature_importance_df.to_csv(os.path.join(results_dir, f'{model_name}_feature_importance.csv'), index=False)
    
    # 4. 교차 검증 점수 저장 (XGBoost는 별도의 교차 검증이 없을 수 있음)
    if cv_scores is not None:
        cv_df = pd.DataFrame({'cv_score': cv_scores})
        cv_df.to_csv(os.path.join(results_dir, f'{model_name}_cv_scores.csv'), index=False)
    
    # 5. 하이퍼파라미터 저장
    with open(os.path.join(results_dir, f'{model_name}_hyperparams.json'), 'w') as f:
        # XGBoost 하이퍼파라미터를 JSON 직렬화 가능하게 변환
        json_hyperparams = {}
        for k, v in hyperparams.items():
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                json_hyperparams[k] = v
            else:
                json_hyperparams[k] = str(v)
        json.dump(json_hyperparams, f)
    
    print(f"{model_name} 모델 비교 데이터 저장 완료: {results_dir}")

# CPU 사용 개수 결정 함수
def get_cpu_count(requested_jobs=0):
    """CPU 코어 수 결정"""
    total_cpus = os.cpu_count() or 4  # CPU 수 감지 (감지 실패 시 4 사용)
    
    if requested_jobs == 0:  # 자동 감지 (75% 사용)
        cpu_count = max(1, int(total_cpus * 0.75))
        print(f"자동 감지된 CPU 수: {total_cpus}, 사용할 코어 수: {cpu_count} (75%)")
        
    elif requested_jobs == -1:  # 모든 코어 사용
        cpu_count = total_cpus
        print(f"모든 CPU 코어 사용: {cpu_count}")
        
    else:  # 사용자 지정 수
        if requested_jobs > total_cpus:
            print(f"경고: 요청한 코어 수({requested_jobs})가 가용 코어 수({total_cpus})보다 많습니다.")
            cpu_count = total_cpus
        else:
            cpu_count = requested_jobs
        print(f"사용자 지정 CPU 코어 수: {cpu_count}/{total_cpus}")
    
    return cpu_count

# CPU 및 메모리 모니터링 함수
def monitor_resources(output_dir, interval=5.0, stop_event=None):
    """CPU 및 메모리 사용량을 모니터링하고 파일로 저장"""
    resource_file = os.path.join(output_dir, f'resource_usage_{timestamp}.csv')
    with open(resource_file, 'w') as f:
        f.write("timestamp,cpu_percent,memory_percent,memory_used_gb,memory_available_gb\n")
    
    resources = []
    
    try:
        while not (stop_event and stop_event.is_set()):
            cpu_percent = psutil.cpu_percent(interval=interval)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_available_gb = memory.available / (1024 ** 3)
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"{current_time},{cpu_percent},{memory_percent},{memory_used_gb:.2f},{memory_available_gb:.2f}\n"
            
            with open(resource_file, 'a') as f:
                f.write(log_line)
            
            resources.append({
                'timestamp': current_time,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_available_gb': memory_available_gb
            })
            
            print(f"[{current_time}] CPU: {cpu_percent}% | 메모리: {memory_percent}% | 사용: {memory_used_gb:.2f}GB | 가용: {memory_available_gb:.2f}GB")
            
            # 다음 모니터링까지 대기 (interval 초)
            time.sleep(interval)
    
    except Exception as e:
        print(f"모니터링 중단: {str(e)}")
    
    print(f"리소스 모니터링 완료, 데이터 저장: {resource_file}")
    return resources

# 사용할 CPU 코어 수 설정
n_jobs = get_cpu_count(args.n_jobs)

# 스레드 이벤트 객체 생성 (스레드 중단용)
stop_monitoring = threading.Event()

# 리소스 모니터링 시작
print(f"리소스 모니터링 시작 (간격: {args.monitor_interval}초)...")
resources = []
resource_monitor = threading.Thread(
    target=monitor_resources, 
    args=(output_dir, args.monitor_interval, stop_monitoring)
)
resource_monitor.daemon = True
resource_monitor.start()

try:
    # 학습 시작 시간 기록
    start_time = time.time()
    
    # 데이터 로드
    print(f"데이터 파일 로드 중: {args.input_file}")
    df = pd.read_csv(args.input_file)
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
    
    # DMatrix 형식으로 데이터 변환
    print("DMatrix 형식으로 데이터 변환 중...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 제한된 하이퍼파라미터 그리드 정의
    print(f"\n하이퍼파라미터 최적화 중... (n_jobs={n_jobs})")
    param_combinations = []
    for max_depth in [3, 6]:  # 2개로 축소
        for learning_rate in [0.05, 0.1]:  # 2개로 축소
            for subsample in [0.8]:  # 1개로 고정
                for colsample_bytree in [0.8]:  # 1개로 고정
                    for min_child_weight in [1, 3]:  # 2개로 축소
                        for gamma in [0]:  # 1개로 고정
                            for reg_alpha in [0, 0.1]:  # 2개로 축소
                                for reg_lambda in [1, 1.5]:  # 2개로 축소
                                    param_combinations.append({
                                        'max_depth': max_depth,
                                        'eta': learning_rate,
                                        'subsample': subsample,
                                        'colsample_bytree': colsample_bytree,
                                        'min_child_weight': min_child_weight,
                                        'gamma': gamma,
                                        'alpha': reg_alpha,
                                        'lambda': reg_lambda,
                                        'objective': 'reg:squarederror',
                                        'eval_metric': 'rmse',
                                        'nthread': n_jobs
                                    })
    
    print(f"총 {len(param_combinations)}개의 파라미터 조합 테스트 예정")
    
    # 5-fold 교차 검증 (초기 모델 평가) - 선택 사항
    print("5-fold 교차 검증 수행 중...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    cv_scores = cross_val_score(
        base_xgb, 
        X_scaled.drop(columns=['query'], errors='ignore'), 
        y, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs
    )
    print(f"교차 검증 MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 네이티브 API로 최적 모델 찾기
    best_score = float('inf')  # 회귀 문제는 낮을수록 좋음
    best_params = None
    best_model = None
    best_iteration = 0
    
    # 검증 시 사용할 eval_list
    eval_list = [(dtrain, 'train'), (dval, 'val')]
    
    # 각 하이퍼파라미터 조합 시도
    for i, params in enumerate(param_combinations):
        print(f"파라미터 조합 {i+1}/{len(param_combinations)} 테스트 중...")
        
        # 모델 학습 (native API)
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=eval_list,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # 검증 세트 점수 확인 (RMSE, 작을수록 좋음)
        val_score = bst.best_score
        
        # 최고 성능 모델 업데이트
        if val_score < best_score:
            best_score = val_score
            best_params = params
            best_model = bst
            best_iteration = bst.best_iteration
    
    print(f"최적 하이퍼파라미터: {best_params}")
    print(f"최적 모델 점수(RMSE): {best_score}")
    print(f"최적 반복 횟수: {best_iteration}")
    
    # 최적 모델로 테스트 세트 예측
    y_test_pred = best_model.predict(dtest)
    
    # 테스트 성능 평가
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"테스트 세트 MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
    
    # 특성 중요도 추출
    feature_importances = best_model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(feature_importances.keys()),
        'importance': list(feature_importances.values())
    }).sort_values('importance', ascending=False)
    
    # 상위 10개 특성 (또는 가능한 모든 특성)
    top_n = min(10, len(importance_df))
    top_features = [(importance_df['feature'].iloc[i], importance_df['importance'].iloc[i]) 
                    for i in range(top_n)]
    print("상위 10개 특성:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    # 특성 중요도 시각화
    plt.figure(figsize=(10, 6))
    # 상위 20개 특성으로 제한
    n_features = min(20, len(importance_df))
    top_df = importance_df.head(n_features)
    plt.barh(range(len(top_df)), top_df['importance'], align='center')
    plt.yticks(range(len(top_df)), top_df['feature'])
    plt.xlabel('Importance')
    plt.title('XGBoost Feature Importance (Gain)')
    plt.tight_layout()
    imp_file = os.path.join(output_dir, f'feature_importance_{timestamp}.png')
    plt.savefig(imp_file)
    plt.close()
    print(f"특성 중요도 그래프 저장: {imp_file}")
    
    # 예측값 vs 실제값 플롯
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y_test, y_test_pred, alpha=args.point_alpha, s=args.point_size)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'XGBoost: Predictions vs Actual Values\nTest R² = {test_r2:.4f}', fontsize=14)

    # 플롯에 MSE 및 R² 텍스트 추가
    text_str = f'MSE: {test_mse:.4f}\nR²: {test_r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, 
            fontsize=12, verticalalignment='top', bbox=props)

    # 플롯에 그리드 추가
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 타임스탬프를 포함한 파일명으로 저장
    pred_file = os.path.join(output_dir, f'prediction_vs_actual_{timestamp}.png')
    plt.savefig(pred_file, dpi=300)  # 더 높은 해상도로 저장
    plt.close()
    print(f"예측 플롯 저장: {pred_file}")
    
    # 실제값 vs 예측값 잔차 플롯 추가
    plt.figure(figsize=(10, 8))
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=args.point_alpha, s=args.point_size)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    residual_file = os.path.join(output_dir, f'residual_plot_{timestamp}.png')
    plt.savefig(residual_file, dpi=300)
    plt.close()
    print(f"잔차 플롯 저장: {residual_file}")
    
    # SHAP 분석
    try:
        print("SHAP 분석 시작...")
        # XGBoost용 SHAP 값 계산
        explainer = shap.TreeExplainer(best_model)
        
        # 테스트 샘플 중 일부만 사용하여 계산 속도 향상
        shap_sample_size = min(500, len(X_test))
        X_test_sample = X_test.iloc[:shap_sample_size]
        
        # SHAP 값 계산 (회귀 모델이므로 shap_values[1] 대신 shap_values 사용)
        shap_values = explainer.shap_values(X_test_sample)
        
        # SHAP Summary Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
        plt.tight_layout()
        shap_bar_file = os.path.join(output_dir, f'shap_bar_{timestamp}.png')
        plt.savefig(shap_bar_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP Bar Plot 저장: {shap_bar_file}")
        
        # SHAP Summary Dot Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, show=False)
        plt.tight_layout()
        shap_dot_file = os.path.join(output_dir, f'shap_dot_{timestamp}.png')
        plt.savefig(shap_dot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP Dot Plot 저장: {shap_dot_file}")
        
        # SHAP 의존성 플롯 (가장 중요한 특성에 대해서만)
        if len(X_test.columns) > 0 and len(importance_df) > 0:
            top_feature = importance_df['feature'].iloc[0]  # 가장 중요한 특성
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(top_feature, shap_values, X_test_sample, show=False)
            plt.tight_layout()
            shap_dep_file = os.path.join(output_dir, f'shap_dependence_{top_feature}_{timestamp}.png')
            plt.savefig(shap_dep_file, dpi=300)
            plt.close()
            print(f"SHAP Dependence Plot 저장: {shap_dep_file}")
    except ImportError:
        print("SHAP 라이브러리가 설치되지 않았습니다. 'pip install shap'로 설치하세요.")
    except Exception as e:
        print(f"SHAP 분석 중 오류 발생: {str(e)}")
    
    # 최적 모델 저장 - XGBoost 모델은 자체 save_model 메서드 사용
    best_model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"모델 저장 완료: {model_path}")
    print(f"스케일러 저장 완료: {scaler_path}")
    
    # 특성 중요도를 CSV로 저장
    imp_csv = os.path.join(output_dir, f'feature_importance_{timestamp}.csv')
    importance_df.to_csv(imp_csv, index=False)
    print(f"특성 중요도 CSV 저장: {imp_csv}")
    
    # 모델 평가 결과를 텍스트 파일로 저장
    with open(results_file, 'w') as f:
        f.write(f"실행 날짜/시각: {timestamp}\n")
        f.write(f"데이터 크기: {df.shape}\n")
        f.write(f"CPU 코어 수: {n_jobs}\n")
        f.write(f"훈련 세트: {X_train.shape}, 검증 세트: {X_val.shape}, 테스트 세트: {X_test.shape}\n")
        f.write(f"최적 하이퍼파라미터: {best_params}\n")
        f.write(f"최적 모델 점수(RMSE): {best_score:.6f}\n")
        f.write(f"테스트 세트 MSE: {test_mse:.6f}, R²: {test_r2:.6f}, MAE: {test_mae:.6f}\n")
        
        f.write("\n상위 10개 특성:\n")
        for feature, importance in top_features:
            f.write(f"  {feature}: {importance:.6f}\n")

        f.write(f"\n조기 중단 정보:\n")
        f.write(f"  최종 반복 횟수: {best_iteration}\n")
        f.write(f"  설정된 early_stopping_rounds: 50\n")

    print(f"모델 평가 결과 저장: {results_file}")

    # 모델 학습 시간 계산
    training_time = time.time() - start_time
    print(f"총 학습 및 평가 시간: {training_time:.2f}초")
    
    # 성능 지표 정리
    metrics = {
        'MSE': test_mse,
        'RMSE': np.sqrt(test_mse),  # RMSE 계산
        'MAE': test_mae,
        'R2': test_r2
    }
    
    # 비교용 표준 형식으로 결과 저장
    # XGBoost는 별도의 교차 검증 단계가 없으므로 None 전달
    save_model_results(
        model=best_model,
        y_test=y_test,
        y_pred=y_test_pred,
        metrics=metrics,
        feature_importance_df=importance_df,
        cv_scores=None,  # XGBoost에서는 별도의 교차 검증 점수가 없음
        training_time=training_time,
        hyperparams=best_params,
        output_dir=output_dir,
        model_name='XGBoost'
    )

finally:
    # 리소스 모니터링 종료
    stop_monitoring.set()
    resource_monitor.join(timeout=10)
    
    # 리소스 사용량 그래프 생성 (자원 모니터링이 수집한 데이터가 있을 경우)
    try:
        resource_file = os.path.join(output_dir, f'resource_usage_{timestamp}.csv')
        if os.path.exists(resource_file):
            resource_df = pd.read_csv(resource_file)
            
            if len(resource_df) > 1:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.plot(resource_df['cpu_percent'])
                plt.title('CPU Usage')
                plt.ylabel('CPU %')
                plt.xlabel('Time (samples)')
                
                plt.subplot(1, 2, 2)
                plt.plot(resource_df['memory_percent'])
                plt.title('Memory Usage')
                plt.ylabel('Memory %')
                plt.xlabel('Time (samples)')
                
                plt.tight_layout()
                resource_plot = os.path.join(output_dir, f'resource_usage_plot_{timestamp}.png')
                plt.savefig(resource_plot)
                plt.close()
                print(f"리소스 사용량 그래프 저장: {resource_plot}")
    except Exception as e:
        print(f"리소스 그래프 생성 중 오류: {str(e)}")

    print("\n모델 사용 예시 (회귀):")
    print(f"loaded_model = xgb.Booster()")
    print(f"loaded_model.load_model('{model_path}')")
    print(f"loaded_scaler = joblib.load('{scaler_path}')")
    print("scaled_data = loaded_scaler.transform(new_data[numeric_features])")
    print("dtest = xgb.DMatrix(scaled_data)")
    print("predictions = loaded_model.predict(dtest)")

