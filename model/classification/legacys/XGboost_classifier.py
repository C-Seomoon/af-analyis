import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from datetime import datetime
import psutil
import time
import threading
from sklearn.metrics import ConfusionMatrixDisplay

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='XGBoost Binary Classifier 모델 학습 (DockQ 분류)')
parser.add_argument('--n_jobs', type=int, default=0, 
                    help='사용할 CPU 코어 수 (0=자동감지, -1=모든 코어 사용)')
parser.add_argument('--input_file', type=str, default='/home/cseomoon/appl/af_analysis-0.1.4/data/final_data_with_rosetta_20250418.csv',
                    help='입력 데이터 파일 경로')
parser.add_argument('--monitor_interval', type=float, default=5.0,
                    help='CPU 모니터링 간격(초)')
parser.add_argument('--point_size', type=float, default=20.0,
                   help='플롯의 데이터 포인트 크기 (기본값: 20.0)')
parser.add_argument('--point_alpha', type=float, default=0.5,
                   help='플롯의 데이터 포인트 투명도 (기본값: 0.5)')

args = parser.parse_args()

# 현재 날짜와 시각을 파일명에 사용할 형식으로 가져오기
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 타임스탬프가 포함된 출력 디렉토리 생성
output_dir = f'xgboost_binary_with_early_stopping_output_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"출력 디렉토리 생성: {output_dir}")

# DockQ 값에 따른 이진 클래스 매핑 함수
def get_dockq_binary_class(dockq_value):
    """
    DockQ 값을 기반으로 이진 클래스 반환
    
    0: DockQ < 0.23 (Incorrect)
    1: DockQ >= 0.23 (Acceptable, Medium, High)
    """
    if dockq_value < 0.23:
        return 0  # Incorrect
    else:
        return 1  # Acceptable, Medium, High

# 클래스 레이블 (시각화 및 출력용)
CLASS_NAMES = ['Incorrect', 'Acceptable+']
CLASS_THRESHOLD = 0.23

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
            
            print(f"[{current_time}] CPU: {cpu_percent}% | Memory: {memory_percent}% | Used: {memory_used_gb:.2f}GB | Available: {memory_available_gb:.2f}GB")
            
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
    # 데이터 로드
    print(f"데이터 파일 로드 중: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"원본 데이터 크기: {df.shape}")

    # DockQ 값을 기준으로 이진 클래스 레이블 생성
    # 원본 DockQ 값 저장
    df['DockQ_orig'] = df['DockQ'].copy()
    
    # 이진 클래스 레이블 변환
    df['DockQ'] = df['DockQ_orig'].apply(get_dockq_binary_class)
    
    # 클래스 분포 확인
    class_counts = df['DockQ'].value_counts().sort_index()
    print("클래스 분포:")
    for i, class_name in enumerate(CLASS_NAMES):
        threshold_str = f"< {CLASS_THRESHOLD:.2f}" if i == 0 else f">= {CLASS_THRESHOLD:.2f}"
        count = class_counts.get(i, 0)
        percent = 100 * count / len(df) if len(df) > 0 else 0
        print(f"  Class {i} ({class_name}, DockQ: {threshold_str}): {count} 샘플 ({percent:.2f}%)")

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
    X = df.drop(columns=['DockQ', 'DockQ_orig'])
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
    
    # 원본 DockQ 값 추출 (시각화 목적)
    y_train_orig = df.loc[train_mask, 'DockQ_orig']
    y_val_orig = df.loc[val_mask, 'DockQ_orig']
    y_test_orig = df.loc[test_mask, 'DockQ_orig']

    print(f"훈련 세트: {X_train.shape}, 검증 세트: {X_val.shape}, 테스트 세트: {X_test.shape}")
    print(f"훈련 쿼리 수: {len(query_train)}, 검증 쿼리 수: {len(query_val)}, 테스트 쿼리 수: {len(query_test)}")
    
    # 각 세트의 클래스 분포 확인
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    val_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    
    print("클래스 비율:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  Class {i} ({class_name}):")
        print(f"    훈련 세트: {train_dist.get(i, 0):.4f}, 검증 세트: {val_dist.get(i, 0):.4f}, 테스트 세트: {test_dist.get(i, 0):.4f}")

    # 모델링에는 쿼리 열 제외
    if 'query' in X_train.columns:
        X_train = X_train.drop(columns=['query']) 
        X_val = X_val.drop(columns=['query'])
        X_test = X_test.drop(columns=['query'])

    # 5-fold 교차 검증 (초기 모델 평가)
    print("5-fold 교차 검증 수행 중...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base_xgb = xgb.XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=n_jobs,
        eval_metric='logloss'  # use_label_encoder 제거
    )
    cv_scores = cross_val_score(
        base_xgb, 
        X_scaled.drop(columns=['query'], errors='ignore'), 
        y, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=n_jobs
    )
    print(f"교차 검증 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # timestamp 생성 직후에 모델 경로 미리 정의
    model_path = os.path.join(output_dir, f'best_xgb_binary_{timestamp}.pkl')
    scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.pkl')

    # 하이퍼파라미터 최적화 - XGBoost 특화 파라미터로 변경
    print(f"\n하이퍼파라미터 최적화 중... (n_jobs={n_jobs})")
    param_grid = {
        'n_estimators': [500],  # 최대 반복 횟수 설정 (early stopping으로 자동 조절됨)
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1, 2, 3]  # 클래스 불균형 처리용
    }

    # GridSearchCV에서 조기 중단 설정
    fit_params = {
        'eval_set': [(X_val, y_val)],
        'early_stopping_rounds': 50,
        'verbose': False
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        xgb.XGBClassifier(
            random_state=42, 
            n_jobs=n_jobs,
            eval_metric='auc'
        ),
        param_grid=param_grid,
        cv=kf,
        scoring='f1',
        n_jobs=1
    )

    # fit_params 전달
    grid_search.fit(X_train, y_train, **fit_params)

    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")

    # 최적 모델로 테스트 세트 평가 - best_iteration 활용
    best_xgb = grid_search.best_estimator_

    # best_iteration이 있는 경우 해당 반복까지만 사용
    if hasattr(best_xgb, 'best_iteration'):
        y_test_pred = best_xgb.predict(X_test, iteration_range=(0, best_xgb.best_iteration + 1))
        y_test_proba = best_xgb.predict_proba(X_test, iteration_range=(0, best_xgb.best_iteration + 1))[:, 1]
    elif hasattr(best_xgb, 'best_iteration_'):
        y_test_pred = best_xgb.predict(X_test, iteration_range=(0, best_xgb.best_iteration_ + 1))
        y_test_proba = best_xgb.predict_proba(X_test, iteration_range=(0, best_xgb.best_iteration_ + 1))[:, 1]
    else:
        # 조기 중단이 사용되지 않은 경우 기본 예측
        y_test_pred = best_xgb.predict(X_test)
        y_test_proba = best_xgb.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"테스트 세트 성능: Accuracy={test_accuracy:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
    
    # 분류 보고서 출력
    print("\n분류 보고서:")
    class_report = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES)
    print(class_report)

    # 특성 중요도 시각화
    feature_importances = best_xgb.feature_importances_
    sorted_idx = feature_importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), feature_importances[sorted_idx])
    plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_idx], rotation=90)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    # 타임스탬프를 포함한 파일명으로 저장
    imp_file = os.path.join(output_dir, f'feature_importance_{timestamp}.png')
    plt.savefig(imp_file)
    plt.close()
    print(f"특성 중요도 그래프 저장: {imp_file}")

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_file = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_file)
    plt.close()
    print(f"혼동 행렬 저장: {cm_file}")
    
    # ROC 곡선 시각화
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    roc_file = os.path.join(output_dir, f'roc_curve_{timestamp}.png')
    plt.savefig(roc_file, dpi=300)
    plt.close()
    print(f"ROC 곡선 저장: {roc_file}")
    
    # 원본 DockQ 값과 예측 확률 관계 산점도
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(y_test_orig, y_test_proba, 
                        c=y_test, cmap='coolwarm', 
                        alpha=args.point_alpha, 
                        s=args.point_size)
    
    # DockQ 임계값 표시
    plt.axvline(x=CLASS_THRESHOLD, color='black', linestyle='--', label=f'DockQ Threshold = {CLASS_THRESHOLD}')
    
    # 확률 임계값 표시 (0.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Probability Threshold = 0.5')
    
    plt.xlabel('Original DockQ Value', fontsize=12)
    plt.ylabel('Predicted Probability of Class 1', fontsize=12)
    plt.title('Relationship between DockQ Value and Predicted Probability', fontsize=14)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(CLASS_NAMES)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    prob_file = os.path.join(output_dir, f'dockq_vs_probability_{timestamp}.png')
    plt.savefig(prob_file, dpi=300)
    plt.close()
    print(f"DockQ-확률 관계 그래프 저장: {prob_file}")

    # 클래스별 확률 분포 히스토그램
    plt.figure(figsize=(10, 6))
    
    # 실제 클래스 0인 샘플의 확률 분포
    class0_probs = y_test_proba[y_test == 0]
    if len(class0_probs) > 0:
        plt.hist(class0_probs, bins=20, alpha=0.5, label=f'Actual Class: {CLASS_NAMES[0]}')
    
    # 실제 클래스 1인 샘플의 확률 분포
    class1_probs = y_test_proba[y_test == 1]
    if len(class1_probs) > 0:
        plt.hist(class1_probs, bins=20, alpha=0.5, label=f'Actual Class: {CLASS_NAMES[1]}')
    
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Predicted Probability of Class 1')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities by Actual Class')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    dist_file = os.path.join(output_dir, f'probability_distribution_{timestamp}.png')
    plt.savefig(dist_file, dpi=300)
    plt.close()
    print(f"확률 분포 히스토그램 저장: {dist_file}")

    # 상위 10개 특성 출력
    top_features = [(X_train.columns[i], feature_importances[i]) for i in sorted_idx[:10]]
    print("상위 10개 특성:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")

    # 특성 중요도를 CSV로 저장
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    imp_csv = os.path.join(output_dir, f'feature_importance_{timestamp}.csv')
    importance_df.to_csv(imp_csv, index=False)
    print(f"특성 중요도 CSV 저장: {imp_csv}")

    # 최적 모델 저장 (타임스탬프 포함)
    joblib.dump(best_xgb, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"모델 저장 완료: {model_path}")
    print(f"스케일러 저장 완료: {scaler_path}")

    # 모델 평가 결과를 텍스트 파일로 저장
    results_file = os.path.join(output_dir, f'model_results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write(f"실행 날짜/시각: {timestamp}\n")
        f.write(f"DockQ 이진 클래스 정의:\n")
        f.write(f"  Class 0 ({CLASS_NAMES[0]}): DockQ < {CLASS_THRESHOLD:.2f}\n")
        f.write(f"  Class 1 ({CLASS_NAMES[1]}): DockQ >= {CLASS_THRESHOLD:.2f}\n")
            
        f.write("\n클래스 분포:\n")
        for i, class_name in enumerate(CLASS_NAMES):
            count = class_counts.get(i, 0)
            percent = 100 * count / len(df) if len(df) > 0 else 0
            f.write(f"  Class {i} ({class_name}): {count} 샘플 ({percent:.2f}%)\n")
        
        f.write(f"\n데이터 크기: {df.shape}\n")
        f.write(f"CPU 코어 수: {n_jobs}\n")
        f.write(f"훈련 세트: {X_train.shape}, 검증 세트: {X_val.shape}, 테스트 세트: {X_test.shape}\n")
        
        f.write("\n클래스 비율:\n")
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f"  Class {i} ({class_name}):\n")
            f.write(f"    훈련 세트: {train_dist.get(i, 0):.4f}, 검증 세트: {val_dist.get(i, 0):.4f}, 테스트 세트: {test_dist.get(i, 0):.4f}\n")
        
        f.write(f"\n교차 검증 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n")
        f.write(f"최적 하이퍼파라미터: {grid_search.best_params_}\n")
        
        f.write(f"\n검증 세트 성능: Accuracy={test_accuracy:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}\n")
        f.write(f"테스트 세트 성능: Accuracy={test_accuracy:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}\n")
        
        f.write("\n분류 보고서:\n")
        f.write(class_report)
        
        f.write("\n상위 10개 특성:\n")
        for feature, importance in top_features:
            f.write(f"  {feature}: {importance:.4f}\n")

        f.write(f"\n조기 중단 정보:\n")
        if hasattr(best_xgb, 'best_iteration'):
            f.write(f"  최종 반복 횟수: {best_xgb.best_iteration}\n")
        elif hasattr(best_xgb, 'best_iteration_'):
            f.write(f"  최종 반복 횟수: {best_xgb.best_iteration_}\n")
        else:
            f.write(f"  조기 중단 미사용\n")
        f.write(f"  설정된 early_stopping_rounds: 50\n")

    print(f"모델 평가 결과 저장: {results_file}")

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

    print("\n모델 사용 예시 (이진 분류):")
    print(f"loaded_model = joblib.load('{model_path}')")
    print(f"loaded_scaler = joblib.load('{scaler_path}')")
    print("scaled_data = loaded_scaler.transform(new_data[numeric_features])")
    print("predictions = loaded_model.predict(scaled_data)  # 클래스 예측 (0 또는 1)")
    print("probabilities = loaded_model.predict_proba(scaled_data)[:, 1]  # 클래스 1의 확률")
