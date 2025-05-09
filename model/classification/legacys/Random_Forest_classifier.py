import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

import seaborn as sns
import joblib
import os
import argparse
from datetime import datetime
import psutil
import time
import threading
from sklearn.metrics import matthews_corrcoef, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
import shap
import json

# plt.rcParams['font.family'] = 'NanumGothic'  # 또는 '', 'AppleGothic' 등
# plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지


# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='Random Forest Classifier 모델 학습 (DockQ 분류)')
parser.add_argument('--n_jobs', type=int, default=0, 
                    help='사용할 CPU 코어 수 (0=자동감지, -1=모든 코어 사용)')
parser.add_argument('--input_file', type=str, default='/home/cseomoon/appl/af_analysis-0.1.4/data/final_data_with_rosetta_20250418.csv',
                    help='입력 데이터 파일 경로')
parser.add_argument('--threshold', type=float, default=0.23,
                    help='DockQ 분류 임계값 (기본값: 0.23)')
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
output_dir = f'RF_classifier_output_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"출력 디렉토리 생성: {output_dir}")

def save_model_results(model, y_test, y_pred, y_pred_proba, metrics, feature_importance_df, 
                       confusion_matrix_data, cv_scores, training_time, hyperparams, 
                       output_dir, model_name):
    """분류 모델 결과를 표준 형식으로 저장"""
    
    # 비교용 데이터 디렉토리 생성
    results_dir = os.path.join(output_dir, 'comparison_data')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 예측값 저장
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_pred_proba
    })
    predictions_df.to_csv(os.path.join(results_dir, f'{model_name}_predictions.csv'), index=False)
    
    # 2. 성능 지표 저장
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = model_name
    metrics_df['training_time'] = training_time
    metrics_df.to_csv(os.path.join(results_dir, f'{model_name}_metrics.csv'), index=False)
    
    # 3. 특성 중요도 저장
    feature_importance_df.to_csv(os.path.join(results_dir, f'{model_name}_feature_importance.csv'), index=False)
    
    # 4. 혼동 행렬 저장
    pd.DataFrame(confusion_matrix_data).to_csv(
        os.path.join(results_dir, f'{model_name}_confusion_matrix.csv'), index=False)
    
    # 5. 교차 검증 점수 저장
    if cv_scores is not None:
        cv_df = pd.DataFrame({'cv_score': cv_scores})
        cv_df.to_csv(os.path.join(results_dir, f'{model_name}_cv_scores.csv'), index=False)
    
    # 6. 하이퍼파라미터 저장
    with open(os.path.join(results_dir, f'{model_name}_hyperparams.json'), 'w') as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v 
                  for k, v in hyperparams.items()}, f)
    
    print(f"{model_name} 모델 비교 데이터 저장 완료: {results_dir}") 



# CPU 사용 개수 결정 함수
def get_cpu_count(requested_jobs=0):
    """
    CPU 코어 수 결정
    
    Parameters:
    -----------
    requested_jobs : int
        요청한 코어 수 (0=자동감지, -1=모든 코어)
    
    Returns:
    --------
    int : 사용할 코어 수
    """
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

# 리소스 모니터링 시작 이전, try 블록 직전에 추가
start_time = time.time()

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

    # DockQ 값을 기준으로 이진 레이블 생성 (0.23 기준)
    threshold = args.threshold
    print(f"DockQ 임계값: {threshold} (이상: 1, 미만: 0)")
    
    # 원본 DockQ 값 저장
    df['DockQ_orig'] = df['DockQ'].copy()
    
    # 레이블 변환
    df['DockQ'] = (df['DockQ'] >= threshold).astype(int)
    
    # 클래스 분포 확인
    class_counts = df['DockQ'].value_counts()
    print(f"클래스 분포:\n0 (DockQ < {threshold}): {class_counts.get(0, 0)} 샘플\n1 (DockQ >= {threshold}): {class_counts.get(1, 0)} 샘플")

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
    
    # 각 세트의 클래스 비율 확인
    train_ratio = y_train.mean()
    val_ratio = y_val.mean()
    test_ratio = y_test.mean()
    print(f"클래스 1의 비율: 훈련 세트 {train_ratio:.2f}, 검증 세트 {val_ratio:.2f}, 테스트 세트 {test_ratio:.2f}")

    # 모델링에는 쿼리 열 제외
    if 'query' in X_train.columns:
        X_train = X_train.drop(columns=['query']) 
        X_val = X_val.drop(columns=['query'])
        X_test = X_test.drop(columns=['query'])

    # 5-fold 교차 검증 (초기 모델 평가)
    print("5-fold 교차 검증 수행 중...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs)
    cv_scores = cross_val_score(
        base_rf, 
        X_scaled.drop(columns=['query'], errors='ignore'), 
        y, 
        cv=cv, 
        scoring='f1',
        n_jobs=n_jobs
    )
    print(f"교차 검증 F1 점수: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 기본 모델 학습 및 평가
    print("기본 모델 학습 중...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs)
    rf.fit(X_train, y_train)

    # 검증 세트 성능 평가
    y_val_pred = rf.predict(X_val)
    y_val_prob = rf.predict_proba(X_val)[:, 1]  # 클래스 1의 확률
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"검증 세트 성능: Accuracy={val_accuracy:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}, F1={val_f1:.4f}")

    # 특성 간 상관관계 확인
    print("특성 간 상관관계 분석 중...")
    plt.figure(figsize=(12, 10))
    correlation = X.drop(columns=['query'], errors='ignore').corr()
    sns.heatmap(correlation, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    # 타임스탬프를 포함한 파일명으로 저장
    corr_file = os.path.join(output_dir, f'correlation_matrix_{timestamp}.png')
    plt.savefig(corr_file)
    plt.close()
    print(f"상관관계 행렬 저장: {corr_file}")

    # 높은 상관관계를 가진 특성들 확인 (0.9 이상)
    high_corr = (correlation.abs() > 0.9) & (correlation != 1.0)
    high_corr_features = []
    for col in high_corr.columns:
        high_corr_pairs = high_corr[col][high_corr[col]].index.tolist()
        if high_corr_pairs:
            high_corr_features.append((col, high_corr_pairs))

    if high_corr_features:
        print("높은 상관관계를 가진 특성들:")
        for feature, corr_features in high_corr_features:
            print(f"  {feature}: {corr_features}")

    # 하이퍼파라미터 최적화
    print(f"\n하이퍼파라미터 최적화 중... (n_jobs={n_jobs})")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']  # 불균형 데이터 처리용
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=n_jobs),
        param_grid=param_grid,
        cv=kf,
        scoring='f1',  # F1 점수를 최적화 기준으로 사용
        n_jobs=1  # GridSearch 자체는 단일 쓰레드로 실행 (내부적으로 병렬화)
    )
    grid_search.fit(X_train, y_train)

    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")

    # 최적 모델로 테스트 세트 평가
    best_rf = grid_search.best_estimator_
    y_test_pred = best_rf.predict(X_test)
    y_test_prob = best_rf.predict_proba(X_test)[:, 1]  # 클래스 1의 확률
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # 추가 평가 지표 계산
    mcc = matthews_corrcoef(y_test, y_test_pred)
    ll = log_loss(y_test, y_test_prob)
    average_precision = average_precision_score(y_test, y_test_prob)

    print(f"테스트 세트 성능: Accuracy={test_accuracy:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}")
    print(f"추가 성능 지표: PR-AUC={average_precision:.4f}, MCC={mcc:.4f}, Log Loss={ll:.4f}")
    
    # 분류 보고서 출력
    print("\n분류 보고서:")
    print(classification_report(y_test, y_test_pred))

    # 특성 중요도 시각화
    feature_importances = best_rf.feature_importances_
    sorted_idx = feature_importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), feature_importances[sorted_idx])
    plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_idx], rotation=90)
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    # 타임스탬프를 포함한 파일명으로 저장
    imp_file = os.path.join(output_dir, f'feature_importance_{timestamp}.png')
    plt.savefig(imp_file)
    plt.close()
    print(f"특성 중요도 그래프 저장: {imp_file}")

    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=[f'DockQ < {threshold}', f'DockQ ≥ {threshold}'],
               yticklabels=[f'DockQ < {threshold}', f'DockQ ≥ {threshold}'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    cm_file = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_file)
    plt.close()
    print(f"혼동 행렬 저장: {cm_file}")

    # ROC 곡선 시각화
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_file = os.path.join(output_dir, f'roc_curve_{timestamp}.png')
    plt.savefig(roc_file)
    plt.close()
    print(f"ROC 곡선 저장: {roc_file}")

    # Precision-Recall 곡선 시각화 (통합 버전)
    plt.figure(figsize=(8, 6))

    # PR 곡선 계산
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    average_precision = average_precision_score(y_test, y_test_prob)

    # PR 곡선 그리기
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AP = {average_precision:.4f})')

    # 무작위 분류기의 기준선 (클래스 1의 비율)
    baseline = y_test.mean()
    plt.plot([0, 1], [baseline, baseline], color='red', lw=2, linestyle='--',
             label=f'Baseline (Class 1 ratio = {baseline:.4f})')

    # 축 범위 및 레이블 설정
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)

    # 그리드, 범례 및 스타일링 추가
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    # 고해상도로 저장
    pr_file = os.path.join(output_dir, f'pr_curve_{timestamp}.png')
    plt.savefig(pr_file, dpi=300)
    plt.close()
    print(f"Precision-Recall 곡선 저장: {pr_file}")

    # PR 곡선 관련 정보 출력
    print(f"Average Precision (PR-AUC): {average_precision:.4f}")
    print(f"Baseline (무작위 분류기): {baseline:.4f}")

    # 예측 확률 히스토그램 (클래스별)
    plt.figure(figsize=(10, 6))
    plt.hist(y_test_prob[y_test == 0], bins=20, alpha=0.5, label=f'DockQ < {threshold}')
    plt.hist(y_test_prob[y_test == 1], bins=20, alpha=0.5, label=f'DockQ ≥ {threshold}')
    plt.xlabel('Predicted Probability of Class 1')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution (Class-wise)')
    plt.legend()
    prob_file = os.path.join(output_dir, f'prediction_probabilities_{timestamp}.png')
    plt.savefig(prob_file)
    plt.close()
    print(f"예측 확률 분포 저장: {prob_file}")
    
    # 원본 DockQ 값과 예측 확률 관계 산점도
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y_test_orig, y_test_prob, c=y_test, cmap='coolwarm', 
                          alpha=args.point_alpha, s=args.point_size)
    plt.axhline(y=0.5, color='gray', linestyle='--')  # 분류 임계선
    plt.axvline(x=threshold, color='gray', linestyle='--')  # DockQ 임계선
    plt.xlabel(f'DockQ (threshold: {threshold})', fontsize=12)
    plt.ylabel('Predicted Probability of Class 1', fontsize=12)
    plt.title('Relationship between DockQ and Predicted Probability', fontsize=14)
    plt.colorbar(scatter, label='Actual Class')
    
    # 사분면 레이블 추가
    plt.text(min(y_test_orig)+(max(y_test_orig)-min(y_test_orig))*0.25, 0.25, 'TN', fontsize=14)
    plt.text(threshold+(max(y_test_orig)-min(y_test_orig))*0.25, 0.25, 'FN', fontsize=14)
    plt.text(min(y_test_orig)+(max(y_test_orig)-min(y_test_orig))*0.25, 0.75, 'FP', fontsize=14)
    plt.text(threshold+(max(y_test_orig)-min(y_test_orig))*0.25, 0.75, 'TP', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    dockq_prob_file = os.path.join(output_dir, f'dockq_vs_probability_{timestamp}.png')
    plt.savefig(dockq_prob_file, dpi=300)
    plt.close()
    print(f"DockQ-확률 관계 그래프 저장: {dockq_prob_file}")

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
    model_path = os.path.join(output_dir, f'best_rf_classifier_{timestamp}.pkl')
    scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.pkl')
    joblib.dump(best_rf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"모델 저장 완료: {model_path}")
    print(f"스케일러 저장 완료: {scaler_path}")

    # 모델 평가 결과를 텍스트 파일로 저장
    results_file = os.path.join(output_dir, f'model_results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write(f"실행 날짜/시각: {timestamp}\n")
        f.write(f"DockQ 임계값: {threshold} (이상: 1, 미만: 0)\n")
        f.write(f"클래스 분포: 0={class_counts.get(0, 0)}, 1={class_counts.get(1, 0)}\n")
        f.write(f"데이터 크기: {df.shape}\n")
        f.write(f"CPU 코어 수: {n_jobs}\n")
        f.write(f"훈련 세트: {X_train.shape}, 검증 세트: {X_val.shape}, 테스트 세트: {X_test.shape}\n")
        f.write(f"클래스 1의 비율: 훈련 세트 {train_ratio:.2f}, 검증 세트 {val_ratio:.2f}, 테스트 세트 {test_ratio:.2f}\n")
        f.write(f"교차 검증 F1 점수: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n")
        f.write(f"최적 하이퍼파라미터: {grid_search.best_params_}\n")
        f.write(f"검증 세트 성능: 정확도={val_accuracy:.4f}, 정밀도={val_precision:.4f}, 재현율={val_recall:.4f}, F1={val_f1:.4f}\n")
        f.write(f"테스트 세트 성능: 정확도={test_accuracy:.4f}, 정밀도={test_precision:.4f}, 재현율={test_recall:.4f}, F1={test_f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {average_precision:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Log Loss: {ll:.4f}\n\n")
        f.write("분류 보고서:\n")
        f.write(classification_report(y_test, y_test_pred))
        f.write("\n상위 10개 특성:\n")
        for feature, importance in top_features:
            f.write(f"  {feature}: {importance:.4f}\n")
    print(f"모델 평가 결과 저장: {results_file}")

    # Learning Curve 추가
    print("Learning Curve 계산 중...")
    plt.figure(figsize=(10, 6))
    try:
        # 데이터 샘플 수를 제한하여 메모리 문제 방지
        sample_size = min(1000, len(X_train))
        X_train_sample = X_train.iloc[:sample_size]
        y_train_sample = y_train.iloc[:sample_size]
        
        train_sizes, train_scores, valid_scores = learning_curve(
            RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=best_rf.max_depth,
                min_samples_split=best_rf.min_samples_split,
                min_samples_leaf=best_rf.min_samples_leaf,
                class_weight=best_rf.class_weight,
                n_jobs=1),  # 메모리 이슈 방지를 위해 단일 스레드 사용
            X_train_sample, 
            y_train_sample, 
            cv=5, 
            scoring='f1',
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=1  # 메모리 이슈 방지를 위해 단일 스레드 사용
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)

        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, 
                 label='Training F1')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, 
                         alpha=0.15, color='blue')

        plt.plot(train_sizes, valid_mean, color='green', marker='s', markersize=5, 
                 linestyle='--', label='Validation F1')
        plt.fill_between(train_sizes, valid_mean + valid_std, valid_mean - valid_std, 
                         alpha=0.15, color='green')

        plt.title('Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        learning_curve_file = os.path.join(output_dir, f'learning_curve_{timestamp}.png')
        plt.savefig(learning_curve_file, dpi=300)
        plt.close()
        print(f"학습 곡선 저장: {learning_curve_file}")
    except Exception as e:
        print(f"학습 곡선 생성 중 오류 발생: {str(e)}")
        print("학습 곡선 생성을 건너뜁니다.")

    # Calibration Curve 추가
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='s', markersize=5, label='Random Forest')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    calibration_file = os.path.join(output_dir, f'calibration_curve_{timestamp}.png')
    plt.savefig(calibration_file, dpi=300)
    plt.close()
    print(f"보정 곡선 저장: {calibration_file}")

    # SHAP 분석 수정
    try:
        print("SHAP 분석 시작 (버전 0.47.2)...")
        
        # 테스트 샘플 준비
        shap_sample_size = min(100, len(X_test))
        X_test_sample = X_test.iloc[:shap_sample_size].copy()
        print(f"샘플 크기: {shap_sample_size}, 특성 수: {X_test_sample.shape[1]}")
        
        # SHAP Explainer 생성
        print("TreeExplainer 생성 중...")
        explainer = shap.TreeExplainer(best_rf)
        
        # SHAP 값 계산
        print("SHAP 값 계산 중...")
        shap_values = explainer(X_test_sample)
        print(f"SHAP 값 타입: {type(shap_values)}")
        print(f"SHAP 값 형태: {shap_values.shape if hasattr(shap_values, 'shape') else '형태 정보 없음'}")
        
        # 클래스 1(관심 클래스)에 대한 SHAP 값만 추출
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
            print(f"다차원 SHAP 값 감지: {shap_values.values.shape}")
            # 클래스 1(양성)의 SHAP 값만 사용
            class_idx = 1
            # 원본 설명 객체에서 클래스 1에 대한 값만 추출하여 새 설명 객체 생성
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
            shap_values_for_plot = class1_exp
            print(f"클래스 1에 대한 SHAP 값으로 변환 완료: {class1_exp.values.shape}")
        else:
            shap_values_for_plot = shap_values
        
        # SHAP 시각화
        # 1. 기존 Summary Plot (레거시 방식)
        print("Summary Plot 생성 중...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_for_plot, X_test_sample, show=False)
        plt.title('SHAP Feature Summary')
        plt.tight_layout()
        summary_file = os.path.join(output_dir, f'shap_summary_{timestamp}.png')
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary Plot 저장: {summary_file}")
        
        # 2. Bar Plot (특성 중요도)
        print("Bar Plot 생성 중...")
        plt.figure(figsize=(12, 8))
        shap.plots.bar(shap_values_for_plot, max_display=15, show=False)
        plt.tight_layout()
        bar_file = os.path.join(output_dir, f'shap_bar_{timestamp}.png')
        plt.savefig(bar_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bar Plot 저장: {bar_file}")
        
        # 파일 확인
        for file_path in [summary_file, bar_file]:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"확인: {os.path.basename(file_path)} 크기: {file_size/1024:.2f} KB")
            else:
                print(f"경고: {os.path.basename(file_path)} 파일이 생성되지 않았습니다")
            
    except ImportError:
        print("SHAP 라이브러리가 설치되지 않았습니다. 'pip install shap'로 설치하세요.")
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"SHAP 분석 중 오류 발생: {str(e)}")
        print(f"상세 오류: {error_msg}")
        
        # 오류 로그 파일에 저장
        with open(os.path.join(output_dir, f'shap_error_{timestamp}.log'), 'w') as f:
            f.write(f"SHAP 분석 중 오류 발생: {str(e)}\n")
            f.write(f"상세 오류:\n{error_msg}")

    # 교차 검증 기반 ROC/PR AUC 분포 시각화
    print("교차 검증 성능 분포 분석 중...")

    # 교차 검증 점수 계산
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = []
    pr_auc_scores = []

    # 교차 검증으로 ROC AUC, PR AUC 계산
    for train_idx, val_idx in cv.split(X_scaled.drop(columns=['query'], errors='ignore'), y):
        X_cv_train = X_scaled.drop(columns=['query'], errors='ignore').iloc[train_idx]
        y_cv_train = y.iloc[train_idx]
        X_cv_val = X_scaled.drop(columns=['query'], errors='ignore').iloc[val_idx]
        y_cv_val = y.iloc[val_idx]
        
        # 분류기 학습
        cv_classifier = RandomForestClassifier(
            n_estimators=best_rf.n_estimators,
            max_depth=best_rf.max_depth,
            min_samples_split=best_rf.min_samples_split,
            min_samples_leaf=best_rf.min_samples_leaf,
            class_weight=best_rf.class_weight,
            random_state=42,
            n_jobs=1
        )
        
        cv_classifier.fit(X_cv_train, y_cv_train)
        y_cv_prob = cv_classifier.predict_proba(X_cv_val)[:, 1]
        
        # ROC AUC 계산
        fpr, tpr, _ = roc_curve(y_cv_val, y_cv_prob)
        roc_auc_scores.append(auc(fpr, tpr))
        
        # PR AUC 계산
        precision, recall, _ = precision_recall_curve(y_cv_val, y_cv_prob)
        pr_auc_scores.append(average_precision_score(y_cv_val, y_cv_prob))

    # 두 지표를 boxplot으로 시각화
    plt.figure(figsize=(10, 6))
    boxplot_data = [roc_auc_scores, pr_auc_scores]
    plt.boxplot(boxplot_data, tick_labels=['ROC-AUC', 'PR-AUC'], 
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red'))

    # 개별 데이터 포인트 표시
    for i, data in enumerate(boxplot_data, 1):
        plt.scatter([i] * len(data), data, color='black', alpha=0.5)

    plt.title('Cross-Validation Performance Distribution')
    plt.ylabel('Score')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    # 타임스탬프를 포함한 파일명으로 저장
    cv_boxplot_file = os.path.join(output_dir, f'cv_boxplot_{timestamp}.png')
    plt.savefig(cv_boxplot_file, dpi=300)
    plt.close()
    print(f"교차 검증 성능 분포 플롯 저장: {cv_boxplot_file}")

    # 교차 검증 점수 정보 저장
    cv_info = pd.DataFrame({
        'roc_auc': roc_auc_scores,
        'pr_auc': pr_auc_scores
    })
    cv_stats = cv_info.describe()
    cv_stats_file = os.path.join(output_dir, f'cv_performance_stats_{timestamp}.csv')
    cv_stats.to_csv(cv_stats_file)
    print(f"교차 검증 성능 통계 저장: {cv_stats_file}")

    # 모델 결과를 표준 형식으로 저장
    save_model_results(best_rf, y_test, y_test_pred, y_test_prob, {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'roc_auc': roc_auc,
        'pr_auc': average_precision,
        'mcc': mcc,
        'log_loss': ll
    }, importance_df, cm, cv_scores, time.time() - start_time, grid_search.best_params_, output_dir, 'RF')

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

    print("\n모델 사용 예시 (분류):")
    print(f"loaded_model = joblib.load('{model_path}')")
    print(f"loaded_scaler = joblib.load('{scaler_path}')")
    print("scaled_data = loaded_scaler.transform(new_data[numeric_features])")
    print("predictions = loaded_model.predict(scaled_data)  # 클래스 예측 (0 또는 1)")
    print("probabilities = loaded_model.predict_proba(scaled_data)[:, 1]  # 클래스 1의 확률")

