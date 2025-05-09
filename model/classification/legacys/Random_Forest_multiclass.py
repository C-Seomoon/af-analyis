import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from datetime import datetime
import psutil
import time
import threading
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='Random Forest Multi-class Classifier 모델 학습 (DockQ 분류)')
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
output_dir = f'multiclass_output_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"출력 디렉토리 생성: {output_dir}")

# DockQ 값에 따른 클래스 매핑 함수
def get_dockq_class(dockq_value):
    """
    DockQ 값을 기반으로 품질 클래스 반환
    
    0: Incorrect        (0.00 <= DockQ < 0.23)
    1: Acceptable       (0.23 <= DockQ < 0.49)
    2: Medium quality   (0.49 <= DockQ < 0.80)
    3: High quality     (DockQ >= 0.80)
    """
    if dockq_value < 0.23:
        return 0  # Incorrect
    elif dockq_value < 0.49:
        return 1  # Acceptable
    elif dockq_value < 0.80:
        return 2  # Medium
    else:
        return 3  # High

# 클래스 레이블 (시각화 및 출력용)
CLASS_NAMES = ['Incorrect', 'Acceptable', 'Medium', 'High']
CLASS_THRESHOLDS = [0.00, 0.23, 0.49, 0.80]

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

    # DockQ 값을 기준으로 다중 클래스 레이블 생성
    # 원본 DockQ 값 저장
    df['DockQ_orig'] = df['DockQ'].copy()
    
    # 다중 클래스 레이블 변환
    df['DockQ'] = df['DockQ_orig'].apply(get_dockq_class)
    
    # 클래스 분포 확인
    class_counts = df['DockQ'].value_counts().sort_index()
    print("클래스 분포:")
    for i, class_name in enumerate(CLASS_NAMES):
        threshold_str = f"{CLASS_THRESHOLDS[i]:.2f}"
        if i < len(CLASS_THRESHOLDS) - 1:
            threshold_str += f" - {CLASS_THRESHOLDS[i+1]:.2f}"
        else:
            threshold_str += " +"
            
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
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs)
    cv_scores = cross_val_score(
        base_rf, 
        X_scaled.drop(columns=['query'], errors='ignore'), 
        y, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=n_jobs
    )
    print(f"교차 검증 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 기본 모델 학습 및 평가
    print("기본 모델 학습 중...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs)
    rf.fit(X_train, y_train)

    # 검증 세트 성능 평가
    y_val_pred = rf.predict(X_val)
    y_val_proba = rf.predict_proba(X_val)
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_weighted_precision = precision_score(y_val, y_val_pred, average='weighted')
    val_weighted_recall = recall_score(y_val, y_val_pred, average='weighted')
    val_weighted_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"검증 세트 성능: Accuracy={val_accuracy:.4f}, Weighted Precision={val_weighted_precision:.4f}, Weighted Recall={val_weighted_recall:.4f}, Weighted F1={val_weighted_f1:.4f}")

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
        scoring='accuracy',  # 다중 분류에서는 정확도 사용
        n_jobs=1  # GridSearch 자체는 단일 쓰레드로 실행 (내부적으로 병렬화)
    )
    grid_search.fit(X_train, y_train)

    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")

    # 최적 모델로 테스트 세트 평가
    best_rf = grid_search.best_estimator_
    y_test_pred = best_rf.predict(X_test)
    y_test_proba = best_rf.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_weighted_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_weighted_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"테스트 세트 성능: Accuracy={test_accuracy:.4f}, Weighted Precision={test_weighted_precision:.4f}, Weighted Recall={test_weighted_recall:.4f}, Weighted F1={test_weighted_f1:.4f}")
    
    # 분류 보고서 출력
    print("\n분류 보고서:")
    class_report = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES)
    print(class_report)

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
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    cm_file = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_file)
    plt.close()
    print(f"혼동 행렬 저장: {cm_file}")
    
    # 원본 DockQ 값과 예측 클래스 관계 산점도
    plt.figure(figsize=(12, 8))
    
    # 색상 매핑 설정 (실제 클래스)
    cmap = plt.cm.get_cmap('viridis', len(CLASS_NAMES))
    
    # 예측 확률에 따른 투명도 설정
    alphas = np.max(y_test_proba, axis=1)
    alphas = 0.3 + 0.7 * alphas  # 최소 0.3, 최대 1.0
    
    # 각 클래스별 예측 확률에 따른 점크기 조정
    sizes = args.point_size * (1 + np.max(y_test_proba, axis=1))
    
    # 산점도 그리기 (원본 DockQ vs 예측 클래스)
    scatter = plt.scatter(y_test_orig, y_test_pred, 
                         c=y_test, cmap=cmap, 
                         alpha=args.point_alpha, 
                         s=sizes)
    
    # 클래스 경계선 그리기
    for threshold in CLASS_THRESHOLDS[1:]:  # 0.23, 0.49, 0.80
        plt.axvline(x=threshold, color='gray', linestyle='--')
        
    # 1:1 대각선 (완벽한 예측)
    x_range = np.linspace(0, 1, 100)
    y_classes = np.array([get_dockq_class(x) for x in x_range])
    plt.plot(x_range, y_classes, 'k--', alpha=0.5)
    
    plt.xlabel('Original DockQ Value', fontsize=12)
    plt.ylabel('Predicted Class', fontsize=12)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.title('Relationship between DockQ Value and Predicted Class', fontsize=14)
    plt.colorbar(scatter, label='Actual Class', ticks=range(len(CLASS_NAMES)))
    plt.colorbar(ticks=range(len(CLASS_NAMES))).set_ticklabels(CLASS_NAMES)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    dockq_pred_file = os.path.join(output_dir, f'dockq_vs_predicted_class_{timestamp}.png')
    plt.savefig(dockq_pred_file, dpi=300)
    plt.close()
    print(f"DockQ-예측 클래스 관계 그래프 저장: {dockq_pred_file}")
    
    # 클래스 별 예측 확률 분포 히스토그램
    plt.figure(figsize=(15, 10))
    
    # 각 클래스별 확률 분포 그리기
    for i, class_name in enumerate(CLASS_NAMES):
        plt.subplot(2, 2, i+1)
        
        # 실제 클래스가 i인 샘플들
        class_mask = (y_test == i)
        
        # 각 클래스별 예측 확률 히스토그램
        for j, target_class in enumerate(CLASS_NAMES):
            probs = y_test_proba[class_mask, j]
            if len(probs) > 0:
                plt.hist(probs, bins=20, alpha=0.5, label=f'Prob of {target_class}')
        
        plt.xlabel(f'Prediction Probability')
        plt.ylabel('Frequency')
        plt.title(f'Class {i}: {class_name} (n={np.sum(class_mask)})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    prob_dist_file = os.path.join(output_dir, f'class_probability_distributions_{timestamp}.png')
    plt.savefig(prob_dist_file, dpi=300)
    plt.close()
    print(f"클래스별 확률 분포 그래프 저장: {prob_dist_file}")

    # 다중 클래스 ROC 곡선 (One-vs-Rest)
    # 원-핫 인코딩 변환
    y_test_bin = label_binarize(y_test, classes=range(len(CLASS_NAMES)))
    n_classes = y_test_bin.shape[1]
    
    # 각 클래스별 ROC 곡선 계산
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 평균 ROC 계산 (macro)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # ROC 곡선 그리기
    plt.figure(figsize=(10, 8))
    
    # 각 클래스별 ROC 곡선
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color, class_name in zip(range(n_classes), colors, CLASS_NAMES):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_name} (AUC = {roc_auc[i]:.3f})'
        )
    
    # 매크로 평균 ROC 곡선
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
        color='navy', linestyle=':', linewidth=4
    )
    
    # 랜덤 분류기 경계
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves (One-vs-Rest)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    roc_file = os.path.join(output_dir, f'roc_curves_multiclass_{timestamp}.png')
    plt.savefig(roc_file, dpi=300)
    plt.close()
    print(f"다중 클래스 ROC 곡선 저장: {roc_file}")

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
    model_path = os.path.join(output_dir, f'best_rf_multiclass_{timestamp}.pkl')
    scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.pkl')
    joblib.dump(best_rf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"모델 저장 완료: {model_path}")
    print(f"스케일러 저장 완료: {scaler_path}")

    # 모델 평가 결과를 텍스트 파일로 저장
    results_file = os.path.join(output_dir, f'model_results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write(f"실행 날짜/시각: {timestamp}\n")
        f.write(f"DockQ 클래스 정의:\n")
        for i, class_name in enumerate(CLASS_NAMES):
            threshold_str = f"{CLASS_THRESHOLDS[i]:.2f}"
            if i < len(CLASS_THRESHOLDS) - 1:
                threshold_str += f" - {CLASS_THRESHOLDS[i+1]:.2f}"
            else:
                threshold_str += " +"
            f.write(f"  Class {i} ({class_name}): DockQ {threshold_str}\n")
            
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
        
        f.write(f"\n검증 세트 성능: Accuracy={val_accuracy:.4f}, Weighted Precision={val_weighted_precision:.4f}, Weighted Recall={val_weighted_recall:.4f}, Weighted F1={val_weighted_f1:.4f}\n")
        f.write(f"테스트 세트 성능: Accuracy={test_accuracy:.4f}, Weighted Precision={test_weighted_precision:.4f}, Weighted Recall={test_weighted_recall:.4f}, Weighted F1={test_weighted_f1:.4f}\n")
        
        f.write("\n클래스별 ROC-AUC:\n")
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f"  Class {i} ({class_name}): {roc_auc[i]:.4f}\n")
        f.write(f"  매크로 평균: {roc_auc['macro']:.4f}\n")
        
        f.write("\n분류 보고서:\n")
        f.write(class_report)
        
        f.write("\n상위 10개 특성:\n")
        for feature, importance in top_features:
            f.write(f"  {feature}: {importance:.4f}\n")
    
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

    print("\n모델 사용 예시 (다중 분류):")
    print(f"loaded_model = joblib.load('{model_path}')")
    print(f"loaded_scaler = joblib.load('{scaler_path}')")
    print("scaled_data = loaded_scaler.transform(new_data[numeric_features])")
    print("predictions = loaded_model.predict(scaled_data)  # 클래스 예측 (0, 1, 2, 3)")
    print("probabilities = loaded_model.predict_proba(scaled_data)  # 각 클래스의 확률")
    print("\n클래스 레이블:")
    for i, class_name in enumerate(CLASS_NAMES):
        threshold_str = f"{CLASS_THRESHOLDS[i]:.2f}"
        if i < len(CLASS_THRESHOLDS) - 1:
            threshold_str += f" - {CLASS_THRESHOLDS[i+1]:.2f}"
        else:
            threshold_str += " +"
        print(f"  {i}: {class_name} (DockQ {threshold_str})") 