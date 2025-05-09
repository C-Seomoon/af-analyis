import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, matthews_corrcoef, log_loss, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay
import shap
import time

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='LightGBM Binary Classifier 모델 학습 (DockQ 분류)')
parser.add_argument('--n_jobs', type=int, default=0, 
                    help='사용할 CPU 코어 수 (0=자동감지, -1=모든 코어 사용)')
parser.add_argument('--input_file', type=str, default='/home/cseomoon/appl/af_analysis-0.1.4/data/final_data_with_rosetta_20250418.csv',
                    help='입력 데이터 파일 경로')
parser.add_argument('--point_size', type=float, default=20.0,
                   help='플롯의 데이터 포인트 크기 (기본값: 20.0)')
parser.add_argument('--point_alpha', type=float, default=0.5,
                   help='플롯의 데이터 포인트 투명도 (기본값: 0.5)')

args = parser.parse_args()

# 현재 날짜와 시각을 파일명에 사용할 형식으로 가져오기
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 타임스탬프가 포함된 출력 디렉토리 생성
output_dir = f'lightgbm_binary_output_{timestamp}'
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

# 사용할 CPU 코어 수 설정
n_jobs = get_cpu_count(args.n_jobs)

try:
    # 학습 시간 측정 시작
    start_time = time.time()
    
    # 모델 및 스케일러 경로를 미리 정의
    model_path = os.path.join(output_dir, f'best_lgb_binary_{timestamp}.pkl')
    scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.pkl')
    
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

    # 시스템의 총 코어 수에 따라 분배 전략 조정
    total_cores = os.cpu_count() or 4
    if total_cores >= 8:
        grid_search_jobs = max(2, n_jobs // 2)  # 최소 2개 또는 절반의 코어
        model_jobs = 2  # 각 모델에는 적은 수의 코어 할당
    else:
        grid_search_jobs = n_jobs
        model_jobs = 1

    # 하이퍼파라미터 그리드 정의 (확장된 버전)
    param_grid = {
        'n_estimators': [100, 300, 500],             # 3개
        'learning_rate': [0.01, 0.1],                # 2개
        'num_leaves': [31, 63],                      # 2개
        'max_depth': [5, -1],                        # 2개 (-1은 제한 없음)
        'min_child_samples': [20, 50],               # 2개
        'subsample': [0.8, 1.0],                     # 2개 (새로 추가)
        'colsample_bytree': [0.8, 1.0],              # 2개 (새로 추가)
        'class_weight': [None, 'balanced']           # 2개
    }

    # 하이퍼파라미터 최적화 먼저 수행
    print("\n하이퍼파라미터 최적화 시작...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV 생성
    grid_search = GridSearchCV(
        lgb.LGBMClassifier(random_state=42, n_jobs=model_jobs),
        param_grid=param_grid,
        cv=kf,
        scoring='f1',
        n_jobs=grid_search_jobs,
        verbose=1
    )

    # fit_params는 GridSearchCV에서 직접 전달할 수 없음 - 모델 재학습 시 사용
    grid_search.fit(X_train, y_train)
    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")

    # 최적 하이퍼파라미터로 모델 생성
    best_params = grid_search.best_params_
    lgb_model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=n_jobs)

    # early stopping 설정
    fit_params = {
        'eval_set': [(X_val, y_val)],
        'eval_metric': 'auc',
        'callbacks': [
            lgb.early_stopping(
                stopping_rounds=50,
                first_metric_only=True,
                verbose=True
            )
        ]
    }

    # 최적 하이퍼파라미터로 모델 재학습 (early stopping 적용)
    print("최적 하이퍼파라미터로 모델 학습 중...")
    lgb_model.fit(X_train, y_train, **fit_params)

    # 조기 중단된 iteration 출력
    print(f"Early stopping 사용: {lgb_model.best_iteration_}번째 iteration에서 중단")

    # 최적 모델로 테스트 세트 평가
    y_test_pred = lgb_model.predict(X_test)
    y_test_proba = lgb_model.predict_proba(X_test)[:, 1]  # 클래스 1의 확률
    
    # 기본 평가 지표
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # 추가 평가 지표
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall, precision)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    ll = log_loss(y_test, y_test_proba)
    
    print(f"테스트 세트 성능: Accuracy={test_accuracy:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
    print(f"추가 성능 지표: PR-AUC={pr_auc:.4f}, MCC={mcc:.4f}, Log Loss={ll:.4f}")
    
    # 분류 보고서 출력
    print("\n분류 보고서:")
    class_report = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES)
    print(class_report)

    # 특성 중요도 시각화 - 'gain' 기준으로 변경
    feature_importances = lgb_model.booster_.feature_importance(importance_type='gain')
    sorted_idx = feature_importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), feature_importances[sorted_idx])
    plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_idx], rotation=90)
    plt.title('LightGBM Feature Importance (Gain)')
    plt.tight_layout()
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
    
    # Precision-Recall 곡선 시각화 (통합 버전)
    plt.figure(figsize=(8, 6))

    # PR 곡선 계산
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    average_precision = average_precision_score(y_test, y_test_proba)

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
    pr_curve_file = os.path.join(output_dir, f'pr_curve_{timestamp}.png')
    plt.savefig(pr_curve_file, dpi=300)
    plt.close()
    print(f"Precision-Recall 곡선 저장: {pr_curve_file}")

    # PR 곡선 관련 정보 출력
    print(f"Average Precision (PR-AUC): {average_precision:.4f}")
    print(f"Baseline (무작위 분류기): {baseline:.4f}")

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
    
    # Learning Curve 추가
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, valid_scores = learning_curve(
        lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs),
        X_train, y_train, cv=5, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=n_jobs
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
    
    # Calibration Curve 추가
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='s', markersize=5, label='LightGBM')
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

    # LightGBM 모델 시각화 - 'gain' 기준으로 변경
    lgb.plot_importance(lgb_model, figsize=(10, 6), importance_type='gain', 
                       title='LightGBM Feature Importance (Gain)')
    lgb_imp_file = os.path.join(output_dir, f'lgb_feature_importance_{timestamp}.png')
    plt.savefig(lgb_imp_file)
    plt.close()
    print(f"LightGBM 특성 중요도 그래프 저장: {lgb_imp_file}")
    
    # LightGBM 트리 시각화 - 선택적 기능
    try:
        lgb.plot_tree(lgb_model, tree_index=0, figsize=(15, 10),
                    show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
        tree_file = os.path.join(output_dir, f'lgb_tree_visualization_{timestamp}.png')
        plt.savefig(tree_file, dpi=300)
        plt.close()
        print(f"LightGBM 트리 시각화 저장: {tree_file}")
    except ImportError:
        print("graphviz 라이브러리가 설치되지 않아 트리 시각화를 생성할 수 없습니다.")
        print("트리 시각화를 원하시면 'pip install graphviz'로 라이브러리를 설치하고 시스템에 graphviz를 설치하세요.")
    except Exception as e:
        print(f"트리 시각화 생성 중 오류 발생: {str(e)}")
    
    # SHAP 분석 수정 - 간소화 버전
    try:
        print("SHAP 분석 시작 (버전 0.47.2)...")
        
        # 테스트 샘플 준비
        shap_sample_size = min(500, len(X_test))
        X_test_sample = X_test.iloc[:shap_sample_size].copy()
        print(f"샘플 크기: {shap_sample_size}, 특성 수: {X_test_sample.shape[1]}")
        
        # SHAP Explainer 생성
        print("TreeExplainer 생성 중...")
        explainer = shap.TreeExplainer(lgb_model)
        
        # SHAP 값 계산
        print("SHAP 값 계산 중...")
        shap_values = explainer(X_test_sample)
        print(f"SHAP 값 타입: {type(shap_values)}")
        print(f"SHAP 값 형태: {shap_values.shape if hasattr(shap_values, 'shape') else '형태 정보 없음'}")
        
        # 클래스 1(관심 클래스)에 대한 SHAP 값만 추출 - 다차원인 경우
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
            print(f"다차원 SHAP 값 감지: {shap_values.values.shape}")
            # 클래스 1(양성)의 SHAP 값만 사용
            class_idx = 1
            # 원본 설명 객체에서 클래스 1에 대한 값만 사용하여 새 설명 객체 생성
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
        
        # 1. Summary Plot
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
        plt.figure(figsize=(14, 10))
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

    # 상위 10개 특성 출력 - 'gain' 기준으로 변경
    top_features = [(X_train.columns[i], feature_importances[i]) for i in sorted_idx[:10]]
    print("상위 10개 특성 (Gain 기준):")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")

    # 특성 중요도를 CSV로 저장 - 'gain' 기준으로 변경
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    imp_csv = os.path.join(output_dir, f'feature_importance_gain_{timestamp}.csv')
    importance_df.to_csv(imp_csv, index=False)
    print(f"특성 중요도 CSV 저장: {imp_csv}")

    # 임계값에 따른 성능 지표 변화 시각화
    print("임계값별 성능 지표 분석 중...")
    plt.figure(figsize=(10, 6))

    # 임계값 범위 설정
    thresholds = np.linspace(0.01, 0.99, 50)
    threshold_metrics = {
        'threshold': thresholds,
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': []
    }

    # 각 임계값별 성능 지표 계산
    for threshold in thresholds:
        y_pred_th = (y_test_proba >= threshold).astype(int)
        threshold_metrics['precision'].append(precision_score(y_test, y_pred_th, zero_division=0))
        threshold_metrics['recall'].append(recall_score(y_test, y_pred_th))
        threshold_metrics['f1'].append(f1_score(y_test, y_pred_th))
        threshold_metrics['accuracy'].append(accuracy_score(y_test, y_pred_th))

    # 지표별 색상 설정
    metrics_colors = {
        'precision': 'blue',
        'recall': 'green',
        'f1': 'red',
        'accuracy': 'purple'
    }

    # 임계값별 성능 지표 그래프 그리기
    for metric, color in metrics_colors.items():
        plt.plot(threshold_metrics['threshold'], threshold_metrics[metric], 
                 label=metric, color=color, linewidth=2)

    # 현재 모델에서 사용 중인 기본 임계값(0.5) 표시
    plt.axvline(x=0.5, color='black', linestyle='--', 
                label='Default Threshold (0.5)')

    # 최적 F1 점수에 해당하는 임계값 찾기
    best_f1_idx = np.argmax(threshold_metrics['f1'])
    best_f1_threshold = threshold_metrics['threshold'][best_f1_idx]
    plt.axvline(x=best_f1_threshold, color='orange', linestyle=':', 
                label=f'Best F1 Threshold ({best_f1_threshold:.2f})')

    plt.title('Performance Metrics at Different Thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    threshold_plot_file = os.path.join(output_dir, f'threshold_metrics_{timestamp}.png')
    plt.savefig(threshold_plot_file, dpi=300)
    plt.close()
    print(f"임계값별 지표 플롯 저장: {threshold_plot_file}")

    # 최적 임계값 정보 출력
    print(f"최적 F1 임계값: {best_f1_threshold:.4f} (F1={threshold_metrics['f1'][best_f1_idx]:.4f})")

    # 교차 검증 기반 ROC/PR AUC 분포 시각화
    print("교차 검증 성능 분포 분석 중...")

    # 교차 검증 점수 계산
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = []
    pr_auc_scores = []

    # 간소화된 LightGBM 모델로 교차 검증 수행
    for train_idx, val_idx in cv.split(X_scaled.drop(columns=['query'], errors='ignore'), y):
        X_cv_train = X_scaled.drop(columns=['query'], errors='ignore').iloc[train_idx]
        y_cv_train = y.iloc[train_idx]
        X_cv_val = X_scaled.drop(columns=['query'], errors='ignore').iloc[val_idx]
        y_cv_val = y.iloc[val_idx]
        
        # 더 가벼운 LightGBM 모델 사용
        cv_classifier = lgb.LGBMClassifier(
            n_estimators=100,
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
    plt.boxplot(boxplot_data, labels=['ROC-AUC', 'PR-AUC'], 
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

    # 모델 학습 시간 계산
    training_time = time.time() - start_time
    
    # 모델 결과를 표준 형식으로 저장
    cv_scores = np.mean(roc_auc_scores)  # 또는 f1 점수 등을 저장해도 됨
    save_model_results(
        lgb_model, 
        y_test, 
        y_test_pred, 
        y_test_proba, 
        {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'roc_auc': test_auc,
            'pr_auc': pr_auc,
            'mcc': mcc,
            'log_loss': ll
        },
        importance_df,
        cm,
        roc_auc_scores,  # None 대신 실제 cv 점수 배열 전달
        training_time,
        best_params,
        output_dir,
        'LightGBM'
    )

    # 최적 모델 저장 (타임스탬프 포함)
    joblib.dump(lgb_model, model_path)
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
        
        f.write(f"\n테스트 세트 성능: Accuracy={test_accuracy:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}\n")
        
        f.write("\n추가 성능 지표:\n")
        f.write(f"  PR-AUC: {pr_auc:.4f}\n")
        f.write(f"  MCC: {mcc:.4f}\n")
        f.write(f"  Log Loss: {ll:.4f}\n")
        
        f.write("\n분류 보고서:\n")
        f.write(class_report)
        
        f.write("\n상위 10개 특성:\n")
        for feature, importance in top_features:
            f.write(f"  {feature}: {importance:.4f}\n")

        f.write(f"\n조기 중단 정보:\n")
        f.write(f"  최종 반복 횟수: {lgb_model.best_iteration_}\n")
        f.write(f"  설정된 early_stopping_rounds: 50\n")

    print(f"모델 평가 결과 저장: {results_file}")

except Exception as e:
    print(f"오류 발생: {str(e)}")
    # 오류 로깅
    with open(os.path.join(output_dir, f'error_log_{timestamp}.txt'), 'w') as f:
        f.write(f"오류 발생: {str(e)}\n")

finally:
    print("\n모델 사용 예시 (이진 분류):")
    print(f"loaded_model = joblib.load('{model_path}')")
    print(f"loaded_scaler = joblib.load('{scaler_path}')")
    print("scaled_data = loaded_scaler.transform(new_data[numeric_features])")
    print("predictions = loaded_model.predict(scaled_data)  # 클래스 예측 (0 또는 1)")
    print("probabilities = loaded_model.predict_proba(scaled_data)[:, 1]  # 클래스 1의 확률")

    # 변수가 정의되지 않은 경우를 처리
    if 'model_path' not in locals():
        model_path = os.path.join(output_dir, f'model_error_{timestamp}.txt')
    if 'scaler_path' not in locals():
        scaler_path = os.path.join(output_dir, f'scaler_error_{timestamp}.txt')
